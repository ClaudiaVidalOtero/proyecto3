"""
Entrenamiento del modelo

"""

import contextlib
import io
import math
import os
import time

import numpy as np
import torch
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Subset

from . import config as C  # hiperparámetros y rutas (BATCH_SIZE, EPOCHS, LEARNING_RATE...)
from .dataset import unwrap


def collate_fn(batch):
    # cada imagen tiene un tamaño distinto y distinto nº de prendas, así que no se
    # pueden apilar en un tensor. Con lo cual, devolvemos listas
    images, targets = zip(*batch)
    return list(images), list(targets)


def subset(dataset, limit):
    # si no hay límite (o es mayor que el dataset) devolvemos todo. Si no, nos quedamos
    # con las primeras 'limit' imágenes (útil para probar rápido sin entrenar con todo)
    return dataset if limit is None or limit >= len(dataset) else Subset(dataset, range(limit))


def to_device(targets, device):
    # mueve todos los tensores de cada diccionario target a la GPU
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def save_atomic(obj, path):
    """Guarda en un temporal y renombra, para no dejar un .pth corrupto."""
    # si se interrumpe a mitad de torch.save, solo se corrompe el .tmp
    # y el checkpoint bueno sigue intacto.
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


# EVALUACIÓN
@torch.no_grad()  # no hace falta calcular gradientes al evaluar, ahorra memoria
def evaluate_loss(model, loader, device, use_amp=False):
    model.train()  # Mask R-CNN solo devuelve pérdidas en modo train
    for m in model.modules():  # ponemos BatchNorm en eval para no tocar sus estadísticas
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    total = 0.0
    for images, targets in loader:
        images = [i.to(device) for i in images]
        targets = to_device(targets, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            # model(...) devuelve un dict con varias pérdidas (clasificador, cajas, máscaras...);
            # las sumamos todas y .item() lo pasa a float normal
            total += sum(model(images, targets).values()).item()
    return total / max(len(loader), 1)  # media por batch (el max evita dividir entre 0)


def _rle(mask_u8):
    # pycocotools necesita las máscaras en orden Fortran (por columnas), de ahí el asfortranarray
    rle = coco_mask.encode(np.asfortranarray(mask_u8))
    # el counts sale en bytes y el formato COCO lo quiere como string
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


@torch.no_grad()
def evaluate_map(model, dataset, device, batch_size=4, workers=0, verbose=True):
    """Devuelve (metrics, segm_eval). metrics = {segm_AP, segm_AP50, bbox_AP, bbox_AP50}."""
    model.eval()  # ahora sí modo eval: devuelve predicciones, no pérdidas
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, collate_fn=collate_fn)

    # aquí montamos "a mano" el ground truth y las detecciones en formato COCO
    # para poder usar COCOeval sin tener un JSON de verdad
    images_gt, anns_gt, dets_segm, dets_bbox = [], [], [], []
    ann_id = 1  # cada anotación necesita un id único
    for images, targets in loader:
        outputs = model([im.to(device) for im in images])
        for im, tgt, out in zip(images, targets, outputs):
            image_id = int(tgt["image_id"])
            h, w = im.shape[-2:]  # el tensor es CHW, así que cogemos las 2 últimas dims
            images_gt.append({"id": image_id, "height": int(h), "width": int(w)})

            # el ground truth es cada anotación de una prenda real
            for m, b, label in zip(tgt["masks"].numpy(), tgt["boxes"].tolist(), tgt["labels"].tolist()):
                # COCO necesita las bounding boxes como [x, y, ancho, alto], pero nuestras cajas
                # son [x1, y1, x2, y2], por eso restamos
                anns_gt.append({"id": ann_id, "image_id": image_id, "category_id": int(label),
                                "segmentation": _rle(m.astype(np.uint8)), "area": float(m.sum()),
                                "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "iscrowd": 0})
                ann_id += 1

            # predicciones del modelo
            # las máscaras salen como probabilidades (N,1,H,W); [:, 0] quita el canal
            # y > 0.5 las binariza
            pmasks = (out["masks"][:, 0] > 0.5).cpu().numpy().astype(np.uint8)
            for m, b, s, label in zip(pmasks, out["boxes"].cpu().tolist(),
                                      out["scores"].cpu().tolist(), out["labels"].cpu().tolist()):
                # guardamos cada detección dos veces: una para evaluar máscaras y otra para cajas
                dets_segm.append({"image_id": image_id, "category_id": int(label),
                                  "score": float(s), "segmentation": _rle(m)})
                dets_bbox.append({"image_id": image_id, "category_id": int(label), "score": float(s),
                                  "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]})

    # métricas a 0 por defecto, por si no hay nada que evaluar
    metrics = {"segm_AP": 0.0, "segm_AP50": 0.0, "bbox_AP": 0.0, "bbox_AP50": 0.0}
    if not anns_gt or not dets_segm:
        # puede pasar que no haya segmentaciones al principio del 
        # entrenamiento si el modelo aún no detecta nada
        print("Aviso: no hay anotaciones o detecciones que evaluar (mAP = 0).")
        return metrics, None

    names = unwrap(dataset).label_map()
    # creamos un objeto COCO vacío y le metemos el dataset a mano
    gt = COCO()
    gt.dataset = {"images": images_gt, "annotations": anns_gt,
                  "categories": [{"id": i, "name": names[i]} for i in sorted(names)]}
    # createIndex imprime cosas por pantalla, así que silenciamos stdout un momento
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()

    segm_eval = None
    for iou_type, dets in (("bbox", dets_bbox), ("segm", dets_segm)):
        # COCOeval imprime una tabla enorme. La capturamos en un buffer y
        # solo la mostramos si verbose=True
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ev = COCOeval(gt, gt.loadRes(dets), iou_type)
            ev.evaluate(); ev.accumulate(); ev.summarize()  # los 3 pasos que pide COCOeval, siempre en este orden
        if verbose:
            print(f"===== {iou_type} =====\n{buf.getvalue()}")
        # stats[0] = AP@[.50:.95] (el mAP "oficial"), stats[1] = AP con IoU 0.5
        metrics[f"{iou_type}_AP"] = float(ev.stats[0])
        metrics[f"{iou_type}_AP50"] = float(ev.stats[1])
        if iou_type == "segm":
            segm_eval = ev  # nos guardamos este para sacar luego el AP por clase
    return metrics, segm_eval


def ap_por_clase(coco_eval, names):
    """AP@[.50:.95] por categoría. names: {índice: nombre}. NaN si no hay GT de esa clase."""
    precision = coco_eval.eval["precision"]  # [IoU, recall, clase, área, maxDets]
    out = {}
    for k, cat_id in enumerate(coco_eval.params.catIds):
        # nos quedamos con la clase k, área "all" (0) y el último maxDets (-1 = el mayor, 100 dets)
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]  # COCO pone -1 donde no hay datos, esos hay que tirarlos o falsean la media
        out[names[cat_id]] = float(p.mean()) if p.size else float("nan")
    return out


# ENTRENAMIENTO
def train_one_epoch(model, loader, optimizer, device, epoch, scaler, max_grad_norm=10.0, log_every=50):
    model.train()
    total = 0.0
    use_amp = scaler.is_enabled()  # si el scaler está activo es que usamos AMP
    base_lr = optimizer.param_groups[0]["lr"]  # learning rate objetivo, al que llegará tras el warm-up
    n_warm = min(C.WARMUP_ITERS, len(loader)) if epoch == 1 else 0   # warm-up solo en la 1ª época
    params = [p for g in optimizer.param_groups for p in g["params"]]  # lista plana de parámetros, para el clip

    for step, (images, targets) in enumerate(loader, start=1):
        if step <= n_warm:
            # warm-up lineal: el learning rate sube de 0.001*base hasta base poco a poco,
            # para que el principio del entrenamiento no "explote" con un learning rate muy alto
            for g in optimizer.param_groups:
                g["lr"] = base_lr * (0.001 + 0.999 * step / n_warm)

        images = [i.to(device) for i in images]
        targets = to_device(targets, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())  # pérdida total = suma de todas las parciales
        # si la pérdida se va a NaN o infinito paramos ya, no tiene sentido seguir entrenando
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"Pérdida no finita (época {epoch}, paso {step}): {loss_dict}. Baja LEARNING_RATE.")

        # set_to_none=True es un poco más eficiente que poner los gradientes a 0
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            # con AMP hay que escalar la pérdida para que los gradientes en fp16 no se
            # queden en 0. Antes de hacer clip se desescalan (unscale_) para que
            # el clip se aplique sobre los valores reales
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)  # recorta gradientes enormes (evita explosiones)
            scaler.step(optimizer)
            scaler.update()
        else:
            # lo mismo pero sin AMP, más simple
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()

        total += loss.item()
        # imprimimos cada log_every pasos y también en el último
        if step % log_every == 0 or step == len(loader):
            print(f"Época {epoch} | paso {step}/{len(loader)} | pérdida: {loss.item():.4f}")
    return total / max(len(loader), 1)  # pérdida media de la época


def train_model(model, train_ds, val_ds, device, resume=True):
    # recortamos los datasets si hay límites en el config (para pruebas rápidas)
    train_ds = subset(train_ds, C.MAX_TRAIN_IMAGES)
    val_ds = subset(val_ds, C.MAX_VAL_IMAGES)
    map_ds = subset(val_ds, C.MAP_VAL_IMAGES)  # el mAP es lento, así que se calcula con un trozo aún más pequeño

    # kwargs compartidos por los dos loaders (así no repetimos código)
    kw = dict(batch_size=C.BATCH_SIZE, num_workers=C.WORKERS, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    # solo optimizamos los parámetros que no estén congelados
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=C.LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    # cada LR_STEP_SIZE épocas el learning rate se multiplica por 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=C.LR_STEP_SIZE, gamma=0.1)
    # AMP solo funciona bien en CUDA, por eso se comprueba el device
    amp_on = bool(C.USE_AMP and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    last_path = os.path.join(C.CHECKPOINT_DIR, "last.pth")  # último checkpoint (para reanudar)
    best_path = os.path.join(C.CHECKPOINT_DIR, "best.pth")  # el de mejor mAP

    best_map = -1.0  # -1 para que la primera época siempre cuente como "mejor"
    history = {"train_loss": [], "val_loss": [], "val_map": [], "val_map50": []}
    start_epoch = 1

    # reanudar entrenamiento si existe un checkpoint (útil para Google Colab)
    if resume and os.path.isfile(last_path):
        # weights_only=True por seguridad
        ckpt = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # los "if ... in ckpt" son para que sigan valiendo checkpoints antiguos que no guardaban esto
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if amp_on and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_map = ckpt.get("best_map", -1.0)
        history = ckpt.get("history", history)
        # setdefault añade las claves de mAP si el checkpoint viejo no las tenía
        for k in ("val_map", "val_map50"):
            history.setdefault(k, [])
        start_epoch = ckpt["epoch"] + 1  # seguimos por la época siguiente a la guardada
        print(f"Reanudando desde la época {start_epoch} (mejor mAP={best_map:.4f})")
        if start_epoch > C.EPOCHS:
            print("El entrenamiento ya estaba completo.")
            return history

    # bucle principal de epochs
    for epoch in range(start_epoch, C.EPOCHS + 1):
        t0 = time.time()  # para medir cuánto tarda cada epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler)
        val_loss = evaluate_loss(model, val_loader, device, use_amp=amp_on)
        # el _ es porque aquí no nos hace falta el objeto segm_eval, solo las métricas
        metrics, _ = evaluate_map(model, map_ds, device, batch_size=C.BATCH_SIZE,
                                  workers=C.WORKERS, verbose=False)
        scheduler.step()  # actualizamos el learning rate al final de la epoch (después del optimizer)
        print(f"Época {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f} | "
              f"segm AP={metrics['segm_AP']:.4f} (AP50={metrics['segm_AP50']:.4f}) | "
              f"bbox AP={metrics['bbox_AP']:.4f} | {(time.time() - t0) / 60:.1f} min")

        # guardamos las métricas de esta epoch para poder graficarlas luego
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map"].append(metrics["segm_AP"])
        history["val_map50"].append(metrics["segm_AP50"])

        # el mejor modelo se decide por mAP de segmentación, no por la val_loss
        is_best = metrics["segm_AP"] > best_map
        if is_best:
            best_map = metrics["segm_AP"]

        # metemos en el checkpoint todo lo necesario para poder reanudar exactamente donde lo dejamos
        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": val_loss, "val_map": metrics["segm_AP"],
                "best_map": best_map, "history": history}
        save_atomic(ckpt, last_path)  # last.pth se guarda siempre
        if is_best:
            save_atomic(ckpt, best_path)  # best.pth solo si hemos mejorado
            print(f"  → Nuevo mejor modelo guardado (segm AP={best_map:.4f})")
    return history
