"""Entrenamiento (AMP, warm-up, clip de gradientes, reanudación) y evaluación COCO (mAP)."""
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

from . import config as C
from .dataset import unwrap


# ------------------------------------------------------------------ utilidades
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def subset(dataset, limit):
    return dataset if limit is None or limit >= len(dataset) else Subset(dataset, range(limit))


def to_device(targets, device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def save_atomic(obj, path):
    """Guarda en un temporal y renombra, para no dejar un .pth corrupto."""
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------------ evaluación
@torch.no_grad()
def evaluate_loss(model, loader, device, use_amp=False):
    model.train()  # Mask R-CNN solo devuelve pérdidas en modo train
    for m in model.modules():  # pero BatchNorm en eval para no tocar sus estadísticas
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    total = 0.0
    for images, targets in loader:
        images = [i.to(device) for i in images]
        targets = to_device(targets, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            total += sum(model(images, targets).values()).item()
    return total / max(len(loader), 1)


def _rle(mask_u8):
    rle = coco_mask.encode(np.asfortranarray(mask_u8))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


@torch.no_grad()
def evaluate_map(model, dataset, device, batch_size=4, workers=0, verbose=True):
    """Devuelve (metrics, segm_eval). metrics = {segm_AP, segm_AP50, bbox_AP, bbox_AP50}."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, collate_fn=collate_fn)

    images_gt, anns_gt, dets_segm, dets_bbox = [], [], [], []
    ann_id = 1
    for images, targets in loader:
        outputs = model([im.to(device) for im in images])
        for im, tgt, out in zip(images, targets, outputs):
            image_id = int(tgt["image_id"])
            h, w = im.shape[-2:]
            images_gt.append({"id": image_id, "height": int(h), "width": int(w)})

            for m, b, label in zip(tgt["masks"].numpy(), tgt["boxes"].tolist(), tgt["labels"].tolist()):
                anns_gt.append({"id": ann_id, "image_id": image_id, "category_id": int(label),
                                "segmentation": _rle(m.astype(np.uint8)), "area": float(m.sum()),
                                "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "iscrowd": 0})
                ann_id += 1

            pmasks = (out["masks"][:, 0] > 0.5).cpu().numpy().astype(np.uint8)
            for m, b, s, label in zip(pmasks, out["boxes"].cpu().tolist(),
                                      out["scores"].cpu().tolist(), out["labels"].cpu().tolist()):
                dets_segm.append({"image_id": image_id, "category_id": int(label),
                                  "score": float(s), "segmentation": _rle(m)})
                dets_bbox.append({"image_id": image_id, "category_id": int(label), "score": float(s),
                                  "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]})

    metrics = {"segm_AP": 0.0, "segm_AP50": 0.0, "bbox_AP": 0.0, "bbox_AP50": 0.0}
    if not anns_gt or not dets_segm:
        print("Aviso: no hay anotaciones o detecciones que evaluar (mAP = 0).")
        return metrics, None

    names = unwrap(dataset).label_map()
    gt = COCO()
    gt.dataset = {"images": images_gt, "annotations": anns_gt,
                  "categories": [{"id": i, "name": names[i]} for i in sorted(names)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()

    segm_eval = None
    for iou_type, dets in (("bbox", dets_bbox), ("segm", dets_segm)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ev = COCOeval(gt, gt.loadRes(dets), iou_type)
            ev.evaluate(); ev.accumulate(); ev.summarize()
        if verbose:
            print(f"===== {iou_type} =====\n{buf.getvalue()}")
        metrics[f"{iou_type}_AP"] = float(ev.stats[0])
        metrics[f"{iou_type}_AP50"] = float(ev.stats[1])
        if iou_type == "segm":
            segm_eval = ev
    return metrics, segm_eval


def ap_por_clase(coco_eval, names):
    """AP@[.50:.95] por categoría. names: {índice: nombre}. NaN si no hay GT de esa clase."""
    precision = coco_eval.eval["precision"]  # [IoU, recall, clase, área, maxDets]
    out = {}
    for k, cat_id in enumerate(coco_eval.params.catIds):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        out[names[cat_id]] = float(p.mean()) if p.size else float("nan")
    return out


# ------------------------------------------------------------------ entrenamiento
def train_one_epoch(model, loader, optimizer, device, epoch, scaler, max_grad_norm=10.0, log_every=50):
    model.train()
    total = 0.0
    use_amp = scaler.is_enabled()
    base_lr = optimizer.param_groups[0]["lr"]
    n_warm = min(C.WARMUP_ITERS, len(loader)) if epoch == 1 else 0   # warm-up solo en la 1ª época
    params = [p for g in optimizer.param_groups for p in g["params"]]

    for step, (images, targets) in enumerate(loader, start=1):
        if step <= n_warm:
            for g in optimizer.param_groups:
                g["lr"] = base_lr * (0.001 + 0.999 * step / n_warm)

        images = [i.to(device) for i in images]
        targets = to_device(targets, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"Pérdida no finita (época {epoch}, paso {step}): {loss_dict}. Baja LEARNING_RATE.")

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()

        total += loss.item()
        if step % log_every == 0 or step == len(loader):
            print(f"Época {epoch} | paso {step}/{len(loader)} | pérdida: {loss.item():.4f}")
    return total / max(len(loader), 1)


def train_model(model, train_ds, val_ds, device, resume=True):
    train_ds = subset(train_ds, C.MAX_TRAIN_IMAGES)
    val_ds = subset(val_ds, C.MAX_VAL_IMAGES)
    map_ds = subset(val_ds, C.MAP_VAL_IMAGES)

    kw = dict(batch_size=C.BATCH_SIZE, num_workers=C.WORKERS, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=C.LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=C.LR_STEP_SIZE, gamma=0.1)
    amp_on = bool(C.USE_AMP and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    last_path = os.path.join(C.CHECKPOINT_DIR, "last.pth")
    best_path = os.path.join(C.CHECKPOINT_DIR, "best.pth")

    best_map = -1.0
    history = {"train_loss": [], "val_loss": [], "val_map": [], "val_map50": []}
    start_epoch = 1

    if resume and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if amp_on and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_map = ckpt.get("best_map", -1.0)
        history = ckpt.get("history", history)
        for k in ("val_map", "val_map50"):
            history.setdefault(k, [])
        start_epoch = ckpt["epoch"] + 1
        print(f"Reanudando desde la época {start_epoch} (mejor mAP={best_map:.4f})")
        if start_epoch > C.EPOCHS:
            print("El entrenamiento ya estaba completo.")
            return history

    for epoch in range(start_epoch, C.EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler)
        val_loss = evaluate_loss(model, val_loader, device, use_amp=amp_on)
        metrics, _ = evaluate_map(model, map_ds, device, batch_size=C.BATCH_SIZE,
                                  workers=C.WORKERS, verbose=False)
        scheduler.step()
        print(f"Época {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f} | "
              f"segm AP={metrics['segm_AP']:.4f} (AP50={metrics['segm_AP50']:.4f}) | "
              f"bbox AP={metrics['bbox_AP']:.4f} | {(time.time() - t0) / 60:.1f} min")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map"].append(metrics["segm_AP"])
        history["val_map50"].append(metrics["segm_AP50"])

        is_best = metrics["segm_AP"] > best_map
        if is_best:
            best_map = metrics["segm_AP"]

        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": val_loss, "val_map": metrics["segm_AP"],
                "best_map": best_map, "history": history}
        save_atomic(ckpt, last_path)
        if is_best:
            save_atomic(ckpt, best_path)
            print(f"  → Nuevo mejor modelo guardado (segm AP={best_map:.4f})")
    return history
