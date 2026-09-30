"""
Evalúa el modelo Mask R-CNN de forma cuantitativa, calculando las
métricas estándar de detección/segmentación: mAP (mean Average Precision).

"""



import contextlib
import io
import numpy as np
import torch
from PIL import Image
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask

from dataset import FashionpediaDataset
from model import get_model, NUM_CLASSES



CHECKPOINT = "checkpoints/best.pth"
IMAGES_DIR = "dataset/provisional_test_no_humans"
ANNOTATIONS_FILE = "dataset/instances_provisional_test_no_humans.json"
IMAGE_SIZE = 256
SCORE_THRESHOLD = 0.05   # umbral bajo: dejamos que COCOeval decida los buenos/malos


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def load_checkpoint(path):
    model = get_model(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def mask_to_rle(binary_mask):
    """
    Convierte una máscara binaria (numpy array H x W, valores 0/1) al
    formato RLE que espera pycocotools/COCOeval, el mismo formato que
    ya habíamos visto al construir las máscaras en dataset.py.
    """
    rle = coco_mask.encode(__import__("numpy").asfortranarray(binary_mask.astype("uint8")))
    rle["counts"] = rle["counts"].decode("utf-8")  # COCOeval espera texto, no bytes
    return rle


@torch.no_grad()
def collect_predictions(model, dataset, device, index_to_category_id):
    """
    Recorre todo el dataset, ejecuta el modelo sobre cada imagen, y
    devuelve una lista de predicciones en el formato exacto que espera
    COCOeval: una lista de diccionarios, uno por cada objeto detectado.
    """
    predictions = []

    for idx in range(len(dataset)):
        image_tensor, target = dataset[idx]
        image_id = int(target["image_id"].item())

        # Para el entrenamiento con el modelo
        # habíamos hecho un resize de todas las imágenes a 256x256,
        # sin embargo, hay que deshacer ese reescalado
        # antes de comparar contra dataset.coco (que usa el tamaño original).
        img_info = dataset.coco.loadImgs(image_id)[0]
        original_width, original_height = img_info["width"], img_info["height"]
        scale_x = original_width / dataset.image_size
        scale_y = original_height / dataset.image_size

        image_tensor = image_tensor.to(device)
        output = model([image_tensor])[0]

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        masks = output["masks"].squeeze(1).cpu().numpy()  # (N, H, W), valores de probabilidad

        for i in range(len(scores)):
            # descartamos las prendas predichas con una certeza menor a SCORE_THRESHOLD
            if scores[i] < SCORE_THRESHOLD:
                continue

            # reescalamos la bounding box de vuelta al tamaño original
            x_min, y_min, x_max, y_max = boxes[i]
            x_min, x_max = x_min * scale_x, x_max * scale_x
            y_min, y_max = y_min * scale_y, y_max * scale_y

            # reescalamos la máscara de vuelta al tamaño original.
            # Primero se binariza en 256x256 (>=0.5) y luego se
            # redimensiona con "nearest", para no mezclar el umbral con
            # la interpolación (igual que ya hacíais con las máscaras
            # en dataset.py).
            binary_mask_small = (masks[i] >= 0.5).astype(np.uint8)
            binary_mask = np.array(
                Image.fromarray(binary_mask_small).resize(
                    (original_width, original_height), Image.NEAREST
                )
            ).astype(bool)

            predictions.append({
                "image_id": image_id,
                "category_id": index_to_category_id[int(labels[i])],
                "bbox": [
                    float(x_min), float(y_min),
                    float(x_max - x_min), float(y_max - y_min),  # COCO usa [x, y, ancho, alto]
                ],
                "score": float(scores[i]),
                "segmentation": mask_to_rle(binary_mask),
            })

        # print informativo de cuántas imágenes lleva procesadas/redimensionadas
        if (idx + 1) % 50 == 0 or idx == len(dataset) - 1:
            print(f"  Procesadas {idx + 1}/{len(dataset)} imágenes")

    return predictions


def run_coco_eval(coco_gt, predictions, iou_type):
    """
    Ejecuta la evaluación oficial de COCO para un tipo de IoU concreto:
    'bbox' (cajas) o 'segm' (máscaras píxel a píxel).
    """
    coco_dt = coco_gt.loadRes(predictions)  # carga las predicciones del modelo
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()

    # no mostramos la tabla grande "coco_eval.summarize", si no un resumen más corto y claro con coco_eval.stats
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()

    stats = coco_eval.stats
    tipo = "cajas" if iou_type == "bbox" else "máscaras"

    print(f"\n--- Métricas ({tipo}) ---")
    print(f"  mAP (IoU 0.50:0.95): {stats[0]:.3f}")
    print(f"  mAR  (IoU 0.50:0.95): {stats[8]:.3f}")

    # AP por clase (IoU 0.50:0.95).
    precision = coco_eval.eval["precision"]
    category_ids = coco_eval.params.catIds

    print(f"\n  AP por clase ({tipo}):")
    for k, category_id in enumerate(category_ids):
        class_precision = precision[:, :, k, 0, -1]
        valid = class_precision > -1        # -1 = esa clase no tenía ejemplos para este umbral
        ap = class_precision[valid].mean() if valid.any() else float("nan")
        name = coco_gt.loadCats(int(category_id))[0]["name"]
        print(f"    {name:<30} {ap:.3f}")

    return coco_eval


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataset = FashionpediaDataset(IMAGES_DIR, ANNOTATIONS_FILE, image_size=IMAGE_SIZE)
    print(f"Imágenes de evaluación: {len(dataset)}")

    # índice de clase limpio -> category_id original del JSON
    index_to_category_id = {v: k for k, v in dataset.categoryid_to_index.items()}

    model = load_checkpoint(CHECKPOINT)
    model.to(device)

    print("\nEjecutando el modelo sobre el conjunto de evaluación...")
    predictions = collect_predictions(model, dataset, device, index_to_category_id)
    print(f"Total de detecciones (score >= {SCORE_THRESHOLD}): {len(predictions)}")

    if len(predictions) == 0:
        print("\nNo se generó ninguna predicción. Revisa CHECKPOINT y SCORE_THRESHOLD.")
        return

    # COCOeval necesita las predicciones en un archivo con este formato exacto
    coco_gt = dataset.coco  # el objeto COCO que ya carga las anotaciones reales (ground truth)

    run_coco_eval(coco_gt, predictions, iou_type="bbox")
    run_coco_eval(coco_gt, predictions, iou_type="segm")
    

if __name__ == "__main__":
    main()
