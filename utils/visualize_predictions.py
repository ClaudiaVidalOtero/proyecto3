"""
Visualización del pipeline completo: Mask R-CNN -> SAM 2 -> agrupar prendas.

Sobre una imagen aleatoria del dataset (a 256x256, como en el resto de .py) muestra:
  1. Imagen original con las anotaciones reales (ground truth) y sus categorías.
  2. Segmentación suavizada con SAM 2, con las categorías detectadas (sin agrupar).
  3. Segmentación suavizada con SAM 2 después de agrupar: prenda + partes - cierres.

"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Añadimos la raíz del proyecto (carpeta padre de utils/)
# al path para que funcionen los imports de src y utils
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.dataset import FashionpediaDataset
from src.model import get_model, NUM_CLASSES
from refine_edges import refine_with_sam2


CHECKPOINT = "checkpoints/best.pth"
IMAGES_DIR = "dataset/provisional_test_no_humans"
ANNOTATIONS_FILE = "dataset/instances_provisional_test_no_humans.json"
IMAGE_SIZE = 256          # el mismo que usamos al entrenar
SCORE_PRENDA = 0.5        # score mínimo para las prendas
SCORE_OTRAS = 0.3         # score mínimo para partes y cierres
SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
OUTPUT = "checkpoints/refined_example.png"

# 60 colores distintos (tab20 + tab20b + tab20c): cada clase tiene siempre el mismo color
CLASS_COLORS = np.array(
    [plt.get_cmap(name)(i)[:3] for name in ("tab20", "tab20b", "tab20c") for i in range(20)]
)


def load_checkpoint(path):
    model = get_model(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def draw_instances(axis, image, items, alpha=0.5):
    """
    Dibuja las máscaras de segmentación píxel a píxel sobre la imagen
    y añade la etiqueta (clase + score) en el centro de cada máscara

    items: lista de diccionarios con label (índice de clase, da el color), text y mask (bool HxW).

    """
    overlay = image.astype(np.float32) / 255
    items = [item for item in items if item["mask"].any()]
    # ordenadas de menor a mayor score, para que las más seguras se pinten encima
    items = sorted(items, key=lambda item: item.get("score", 1.0))

    for item in items:
        color = CLASS_COLORS[item["label"] % len(CLASS_COLORS)]
        # mezcla del color con la imagen solo en los píxeles de la máscara
        overlay[item["mask"]] = (1 - alpha) * overlay[item["mask"]] + alpha * color

    axis.imshow(np.clip(overlay, 0, 1))

    for item in items:
        color = CLASS_COLORS[item["label"] % len(CLASS_COLORS)]
        # contorno de la máscara
        axis.contour(item["mask"].astype(float), levels=[0.5], colors=[color], linewidths=1.2)
        # etiqueta en el centro de la máscara
        ys, xs = np.nonzero(item["mask"])
        axis.text(
            xs.mean(), ys.mean(), item["text"],
            color="white", backgroundcolor="black", fontsize=6, ha="center", va="center",
        )
    axis.axis("off")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cargamos imágenes del dataset
    dataset = FashionpediaDataset(IMAGES_DIR, ANNOTATIONS_FILE, image_size=IMAGE_SIZE)
    if len(dataset) == 0:
        raise RuntimeError("El dataset está vacío. Revisa IMAGES_DIR y ANNOTATIONS_FILE.")

    # nombres de categorías ("shirt", "pants"...) a partir del índice de clase (23, 15...)
    names = {
        class_index: dataset.coco.loadCats(category_id)[0]["name"]
        for category_id, class_index in dataset.categoryid_to_index.items()
    }

    # imagen aleatoria: la original (para el modelo) y su ground truth (para el panel 1)
    index = random.randint(0, len(dataset) - 1)
    print(f"Imagen elegida: índice {index} (de {len(dataset)})")
    _, target = dataset[index]
    info = dataset.coco.imgs[dataset.image_ids[index]]
    imagen = Image.open(os.path.join(IMAGES_DIR, info["file_name"])).convert("RGB")
    image = np.array(imagen.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))

    # cargamos el checkpoint del modelo y SAM 2
    model = load_checkpoint(CHECKPOINT).to(device)
    predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL_ID, device=device.type)

    # Mask R-CNN detecta prendas, partes y cierres
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255).to(device)
    with torch.no_grad():
        model_output = model([tensor])[0]

    # SAM 2 refina y se agrupan las prendas (prenda + partes - cierres)
    _, prendas, instancias = refine_with_sam2(
        model_output, image, predictor, names,
        score_prenda=SCORE_PRENDA, score_otras=SCORE_OTRAS,
    )

    # 1) anotaciones reales
    reales = [
        {"label": int(label), "text": names[int(label)], "mask": mask.astype(bool)}
        for label, mask in zip(target["labels"].numpy(), target["masks"].numpy())
    ]
    # 2) SAM 2 sin agrupar
    without_joints = [
        {"label": i["label"], "text": f'{i["name"]} ({i["score"]:.2f})', "mask": i["mask"], "score": i["score"]}
        for i in instancias
    ]
    # 3) SAM 2 agrupado
    with_joints = [
        {"label": p["label"], "mask": p["mask"], "score": p["score"],
         "text": f'{p["name"]} ({p["score"]:.2f})  +{len(p["partes"])} partes -{p["cierres"]} cierres'}
        for p in prendas
    ]

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    draw_instances(axes[0], image, reales)
    axes[0].set_title(f"Original con anotaciones reales (índice {index})")
    draw_instances(axes[1], image, without_joints)
    axes[1].set_title("SAM 2 sin agrupar (prendas, partes y cierres)")
    draw_instances(axes[2], image, with_joints)
    axes[2].set_title("SAM 2 agrupado: prenda + partes - cierres")
    figure.tight_layout()

    # guarda la figura en disco
    output_dir = os.path.dirname(OUTPUT)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    figure.savefig(OUTPUT, dpi=150)
    print(f"Imagen guardada en: {OUTPUT}")
    print(f"Instancias sin agrupar: {len(instancias)} | prendas finales: {len(prendas)}")
    for p in prendas:
        print(f'  {p["name"]} ({p["score"]:.2f}): +{len(p["partes"])} partes, -{p["cierres"]} cierres')
    plt.show()


if __name__ == "__main__":
    main()
