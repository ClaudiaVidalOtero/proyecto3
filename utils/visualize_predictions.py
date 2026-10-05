"""
Muestra por pantalla la segmentación generada por el modelo 
sobre una imagen aleatoria del dataset, al lado de
la segmentación real (ground truth) de la misma imagen.

"""


import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

# Añadimos la raíz del proyecto (carpeta padre de utils/) 
# al path para que funcionen los imports de src
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.dataset import FashionpediaDataset
from src.model import get_model, NUM_CLASSES


CHECKPOINT = "checkpoints/best.pth"
IMAGES_DIR = "dataset/provisional_test_no_humans"
ANNOTATIONS_FILE = "dataset/instances_provisional_test_no_humans.json"
IMAGE_SIZE = 256          # el mismo que usamos al entrenar
THRESHOLD = 0.3           # score mínimo para mostrar una predicción
OUTPUT = "checkpoints/prediction_example.png"


# 60 colores distintos (tab20 + tab20b + tab20c) para poder dar un color
# fijo a cada clase: así "camisa" tiene el mismo color en la anotación
# real y en la predicción, y se pueden comparar a simple vista.
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


def draw_instances(axis, image, labels, scores, names, masks, threshold=0.0, alpha=0.5):
    """
    Dibuja las máscaras de segmentación píxel a píxel sobre la imagen
    y añadde la etiqueta (clase + score) en el centro de cada máscara
    """
    overlay = image.astype(np.float32).copy()

    # nos quedamos con las instancias que superan el umbral, ordenadas de
    # menor a mayor score, para que las más seguras se pinten encima
    kept = [i for i in range(len(labels)) if scores[i] >= threshold]
    kept = sorted(kept, key=lambda i: scores[i])

    binary_masks = {}
    for i in kept:
        binary = masks[i] >= 0.5          # máscara binaria: True = píxel de la prenda
        if not binary.any():
            continue
        binary_masks[i] = binary
        color = CLASS_COLORS[int(labels[i]) % len(CLASS_COLORS)]
        # mezcla del color con la imagen solo en los píxeles de la máscara
        overlay[binary] = (1 - alpha) * overlay[binary] + alpha * color

    axis.imshow(np.clip(overlay, 0, 1))

    for i, binary in binary_masks.items():
        color = CLASS_COLORS[int(labels[i]) % len(CLASS_COLORS)]
        # contorno de la máscara
        axis.contour(binary.astype(float), levels=[0.5], colors=[color], linewidths=1.2)
        # etiqueta en el centro de la máscara
        ys, xs = np.nonzero(binary)
        axis.text(
            xs.mean(),
            ys.mean(),
            f"{names.get(int(labels[i]), int(labels[i]))} ({scores[i]:.2f})",
            color="white",
            backgroundcolor="black",
            fontsize=6,
            ha="center",
            va="center",
        )
    axis.axis("off")


def main():
    # cargamos imágenes del dataset
    dataset = FashionpediaDataset(
        IMAGES_DIR,
        ANNOTATIONS_FILE,
        image_size=IMAGE_SIZE,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"El dataset está vacío. Revisa IMAGES_DIR y ANNOTATIONS_FILE.")

    # seleccionamos imagen aleatoria
    index = random.randint(0, len(dataset) - 1)
    print(f"Imagen elegida: índice {index} (de {len(dataset)})")
    image_tensor, target = dataset[index]

    # cargamos el checkpoint del modelo
    model = load_checkpoint(CHECKPOINT)
    with torch.no_grad():
        prediction = model([image_tensor])[0]

    # obtenemos nombres de categorías ("shirt". "pants"...) a partir de los números (23, 15...)
    names = {
        class_index: dataset.coco.loadCats(category_id)[0]["name"]
        for category_id, class_index in dataset.categoryid_to_index.items()
    }

    # adaptamos formato de tensores a numpy (CANALES, ALTO, ANCHO) -> (ALTO, ANCHO, CANALES)
    image = image_tensor.permute(1, 2, 0).numpy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))

    # dibuja máscara real
    draw_instances(
        axes[0],
        image,
        target["labels"].numpy(),
        np.ones(len(target["labels"])),
        names,
        masks=target["masks"].numpy(),
    )
    axes[0].set_title(f"Anotaciones reales (índice {index})")

    # dibuja máscar predicha por el modelo
    draw_instances(
        axes[1],
        image,
        prediction["labels"].numpy(),
        prediction["scores"].numpy(),
        names,
        masks=prediction["masks"].squeeze(1).numpy(),
        threshold=THRESHOLD,
    )
    axes[1].set_title(f"Predicción (score >= {THRESHOLD})")
    figure.tight_layout()

    # guarda la imagen de las máscaras en disco
    output_dir = os.path.dirname(OUTPUT)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    figure.savefig(OUTPUT, dpi=150)
    print(f"Imagen guardada en: {OUTPUT}")
    print(f"Detecciones totales: {len(prediction['scores'])}")
    print(f"Detecciones mostradas: {(prediction['scores'] >= THRESHOLD).sum().item()}")
    plt.show()


if __name__ == "__main__":
    main()
