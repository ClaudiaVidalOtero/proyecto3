"""
probar_mascara.py
Script para leer una imagen de Fashionpedia y construir su máscara de segmentación.

"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

# rutas de las imágenes y sus archivos de anotaciones
ANNOTATIONS_FILE = "dataset/instances_attributes_train2020.json"
IMAGES_DIR = "dataset/train"

coco = COCO(ANNOTATIONS_FILE)       # cargamos el archivo de anotaciones

image_id = coco.getImgIds()[0]          # cogemos la primera imagen del dataset
img_info = coco.loadImgs(image_id)[0]
print("Imagen elegida:", img_info["file_name"])
print("Tamaño original:", img_info["width"], "x", img_info["height"])

# cargamos la imagen real desde disco
img_path = f"{IMAGES_DIR}/{img_info['file_name']}"
image = Image.open(img_path).convert("RGB")
image_np = np.array(image)

# cogemos todas las anotaciones (prendas de ropa) que aparecen en la imagen
annotations_ids = coco.getAnnIds(imgIds=image_id)
annotations = coco.loadAnns(annotations_ids)
print(f"Esta imagen tiene {len(annotations)} prendas anotadas")

for annotation in annotations:
    category_name = coco.loadCats(annotation["category_id"])[0]["name"]
    print(f"  - categoría: {category_name} (id={annotation['category_id']})")

# construimos la máscara sobre una "imagen de zeros"
height, width = img_info["height"], img_info["width"]
mask = np.zeros((height, width), dtype=np.uint8)

# ordenamos de prenda más grande a prenda más pequeña por, si se solapan varias, seleccionamos la más grande para la máscara
annotations_ordenadas = sorted(annotations, key=lambda a: a.get("area", 0), reverse=True)

for i, annotation in enumerate(annotations_ordenadas):
    category_id = annotation["category_id"]
    segmentation = annotation["segmentation"]

    # la segmentación puede venir como polígono (lista) o como RLE (dict)
    if isinstance(segmentation, list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation["counts"], list):
        rle = coco_mask.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation

    instancia = coco_mask.decode(rle).astype(bool)
    # pintamos esa zona con el id de la categoría
    mask[instancia] = category_id % 255  # solo para que se vean colores distintos

# mostramos la imagen original y la máscara al lado
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image_np)
axes[0].set_title("Imagen original")
axes[0].axis("off")

axes[1].imshow(mask, cmap="tab20")
axes[1].set_title("Máscara (cada color = una prenda)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("mask_example.png", dpi=150)
plt.show()
