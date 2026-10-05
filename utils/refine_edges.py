"""
Segmenta prendas con Mask R-CNN, refina las máscaras con SAM 2 y agrupa
cada prenda con sus partes (se suman) y sus cierres (se restan).

pip install "git+https://github.com/facebookresearch/sam2.git"
"""

import os
import random

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Añadimos la raíz del proyecto (carpeta padre de utils/) 
# al path para que funcionen los imports de src
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.dataset import FashionpediaDataset
from src.model import load_model, NUM_CLASSES


CHECKPOINT = "checkpoints/best.pth"
IMAGES_DIR = "dataset/provisional_test_no_humans"
ANNOTATIONS_FILE = "dataset/instances_provisional_test_no_humans.json"
IMAGE_SIZE = 256
SCORE_PRENDA = 0.5    # score mínimo para las prendas
SCORE_OTRAS = 0.3     # score mínimo para partes y cierres
SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"

# prendas a colorear
PRENDAS = {
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest",
    "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape",
}
# partes que se suman a la prenda a la que pertenecen
PARTES = {"hood", "collar", "lapel", "epaulette", "sleeve", "pocket", "ruffle"}
# se restan de la prenda para no colorearlos
CIERRES = {"zipper", "buckle", "rivet", "bead", "sequin", "applique"}


def contenido_en(a, b):
    """Fracción de la máscara `a` que cae dentro de la máscara `b`."""
    return (a & b).sum() / max(a.sum(), 1)


@torch.no_grad()
def segmentar_prendas(model, predictor, image, device, label2name):
    """
    imagen: PIL.Image RGB (cualquier tamaño).
    Devuelve una lista de prendas (de mayor a menor score). Cada una es un diccionario con:
      nombre, score, mascara (bool 256x256 = prenda + partes - cierres), partes (nombres), cierres (cuántos).
    """
    # resize a 256x256 y valores 0-1
    image = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255).to(device)

    # Mask R-CNN detecta prendas, partes y cierres
    model_output = model([tensor])[0]

    # SAM 2 refina cada instancia usando su caja como prompt
    predictor.set_image(image)
    prendas, partes, cierres = [], [], []
    for box, label, score in zip(model_output["boxes"], model_output["labels"], model_output["scores"]):
        name = label2name[int(label)]
        if name in PRENDAS and score >= SCORE_PRENDA:
            lista = prendas
        elif name in PARTES and score >= SCORE_OTRAS:
            lista = partes
        elif name in CIERRES and score >= SCORE_OTRAS:
            lista = cierres
        else:
            continue
        masks, _, _ = predictor.predict(box=box.cpu().numpy(), multimask_output=False)
        lista.append({"name": name, "score": float(score), "mask": masks[0].astype(bool),
                      "partes": [], "cierres": 0})

    # cada parte se suma a la prenda que más la contiene (si al menos el 50% está dentro)
    for parte in partes:
        if not prendas:
            break
        contenido = [contenido_en(parte["mask"], prenda["mask"]) for prenda in prendas]
        mejor = int(np.argmax(contenido))
        if contenido[mejor] >= 0.5:
            prendas[mejor]["mask"] |= parte["mask"]
            prendas[mejor]["partes"].append(parte["name"])

    # los cierres que caen dentro de una prenda (al menos el 20%) se restan de ella
    for cierre in cierres:
        for prenda in prendas:
            if contenido_en(cierre["mask"], prenda["mask"]) >= 0.2:
                prenda["mask"] &= ~cierre["mask"]
                prenda["cierres"] += 1

    return sorted(prendas, key=lambda p: -p["score"])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FashionpediaDataset(IMAGES_DIR, ANNOTATIONS_FILE, image_size=IMAGE_SIZE)

    # índice de clase del modelo (1..46) -> nombre de la categoría
    nombres = {c["id"]: c["name"] for c in dataset.coco.loadCats(dataset.coco.getCatIds())}
    label2name = {i: nombres[cat_id] for cat_id, i in dataset.categoryid_to_index.items()}

    model = load_model(CHECKPOINT, num_classes=NUM_CLASSES).to(device).eval()
    predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL_ID, device=device.type)

    # imagen aleatoria del dataset
    idx = random.randint(0, len(dataset) - 1)
    info = dataset.coco.imgs[dataset.image_ids[idx]]
    imagen = Image.open(os.path.join(IMAGES_DIR, info["file_name"])).convert("RGB")

    prendas = segmentar_prendas(model, predictor, imagen, device, label2name)

    # original y prendas finales (prenda + partes - cierres) en rojo
    original = np.array(imagen.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
    con_mascaras = original.copy()
    for p in prendas:
        con_mascaras[p["mask"]] = (0.5 * con_mascaras[p["mask"]] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        print(f'{p["name"]} ({p["score"]:.2f}): +{len(p["partes"])} partes, -{p["cierres"]} cierres')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(con_mascaras)
    axes[1].set_title("Prendas: SAM 2 + partes - cierres")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
