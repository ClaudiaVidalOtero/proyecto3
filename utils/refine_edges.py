"""
Refina con SAM 2 el output de Mask R-CNN y agrupa cada prenda con sus
partes (se suman) y sus cierres (se restan).

"""

import numpy as np
import torch
from PIL import Image

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
    """Fracción de la máscara 'a' que cae dentro de la máscara 'b'."""
    return (a & b).sum() / max(a.sum(), 1)


@torch.no_grad()
def refine_with_sam2(
    model_output,
    image,
    predictor,
    label2name,
    score_prenda=0.5,
    score_otras=0.3,
    min_parte_dentro=0.5,
    min_cierre_dentro=0.2,
    color=(255, 0, 0),
    alpha=0.5,
):
    """
    Parámetros
    ----------
    model_output : dict de Mask R-CNN (`model([tensor])[0]`) con boxes, labels y scores.
    image        : imagen RGB (PIL.Image o np.ndarray HxWx3 uint8) EXACTAMENTE
                   la misma que se le pasó al modelo (mismo tamaño), porque las
                   cajas están en sus coordenadas.
    predictor    : SAM2ImagePredictor ya cargado.
    label2name   : dict {índice de clase del modelo: nombre de categoría}.

    Devuelve
    --------
    refined_image : np.ndarray HxWx3 uint8 con las prendas (prenda + partes - cierres) coloreadas.
    prendas         : lista de dicts (de mayor a menor score) ya agrupados, con
                      name, label, score, mask (bool HxW), partes (nombres), cierres (cuántos).
    instancias      : lista de dicts (name, label, score, mask) con prendas, partes y
                      cierres refinados con SAM 2 pero SIN agrupar.
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    # SAM 2 refina cada instancia usando su caja como prompt
    predictor.set_image(image)
    instancias = []
    prendas, partes, cierres = [], [], []
    for box, label, score in zip(model_output["boxes"], model_output["labels"], model_output["scores"]):
        name = label2name[int(label)]
        if name in PRENDAS and score >= score_prenda:
            lista = prendas
        elif name in PARTES and score >= score_otras:
            lista = partes
        elif name in CIERRES and score >= score_otras:
            lista = cierres
        else:
            continue
        masks, _, _ = predictor.predict(box=box.cpu().numpy(), multimask_output=False)
        mask = masks[0].astype(bool)
        info = {"name": name, "label": int(label), "score": float(score)}
        instancias.append({**info, "mask": mask})
        lista.append({**info, "mask": mask, "partes": [], "cierres": 0})

    # cada parte se suma a la prenda que más la contiene
    for parte in partes:
        if not prendas:
            break
        contenido = [contenido_en(parte["mask"], prenda["mask"]) for prenda in prendas]
        mejor = int(np.argmax(contenido))
        if contenido[mejor] >= min_parte_dentro:
            prendas[mejor]["mask"] = prendas[mejor]["mask"] | parte["mask"]
            prendas[mejor]["partes"].append(parte["name"])

    # los cierres que caen dentro de una prenda se restan de ella
    for cierre in cierres:
        for prenda in prendas:
            if contenido_en(cierre["mask"], prenda["mask"]) >= min_cierre_dentro:
                prenda["mask"] = prenda["mask"] & ~cierre["mask"]
                prenda["cierres"] += 1

    prendas.sort(key=lambda p: -p["score"])

    # imagen con las prendas finales coloreadas
    refined_image = image.copy()
    for p in prendas:
        m = p["mask"]
        refined_image[m] = ((1 - alpha) * refined_image[m] + alpha * np.array(color)).astype(np.uint8)

    return refined_image, prendas, instancias