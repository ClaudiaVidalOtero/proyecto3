"""Mask R-CNN -> refinado con SAM 2 -> prenda + partes − cierres -> alfa suave por prenda."""
import cv2
import numpy as np
import torch
from PIL import Image

from . import config as C

# ---------------------------------------------------------------- categorías (edita a tu gusto)
PRENDAS = {
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest",
    "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape",
}
PARTES = {"hood", "collar", "lapel", "epaulette", "sleeve", "pocket", "ruffle"}   # se SUMAN a la prenda
CIERRES = {"zipper", "buckle", "rivet", "bead", "sequin", "applique"}              # se RESTAN
RELEVANTES = PRENDAS | PARTES | CIERRES

# Si dos prendas se solapan, la de mayor número se queda con la zona común.
PRIORIDAD = {
    "coat": 6, "cape": 6, "jacket": 5, "cardigan": 5, "vest": 4,
    "sweater": 3, "shirt, blouse": 3, "top, t-shirt, sweatshirt": 3,
    "dress": 2, "jumpsuit": 2, "pants": 1, "shorts": 1, "skirt": 1,
}


def check_categories(label2name):
    """Avisa si algún nombre de arriba no existe en tu JSON."""
    faltan = RELEVANTES - set(label2name.values())
    print("⚠️ Nombres que no existen en tu JSON:" if faltan else "✓ Categorías OK", sorted(faltan) or "")


# ---------------------------------------------------------------- 1) detección (resolución ORIGINAL)
@torch.no_grad()
def detectar(model, img_rgb, device, score_min=0.3):
    model.eval()
    H, W = img_rgb.shape[:2]
    s = min(1.0, C.MAX_SIDE / max(H, W))   # misma escala que en entrenamiento
    sw, sh = max(1, round(W * s)), max(1, round(H * s))
    small = img_rgb if s == 1.0 else np.array(
        Image.fromarray(img_rgb).resize((sw, sh), Image.Resampling.BILINEAR))
    sx, sy = sw / W, sh / H

    out = model([torch.from_numpy(small).permute(2, 0, 1).float().div(255).to(device)])[0]
    keep = out["scores"] >= score_min

    boxes = out["boxes"][keep].cpu().numpy().astype(np.float32)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] / sx).clip(0, W)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] / sy).clip(0, H)
    labels = out["labels"][keep].cpu().numpy()
    scores = out["scores"][keep].cpu().numpy()
    # se reescalan las probabilidades y luego se umbraliza: bordes más suaves
    masks = [cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
             for p in out["masks"][keep, 0].cpu().numpy()]
    return boxes, labels, scores, masks


# ---------------------------------------------------------------- 2) refinado con SAM 2
def load_sam(device_type):
    from sam2.sam2_image_predictor import SAM2ImagePredictor   # import perezoso
    predictor = SAM2ImagePredictor.from_pretrained(C.SAM2_MODEL_ID, device=device_type)
    print("SAM 2 cargado:", C.SAM2_MODEL_ID)
    return predictor


def iou_mascaras(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def punto_interior(mask):
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(dist.argmax(), dist.shape)
    return np.array([[x, y]], dtype=np.float32)


def refinar_con_sam(predictor, mask, box, iou_min=0.6):
    """Prompt = caja + punto interior. Se elige la candidata más parecida a la de Mask R-CNN;
    si ninguna llega a iou_min se conserva la de Mask R-CNN. Devuelve (máscara, usó_sam)."""
    if not mask.any():
        return mask, False
    masks, _, _ = predictor.predict(point_coords=punto_interior(mask), point_labels=np.array([1]),
                                    box=np.asarray(box, dtype=np.float32), multimask_output=True)
    cands = masks.astype(bool)
    ious = [iou_mascaras(c, mask) for c in cands]
    best = int(np.argmax(ious))
    return (cands[best], True) if ious[best] >= iou_min else (mask, False)


# ---------------------------------------------------------------- 3) prenda + partes − cierres
def contenido_en(a, b):
    area = a.sum()
    return float(np.logical_and(a, b).sum() / area) if area else 0.0


def combinar_instancias(instancias, shape, umbral_parte=0.5, umbral_cierre=0.2,
                        dilatar_cierres=2, suavizado=1.5):
    """Devuelve una entrada por prenda con mask_final (bool) y alpha (float 0-1)."""
    H, W = shape
    prendas = [dict(i, partes=[], cierres=0) for i in instancias if i["nombre"] in PRENDAS]
    partes = [i for i in instancias if i["nombre"] in PARTES]
    cierres = [i for i in instancias if i["nombre"] in CIERRES]
    if not prendas:
        return []

    for g in prendas:
        g["mask_prenda"] = g["mask"].copy()

    # cada parte se suma a la prenda que más la contiene
    for p in partes:
        cont = [contenido_en(p["mask"], g["mask"]) for g in prendas]
        j = max(range(len(prendas)), key=lambda k: (round(cont[k], 3), -int(prendas[k]["mask"].sum())))
        if cont[j] >= umbral_parte:
            prendas[j]["partes"].append(p["nombre"])
            prendas[j]["mask_prenda"] |= p["mask"]

    # se restan los cierres que caen dentro de la prenda (con una pequeña dilatación)
    r = max(1, round(dilatar_cierres * max(H, W) / 1024))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    for g in prendas:
        quitar = np.zeros((H, W), bool)
        for c in cierres:
            if contenido_en(c["mask"], g["mask_prenda"]) >= umbral_cierre:
                quitar |= c["mask"]
                g["cierres"] += 1
        if dilatar_cierres and quitar.any():
            quitar = cv2.dilate(quitar.astype(np.uint8), kernel) > 0
        g["mask_final"] = g["mask_prenda"] & ~quitar

    # solapes: gana la prenda de mayor prioridad (ocupa su área completa, antes de restar cierres)
    ocupado = np.zeros((H, W), bool)
    for g in sorted(prendas, key=lambda g: -PRIORIDAD.get(g["nombre"], 0)):
        g["mask_final"] = g["mask_final"] & ~ocupado
        ocupado |= g["mask_prenda"]

    # alfa suave
    sigma = max(1.0, suavizado * max(H, W) / 1024)
    for g in prendas:
        g["alpha"] = np.clip(cv2.GaussianBlur(g["mask_final"].astype(np.float32), (0, 0), sigma), 0, 1)

    return sorted(prendas, key=lambda g: -g["score"])


# ---------------------------------------------------------------- pipeline completo
def segmentar_prendas(det_model, predictor, img_rgb, device, label2name, usar_sam=True):
    """img_rgb: np.uint8 (H, W, 3) a resolución original. predictor puede ser None si usar_sam=False."""
    boxes, labels, scores, masks = detectar(det_model, img_rgb, device,
                                            score_min=min(C.SCORE_PRENDA, C.SCORE_OTRAS))
    candidatas = []
    for b, l, s, m in zip(boxes, labels, scores, masks):
        nombre = label2name[int(l)]
        if nombre in RELEVANTES and s >= (C.SCORE_PRENDA if nombre in PRENDAS else C.SCORE_OTRAS):
            candidatas.append((nombre, float(s), b, m))

    if usar_sam and candidatas:
        predictor.set_image(img_rgb)   # el embedding se calcula una sola vez

    instancias = []
    for nombre, s, b, m in candidatas:
        m_ref, ok = refinar_con_sam(predictor, m, b) if usar_sam else (m, False)
        instancias.append(dict(nombre=nombre, score=s, box=b, mask_rcnn=m, mask=m_ref, sam=ok))
    return combinar_instancias(instancias, img_rgb.shape[:2])
