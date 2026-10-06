"""Mask R-CNN -> refinado con SAM 2 -> prenda + partes − cierres -> alfa suave por prenda."""
import cv2
import numpy as np
import torch
from PIL import Image

from . import config as C  # umbrales SCORE_PRENDA y SCORE_OTRAS, MAX_SIDE, id del modelo SAM 2...

# cateogorías de prendas (tienen que coincidir exactamente con los nombres del JSON de Fashionpedia)
PRENDAS = {
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest",
    "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape",
}
PARTES = {"hood", "collar", "lapel", "epaulette", "sleeve", "pocket", "ruffle"}   # se SUMAN a la prenda
CIERRES = {"zipper", "buckle", "rivet", "bead", "sequin", "applique"}              # se RESTAN
RELEVANTES = PRENDAS | PARTES | CIERRES  # el | es la unión de conjuntos, o sea todo lo que nos interesa

# si dos prendas se solapan, la de mayor número se queda con la zona común.
# (por ejemplo, un abrigo encima de una camiseta, nos quedaríamos con que esa zona es abrigo)
PRIORIDAD = {
    "coat": 6, "cape": 6, "jacket": 5, "cardigan": 5, "vest": 4,
    "sweater": 3, "shirt, blouse": 3, "top, t-shirt, sweatshirt": 3,
    "dress": 2, "jumpsuit": 2, "pants": 1, "shorts": 1, "skirt": 1,
}


def check_categories(label2name):
    """Avisa si algún nombre de arriba no existe en tu JSON."""
    # diferencia de conjuntos para buscar los nombres que usamos aquí pero que no están en el dataset
    faltan = RELEVANTES - set(label2name.values())
    # si faltan, imprime el aviso y la lista; si no, imprime ok y un string vacío (el "or" hace de if/else)
    print("Nombres que no existen en tu JSON:" if faltan else "No falta ninguna categoría del JSON, todo correcto", sorted(faltan) or "")


# 1) detección (en resolución original)
@torch.no_grad()
def detectar(model, img_rgb, device, score_min=0.3):
    model.eval()
    H, W = img_rgb.shape[:2]
    s = min(1.0, C.MAX_SIDE / max(H, W))   # misma escala que en entrenamiento
    sw, sh = max(1, round(W * s)), max(1, round(H * s))
    # si no hace falta reducir (s == 1) usamos la imagen tal cual, si no, la reescalamos con PIL
    small = img_rgb if s == 1.0 else np.array(
        Image.fromarray(img_rgb).resize((sw, sh), Image.Resampling.BILINEAR))
    # factores reales de escala en x e y (para deshacerlo luego en las cajas)
    sx, sy = sw / W, sh / H

    # pasamos de numpy HWC uint8 a tensor CHW float 0-1, igual que en el dataset
    # el [0] coge el resultado de la primera (y única) imagen del batch
    out = model([torch.from_numpy(small).permute(2, 0, 1).float().div(255).to(device)])[0]
    keep = out["scores"] >= score_min  # filtro por confianza

    boxes = out["boxes"][keep].cpu().numpy().astype(np.float32)
    # las cajas están en coordenadas de la imagen reducida,
    # así que las dividimos entre la escala para pasarlas a la original 
    # y con clip evitamos que se salgan de la imagen. Columnas 0 y 2 = x, 1 y 3 = y
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] / sx).clip(0, W)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] / sy).clip(0, H)
    labels = out["labels"][keep].cpu().numpy()
    scores = out["scores"][keep].cpu().numpy()
    # se reescalan las probabilidades y luego se umbraliza (para que queden bordes más suaves)
    # si binarizáramos primero y agrandáramos después saldrían los bordes con "escalones"
    masks = [cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
             for p in out["masks"][keep, 0].cpu().numpy()]
    return boxes, labels, scores, masks


# 2) refinado con SAM 2
def load_sam(device_type):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor.from_pretrained(C.SAM2_MODEL_ID, device=device_type)
    print("SAM 2 cargado:", C.SAM2_MODEL_ID)
    return predictor


def iou_mascaras(a, b):
    # IoU mide cuánto se parecen dos máscaras (1 = idénticas, 0 = nada en común)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0  # el if evita dividir entre 0


def punto_interior(mask):
    # distanceTransform pone en cada píxel de la máscara su distancia al borde más cercano,
    # así que el máximo es el punto más "metido" dentro de la prenda (el más seguro para SAM)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    # argmax da un índice plano, unravel_index lo convierte a (fila, columna) = (y, x)
    y, x = np.unravel_index(dist.argmax(), dist.shape)
    return np.array([[x, y]], dtype=np.float32)  # SAM quiere (x, y), ojo que está al revés que numpy


def refinar_con_sam(predictor, mask, box, iou_min=0.6):
    """Le damos a SAM cada caja con un punto interior. Se elige la candidata más parecida a 
    la de Mask R-CNN. Si ninguna llega a iou_min se conserva la de Mask R-CNN. 
    Devuelve (máscara, usó_sam)."""
    if not mask.any():
        return mask, False  # máscara vacía, no hay nada que refinar
    # le damos a SAM la caja y un punto positivo (label 1 = "esto es parte del objeto").
    # multimask_output=True hace que devuelva 3 máscaras candidatas en vez de una
    masks, _, _ = predictor.predict(point_coords=punto_interior(mask), point_labels=np.array([1]),
                                    box=np.asarray(box, dtype=np.float32), multimask_output=True)
    cands = masks.astype(bool)
    # comparamos cada candidata de SAM con la máscara original de Mask R-CNN y nos quedamos con la más parecida
    ious = [iou_mascaras(c, mask) for c in cands]
    best = int(np.argmax(ious))
    # si ni la mejor se parece lo suficiente, SAM probablemente se ha ido a otra cosa: nos fiamos de Mask R-CNN
    return (cands[best], True) if ious[best] >= iou_min else (mask, False)


# 3) prenda + partes − cierres
def contenido_en(a, b):
    # qué fracción de 'a' está dentro de 'b' (no es simétrico, se divide solo entre el área de 'a')
    area = a.sum()
    return float(np.logical_and(a, b).sum() / area) if area else 0.0


def combinar_instancias(instancias, shape, umbral_parte=0.5, umbral_cierre=0.2,
                        dilatar_cierres=2, suavizado=1.5):
    """Devuelve una entrada por prenda con mask_final (bool) y alpha (float 0-1)."""
    H, W = shape
    # separamos las instancias en 3 grupos según su categoría
    # dict(i, partes=[], cierres=0) copia el diccionario y le añade campos nuevos 
    # (así no modificamos el original)
    prendas = [dict(i, partes=[], cierres=0) for i in instancias if i["nombre"] in PRENDAS]
    partes = [i for i in instancias if i["nombre"] in PARTES]
    cierres = [i for i in instancias if i["nombre"] in CIERRES]
    if not prendas:
        return []  # sin prendas no hay a qué pegarle partes ni cierres

    # partimos de la máscara de cada prenda; luego le iremos sumando partes
    for g in prendas:
        g["mask_prenda"] = g["mask"].copy()

    # cada parte se suma a la prenda que más la contiene
    for p in partes:
        cont = [contenido_en(p["mask"], g["mask"]) for g in prendas]
        # elegimos la prenda con mayor % de contención. La key es una tupla, primero el % redondeado y,
        # si hay empate, la prenda más pequeña (por eso el signo menos en el área)
        j = max(range(len(prendas)), key=lambda k: (round(cont[k], 3), -int(prendas[k]["mask"].sum())))
        if cont[j] >= umbral_parte:  # solo si al menos la mitad de la parte cae dentro de la prenda
            prendas[j]["partes"].append(p["nombre"])
            prendas[j]["mask_prenda"] |= p["mask"]  # |= es la unión de máscaras

    # se restan los cierres que caen dentro de la prenda (con una pequeña dilatación)
    # r escala con el tamaño de la imagen (referencia 1024 px) y como mínimo 1; el kernel es una elipse de (2r+1)x(2r+1)
    r = max(1, round(dilatar_cierres * max(H, W) / 1024))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    for g in prendas:
        quitar = np.zeros((H, W), bool)  # máscara acumulada de todo lo que hay que quitar a esta prenda
        for c in cierres:
            # un cierre cuenta si al menos el 20% está sobre la prenda (umbral bajo porque suelen ser cosas pequeñas)
            if contenido_en(c["mask"], g["mask_prenda"]) >= umbral_cierre:
                quitar |= c["mask"]
                g["cierres"] += 1
        if dilatar_cierres and quitar.any():
            # dilatamos un poco los cierres para que no queden bordes sueltos alrededor
            quitar = cv2.dilate(quitar.astype(np.uint8), kernel) > 0
        g["mask_final"] = g["mask_prenda"] & ~quitar  # prenda AND (NOT cierres)

    # gana la prenda de mayor prioridad (ocupa su área completa, antes de restar cierres)
    ocupado = np.zeros((H, W), bool)
    # recorremos de mayor a menor prioridad (el menos delante del lambda invierte el orden)
    for g in sorted(prendas, key=lambda g: -PRIORIDAD.get(g["nombre"], 0)):
        g["mask_final"] = g["mask_final"] & ~ocupado  # quitamos lo que ya se llevó una prenda de más prioridad
        # se marca con mask_prenda (no mask_final) para que los huecos de los cierres también
        # cuenten como ocupados y no se los quede una prenda de debajo
        ocupado |= g["mask_prenda"]

    # alfa suave
    # sigma del desenfoque también escala con el tamaño de la imagen, mínimo 1
    sigma = max(1.0, suavizado * max(H, W) / 1024)
    for g in prendas:
        # desenfocar la máscara binaria da bordes degradados (alfa de 0 a 1); (0, 0) deja que OpenCV calcule el kernel desde sigma
        g["alpha"] = np.clip(cv2.GaussianBlur(g["mask_final"].astype(np.float32), (0, 0), sigma), 0, 1)

    # devolvemos ordenadas por confianza, la más segura primero
    return sorted(prendas, key=lambda g: -g["score"])


# pipeline completo
def segmentar_prendas(det_model, predictor, img_rgb, device, label2name, usar_sam=True):
    """img_rgb: np.uint8 (H, W, 3) a resolución original. predictor puede ser None si usar_sam=False."""
    # detectamos con el umbral más bajo de los dos, y más abajo filtramos cada categoría con el suyo
    boxes, labels, scores, masks = detectar(det_model, img_rgb, device,
                                            score_min=min(C.SCORE_PRENDA, C.SCORE_OTRAS))
    candidatas = []
    for b, l, s, m in zip(boxes, labels, scores, masks):
        nombre = label2name[int(l)]  # pasa de índice del modelo a nombre de la clase
        # nos quedamos con las relevantes; las prendas usan un umbral de score y partes/cierres otro
        if nombre in RELEVANTES and s >= (C.SCORE_PRENDA if nombre in PRENDAS else C.SCORE_OTRAS):
            candidatas.append((nombre, float(s), b, m))

    if usar_sam and candidatas:
        predictor.set_image(img_rgb)   # el embedding se calcula una sola vez
        # lo que más cuesta de SAM es codificar la imagen, después cada predict() es rápido

    instancias = []
    for nombre, s, b, m in candidatas:
        # si usar_sam está es false se salta SAM y se deja la máscara de Mask R-CNN tal cual
        m_ref, ok = refinar_con_sam(predictor, m, b) if usar_sam else (m, False)
        # guardamos las dos máscaras (original y refinada) por si luego se quieren compararlas
        instancias.append(dict(nombre=nombre, score=s, box=b, mask_rcnn=m, mask=m_ref, sam=ok))
    return combinar_instancias(instancias, img_rgb.shape[:2])
