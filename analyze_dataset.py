from pathlib import Path
import json
import csv
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Raíz del proyecto:
# proyecto3/
# ├── dataset/
# │   ├── train/
# │   ├── val/
# │   ├── instances_attributes_train2020.json
# │   └── instances_attributes_val2020.json
# └── scripts/
#     └── analyze_images.py

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"

# Cambia estos nombres si tus archivos tienen otro nombre.
SPLITS = {
    "train": {
        "images": DATASET_DIR / "train",
        "annotations": DATASET_DIR / "instances_attributes_train2020.json",
    },
    "val": {
        "images": DATASET_DIR / "test",
        "annotations": DATASET_DIR / "instances_attributes_val2020.json",
    },
}

OUTPUT_DIR = BASE_DIR / "outputs" / "image_analysis"

# Número máximo de ejemplos que guardaremos por categoría.
MAX_EXAMPLES_PER_GROUP = 20

# Para probar rápidamente:
# pon un número, por ejemplo 500.
# None = procesar todas las imágenes.
MAX_IMAGES_PER_SPLIT = None

# Porcentaje mínimo de píxeles blancos para considerar
# que el fondo es predominantemente blanco.
WHITE_BACKGROUND_THRESHOLD = 0.70

# Un píxel se considera "blanco" si sus canales están
# suficientemente cerca de 255.
WHITE_PIXEL_THRESHOLD = 235

# ============================================================
# GRUPOS
# ============================================================

GROUPS = {
    "modelo": "Probable modelo/persona",
    "fondo_blanco": "Probable fondo blanco",
    "modelo_y_fondo_blanco": "Modelo + fondo blanco",
    "sin_modelo_fondo_no_blanco": "Sin modelo + fondo no blanco",
    "indeterminado": "Indeterminado",
}


# ============================================================
# UTILIDADES
# ============================================================

def load_json(path):
    print(f"Cargando anotaciones: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_annotations_index(data):
    """
    Agrupa las anotaciones por image_id.
    """
    annotations_by_image = {}

    for ann in data.get("annotations", []):
        image_id = ann["image_id"]

        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []

        annotations_by_image[image_id].append(ann)

    return annotations_by_image


def find_image(images_dir, file_name):
    """
    Busca la imagen indicada por file_name.

    Primero prueba directamente:
        images_dir / file_name

    Si no existe, busca recursivamente por nombre.
    """
    direct_path = images_dir / file_name

    if direct_path.exists():
        return direct_path

    matches = list(images_dir.rglob(Path(file_name).name))

    if matches:
        return matches[0]

    return None


def decode_segmentation_mask(annotation, height, width):
    """
    Convierte una segmentación COCO a una máscara binaria.

    Soporta:
    - polígonos
    - RLE
    """

    segmentation = annotation.get("segmentation")

    if not segmentation:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)

    # --------------------------------------------------------
    # Caso 1: polygon
    # --------------------------------------------------------

    if isinstance(segmentation, list):

        for polygon in segmentation:

            if len(polygon) < 6:
                continue

            points = np.array(
                polygon,
                dtype=np.float32
            ).reshape(-1, 2)

            points = np.round(points).astype(np.int32)

            cv2.fillPoly(
                mask,
                [points],
                1
            )

        return mask

    # --------------------------------------------------------
    # Caso 2: RLE
    # --------------------------------------------------------

    if isinstance(segmentation, dict):

        try:
            from pycocotools import mask as mask_utils

            rle = segmentation.copy()

            counts = rle.get("counts")

            # Fashionpedia puede almacenar counts como string.
            if isinstance(counts, str):
                rle["counts"] = counts.encode("utf-8")

            decoded = mask_utils.decode(rle)

            if decoded.ndim == 3:
                decoded = decoded[:, :, 0]

            return decoded.astype(np.uint8)

        except Exception:
            return None

    return None


def create_clothing_mask(image_info, annotations):
    """
    Crea una máscara combinada con todas las prendas
    anotadas en la imagen.
    """

    height = image_info["height"]
    width = image_info["width"]

    clothing_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for ann in annotations:

        mask = decode_segmentation_mask(
            ann,
            height,
            width
        )

        if mask is not None:
            clothing_mask |= mask

    return clothing_mask


# ============================================================
# DETECCIÓN DE FONDO BLANCO
# ============================================================

def calculate_white_background_ratio(image, clothing_mask):
    """
    Calcula qué porcentaje del fondo (fuera de las máscaras
    de ropa) es predominantemente blanco.

    No analizamos solamente toda la imagen porque una camisa
    blanca, por ejemplo, no debería contar como fondo blanco.
    """

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Zona que NO pertenece a una prenda.
    background = clothing_mask == 0

    if np.count_nonzero(background) == 0:
        return 0.0

    pixels = rgb[background]

    white_pixels = np.all(
        pixels >= WHITE_PIXEL_THRESHOLD,
        axis=1
    )

    ratio = white_pixels.mean()

    return float(ratio)


# ============================================================
# DETECCIÓN DE PERSONA
# ============================================================

def detect_person_hog(image):
    """
    Detecta personas mediante HOG + SVM de OpenCV.

    Es una heurística rápida. No sustituye a un detector
    moderno, pero sirve para obtener una primera clasificación
    del dataset sin entrenar otro modelo.
    """

    hog = cv2.HOGDescriptor()

    hog.setSVMDetector(
        cv2.HOGDescriptor_getDefaultPeopleDetector()
    )

    # Reducimos el tamaño para acelerar el análisis.
    max_width = 800

    height, width = image.shape[:2]

    if width > max_width:

        scale = max_width / width

        new_width = int(width * scale)
        new_height = int(height * scale)

        resized = cv2.resize(
            image,
            (new_width, new_height)
        )

    else:
        resized = image

    try:

        boxes, weights = hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        if len(boxes) == 0:
            return False, 0.0

        # Nos quedamos con la detección de mayor confianza.
        max_weight = float(np.max(weights))

        return True, max_weight

    except Exception:
        return False, 0.0


# ============================================================
# CLASIFICACIÓN
# ============================================================

def classify_image(
    image,
    clothing_mask
):
    """
    Clasifica una imagen en uno de los grupos definidos.
    """

    white_ratio = calculate_white_background_ratio(
        image,
        clothing_mask
    )

    has_person, person_score = detect_person_hog(
        image
    )

    white_background = (
        white_ratio >= WHITE_BACKGROUND_THRESHOLD
    )

    if has_person and white_background:

        group = "modelo_y_fondo_blanco"

    elif has_person:

        group = "modelo"

    elif white_background:

        group = "fondo_blanco"

    else:

        group = "sin_modelo_fondo_no_blanco"

    return {
        "group": group,
        "has_person": has_person,
        "person_score": person_score,
        "white_ratio": white_ratio,
    }


# ============================================================
# VISUALIZACIÓN
# ============================================================

def create_visualization(
    image_path,
    output_path,
    result,
    split,
):
    """
    Guarda una copia de la imagen con información
    de la clasificación.
    """

    image = Image.open(image_path).convert("RGB")

    # Limitamos tamaño para que los ejemplos no ocupen
    # demasiado espacio.
    max_width = 1000

    if image.width > max_width:

        scale = max_width / image.width

        image = image.resize(
            (
                max_width,
                int(image.height * scale)
            )
        )

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            24
        )
    except Exception:
        font = ImageFont.load_default()

    group_name = GROUPS[result["group"]]

    text = (
        f"{split}\n"
        f"{group_name}\n"
        f"Persona: {result['has_person']}\n"
        f"Fondo blanco: {result['white_ratio']:.1%}"
    )

    # Fondo negro semitransparente aproximado.
    bbox = draw.multiline_textbbox(
        (20, 20),
        text,
        font=font,
        spacing=5
    )

    draw.rectangle(
        (
            bbox[0] - 10,
            bbox[1] - 10,
            bbox[2] + 10,
            bbox[3] + 10
        ),
        fill=(0, 0, 0)
    )

    draw.multiline_text(
        (20, 20),
        text,
        fill=(255, 255, 255),
        font=font,
        spacing=5
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(output_path)


# ============================================================
# PROCESAMIENTO
# ============================================================

def process_split(
    split_name,
    config,
    csv_rows,
):
    images_dir = config["images"]
    annotations_path = config["annotations"]

    if not images_dir.exists():

        print(
            f"\n[AVISO] No existe: {images_dir}"
        )

        return

    if not annotations_path.exists():

        print(
            f"\n[AVISO] No existe: {annotations_path}"
        )

        return

    data = load_json(
        annotations_path
    )

    annotations_by_image = build_annotations_index(
        data
    )

    images = data.get("images", [])

    if MAX_IMAGES_PER_SPLIT is not None:

        images = images[
            :MAX_IMAGES_PER_SPLIT
        ]

    print()
    print("=" * 70)
    print(f"PROCESANDO: {split_name}")
    print("=" * 70)

    print(
        f"Imágenes a analizar: {len(images)}"
    )

    counters = {
        group: 0
        for group in GROUPS
    }

    example_counters = {
        group: 0
        for group in GROUPS
    }

    for index, image_info in enumerate(images):

        image_id = image_info["id"]
        file_name = image_info["file_name"]

        image_path = find_image(
            images_dir,
            file_name
        )

        if image_path is None:

            print(
                f"[AVISO] Imagen no encontrada: "
                f"{file_name}"
            )

            continue

        try:

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                continue

            height, width = image.shape[:2]

            annotations = annotations_by_image.get(
                image_id,
                []
            )

            clothing_mask = create_clothing_mask(
                image_info,
                annotations
            )

            result = classify_image(
                image,
                clothing_mask
            )

            group = result["group"]

            counters[group] += 1

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            csv_rows.append({
                "split": split_name,
                "image_id": image_id,
                "file_name": file_name,
                "group": group,
                "has_person": result["has_person"],
                "person_score": result["person_score"],
                "white_background_ratio": result["white_ratio"],
                "num_annotations": len(annotations),
            })

            # ------------------------------------------------
            # Guardar ejemplos
            # ------------------------------------------------

            if (
                example_counters[group]
                < MAX_EXAMPLES_PER_GROUP
            ):

                output_path = (
                    OUTPUT_DIR
                    / "examples"
                    / split_name
                    / group
                    / f"{image_id}.jpg"
                )

                create_visualization(
                    image_path,
                    output_path,
                    result,
                    split_name
                )

                example_counters[group] += 1

            # ------------------------------------------------
            # Progreso
            # ------------------------------------------------

            if (
                (index + 1) % 100 == 0
                or index == len(images) - 1
            ):

                print(
                    f"\rProcesadas: "
                    f"{index + 1}/{len(images)}",
                    end=""
                )

        except Exception as e:

            print(
                f"\n[ERROR] {file_name}: {e}"
            )

    print("\n")

    print(
        f"Resultados de {split_name}:"
    )

    total = sum(counters.values())

    for group, count in counters.items():

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"  {GROUPS[group]:35s}"
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_rows = []

    print()
    print("=" * 70)
    print("FASHIONPEDIA - ANÁLISIS DE IMÁGENES")
    print("=" * 70)
    print()
    print(f"Dataset: {DATASET_DIR}")
    print(f"Salida:  {OUTPUT_DIR}")
    print()

    for split_name, config in SPLITS.items():

        process_split(
            split_name,
            config,
            csv_rows
        )

    # ========================================================
    # GUARDAR CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "image_analysis.csv"
    )

    fieldnames = [
        "split",
        "image_id",
        "file_name",
        "group",
        "has_person",
        "person_score",
        "white_background_ratio",
        "num_annotations",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )

    print()
    print("=" * 70)
    print("ANÁLISIS TERMINADO")
    print("=" * 70)

    print(
        f"\nCSV: {csv_path}"
    )

    print(
        f"Ejemplos: "
        f"{OUTPUT_DIR / 'examples'}"
    )

    print(
        f"\nImágenes analizadas: "
        f"{len(csv_rows)}"
    )


if __name__ == "__main__":
    main()