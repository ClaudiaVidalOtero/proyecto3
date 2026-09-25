import argparse
import csv
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


def get_images(folder):
    """Devuelve las rutas de todas las imágenes sin cargarlas en memoria."""
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def detect_person(image, hog):
    """
    Detecta personas usando el detector HOG de OpenCV.

    Devuelve:
        label       -> con_modelo / sin_modelo / revisar
        confidence  -> confianza aproximada del detector
        area_ratio  -> proporción de la imagen ocupada por la detección
    """

    height, width = image.shape[:2]

    # Reducimos imágenes enormes para acelerar el detector.
    max_dimension = 800

    scale = min(1.0, max_dimension / max(height, width))

    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)

        image_small = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
    else:
        image_small = image

    rects, weights = hog.detectMultiScale(
        image_small,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    if len(rects) == 0:
        return "sin_modelo", 0.0, 0.0

    # Nos quedamos con la detección más importante.
    best_idx = max(
        range(len(rects)),
        key=lambda i: float(weights[i])
    )

    x, y, w, h = rects[best_idx]
    confidence = float(weights[best_idx])

    image_area = image_small.shape[0] * image_small.shape[1]
    person_area = w * h
    area_ratio = person_area / image_area

    # ---------------------------------------------------------
    # Clasificación
    # ---------------------------------------------------------
    #
    # HOG puede producir falsos positivos, por lo que usamos
    # tres categorías.
    #
    # Persona claramente detectada:
    if confidence >= 0.5 and area_ratio >= 0.08:
        return "con_modelo", confidence, area_ratio

    # Detección muy débil:
    if confidence < 0.2:
        return "sin_modelo", confidence, area_ratio

    # Entre ambos casos -> revisión manual.
    return "revisar", confidence, area_ratio


def main():

    parser = argparse.ArgumentParser(
        description="Separa imágenes de prendas con/sin modelo."
    )

    parser.add_argument(
        "input",
        help="Carpeta con las imágenes originales"
    )

    parser.add_argument(
        "output",
        help="Carpeta donde se guardará el resultado"
    )

    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copiar imágenes en vez de moverlas"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: no existe {input_dir}")
        return

    # Carpetas de salida
    with_model = output_dir / "con_modelo"
    without_model = output_dir / "sin_modelo"
    review = output_dir / "revisar"

    with_model.mkdir(parents=True, exist_ok=True)
    without_model.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "resultados.csv"

    # ---------------------------------------------------------
    # Detector HOG de personas de OpenCV
    # ---------------------------------------------------------

    hog = cv2.HOGDescriptor()

    hog.setSVMDetector(
        cv2.HOGDescriptor_getDefaultPeopleDetector()
    )

    images = list(get_images(input_dir))

    print(f"Imágenes encontradas: {len(images)}")
    print()

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "archivo",
            "clase",
            "confianza",
            "area_persona"
        ])

        for image_path in tqdm(images, desc="Procesando"):

            try:

                image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_COLOR
                )

                if image is None:
                    print(
                        f"\nNo se pudo leer: {image_path}"
                    )
                    continue

                label, confidence, area_ratio = detect_person(
                    image,
                    hog
                )

                # Liberamos inmediatamente la imagen.
                del image

                # -------------------------------------------------
                # Nombre de salida
                #
                # Conservamos la estructura relativa del dataset
                # para evitar colisiones de nombres.
                # -------------------------------------------------

                relative_path = image_path.relative_to(input_dir)

                if label == "con_modelo":
                    destination = with_model / relative_path

                elif label == "sin_modelo":
                    destination = without_model / relative_path

                else:
                    destination = review / relative_path

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                # -------------------------------------------------
                # Copiar o mover
                # -------------------------------------------------

                if args.copy:
                    shutil.copy2(
                        image_path,
                        destination
                    )
                else:
                    shutil.move(
                        image_path,
                        destination
                    )

                writer.writerow([
                    str(relative_path),
                    label,
                    f"{confidence:.4f}",
                    f"{area_ratio:.4f}"
                ])

                # Guardamos cada resultado inmediatamente.
                csv_file.flush()

            except Exception as e:

                print(
                    f"\nERROR procesando {image_path}: {e}"
                )

    print()
    print("=" * 50)
    print("Proceso terminado")
    print("=" * 50)
    print(f"Resultado: {output_dir}")
    print(f"CSV:       {csv_path}")


if __name__ == "__main__":
    main()