from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import zipfile
import random
import shutil


# ============================================================
# CONFIGURACIÓN
# ============================================================

REPO_ID = "sahirp/deepfashion2"

# Número de imágenes + JSON que quieres
NUM_SAMPLES = 5000

# True  -> selección aleatoria
# False -> primeras NUM_SAMPLES
RANDOM_SELECTION = True

# Semilla para reproducibilidad
SEED = 42


# ============================================================
# ELEGIR CARPETA DE DESTINO
# ============================================================

print("Introduce la carpeta donde quieres guardar el dataset.")
print("Ejemplos:")
print("  Windows: C:/Users/TuNombre/Desktop/deepfashion2")
print("  Linux:   /home/usuario/datasets/deepfashion2")
print()

output_input = input("Ruta de destino: ").strip()

if not output_input:
    raise ValueError("No has introducido ninguna ruta.")

OUTPUT_DIR = Path(output_input).expanduser()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nLos archivos se guardarán en:")
print(OUTPUT_DIR.resolve())


# ============================================================
# CARPETAS DEL SUBSET
# ============================================================

SUBSET_DIR = OUTPUT_DIR / f"subset_{NUM_SAMPLES}"
IMAGE_DIR = SUBSET_DIR / "image"
ANNO_DIR = SUBSET_DIR / "annos"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ANNO_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DESCARGAR TRAIN.ZIP
# ============================================================

print("\nDescargando train.zip desde Hugging Face...")
print("Esto puede tardar porque el ZIP completo ocupa varios GB.\n")

zip_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="train.zip",
    repo_type="dataset",
    local_dir=str(OUTPUT_DIR),
)

zip_path = Path(zip_path)

print(f"\nZIP descargado en:")
print(zip_path)


# ============================================================
# ANALIZAR ZIP
# ============================================================

print("\nAnalizando train.zip...")

with zipfile.ZipFile(zip_path, "r") as z:

    all_files = z.namelist()

    image_files = [
        f for f in all_files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and "/image/" in f
    ]

    json_files = [
        f for f in all_files
        if f.lower().endswith(".json")
        and "/annos/" in f
    ]

    print(f"Imágenes encontradas: {len(image_files):,}")
    print(f"JSON encontrados:      {len(json_files):,}")


    # ========================================================
    # ÍNDICE DE JSON
    # ========================================================

    json_by_stem = {
        Path(f).stem: f
        for f in json_files
    }


    # ========================================================
    # SOLO IMÁGENES CON JSON
    # ========================================================

    valid_images = [
        f for f in image_files
        if Path(f).stem in json_by_stem
    ]

    print(
        f"Pares imagen/JSON disponibles: "
        f"{len(valid_images):,}"
    )

    if NUM_SAMPLES > len(valid_images):
        raise ValueError(
            f"Has pedido {NUM_SAMPLES:,} muestras, "
            f"pero solo hay {len(valid_images):,} pares."
        )


    # ========================================================
    # SELECCIÓN
    # ========================================================

    if RANDOM_SELECTION:
        random.seed(SEED)

        selected_images = random.sample(
            valid_images,
            NUM_SAMPLES
        )
    else:
        selected_images = valid_images[:NUM_SAMPLES]


    # ========================================================
    # EXTRAER IMAGEN + JSON
    # ========================================================

    print(
        f"\nExtrayendo {NUM_SAMPLES:,} "
        f"imágenes y JSON...\n"
    )

    for image_path in tqdm(
        selected_images,
        desc="Extrayendo",
        unit="par",
        ncols=100
    ):

        image_name = Path(image_path).name
        stem = Path(image_path).stem

        json_path = json_by_stem[stem]

        image_destination = IMAGE_DIR / image_name
        json_destination = ANNO_DIR / f"{stem}.json"


        # Imagen
        with z.open(image_path) as source:
            with open(image_destination, "wb") as target:
                shutil.copyfileobj(source, target)


        # JSON
        with z.open(json_path) as source:
            with open(json_destination, "wb") as target:
                shutil.copyfileobj(source, target)


# ============================================================
# RESULTADO
# ============================================================

print("\n" + "=" * 60)
print("EXTRACCIÓN TERMINADA")
print("=" * 60)

print(f"Imágenes: {NUM_SAMPLES:,}")
print(f"JSONs:    {NUM_SAMPLES:,}")

print(f"\nDataset:")
print(SUBSET_DIR)

print("=" * 60)