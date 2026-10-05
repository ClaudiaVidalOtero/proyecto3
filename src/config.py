"""Toda la configuración en un solo sitio. Edita aquí rutas e hiperparámetros."""
from pathlib import Path

# ---- Rutas (relativas a la raíz del repo) ----
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "dataset"

TRAIN_IMAGES_DIR = str(DATA_DIR / "train_classified" / "not_human")
TRAIN_ANNOTATIONS = str(DATA_DIR / "instances_no_humans.json")
TEST_IMAGES_DIR = str(DATA_DIR / "train_classified" / "not_human_test")
TEST_ANNOTATIONS = str(DATA_DIR / "instances_test_no_humans.json")

CHECKPOINT_DIR = str(REPO / "checkpoints")   # best.pth y last.pth
OUTPUT_DIR = str(REPO / "outputs")           # gráficas y resultados
# Imagen (o carpeta de imágenes) que usa "python main.py segment" si no se pasa --input
SEGMENT_INPUT = str(REPO / "outputs" / "predictions.png")

VAL_FRACTION = 0.116
SPLIT_SEED = 42

# ---- Resolución ----
MAX_SIDE = 768                          # lado largo máximo (sin deformar)
MIN_SIZE = (384, 416, 448, 480, 512)    # escalas aleatorias en train; en eval se usa la última

# ---- Entrenamiento ----
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 0.0025
WARMUP_ITERS = 500      # el lr sube linealmente durante la 1ª época
LR_STEP_SIZE = 3        # lr / 10 cada N épocas
WORKERS = 2
USE_AMP = True
AUGMENT = True
PRETRAINED = True

MAX_TRAIN_IMAGES = None   # pon p. ej. 50 para una prueba rápida
MAX_VAL_IMAGES = None
MAP_VAL_IMAGES = 300      # imágenes de val para el mAP de cada época

# ---- Inferencia / SAM 2 ----
SCORE_THRESHOLD = 0.5
SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"   # más ligero: facebook/sam2.1-hiera-base-plus
SCORE_PRENDA = 0.5
SCORE_OTRAS = 0.3