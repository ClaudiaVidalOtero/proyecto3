"""Toda la configuración en un solo sitio. Edita aquí rutas e hiperparámetros."""
from pathlib import Path

# ---- Rutas (relativas a la raíz del repo) ----
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "dataset"

TRAIN_IMAGES_DIR = str(DATA_DIR / "train_classified" / "not_human")
TRAIN_ANNOTATIONS = str(DATA_DIR / "instances_no_humans.json")
TEST_IMAGES_DIR = str(DATA_DIR / "provisional_test_no_human_test")
TEST_ANNOTATIONS = str(DATA_DIR / "instances_provisional_test_no_humans.json")

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

# ---- Recoloreado con InstructPix2Pix ----
PIX2PIX_CHECKPOINT = str(Path(CHECKPOINT_DIR) / "best.pth")
PIX2PIX_IMAGES_DIR = str(DATA_DIR / "provisional_test_no_humans")
PIX2PIX_ANNOTATIONS_FILE = str(DATA_DIR / "instances_provisional_test_no_humans.json")
PIX2PIX_IMAGE_SIZE = 256
PIX2PIX_OUTPUT_DIR = str(Path(OUTPUT_DIR) / "pix2pix")
PIX2PIX_OUTPUT_PREFIX = "recoloreado_"
PIX2PIX_TEXTURES_DIR = str(REPO / "textures")

PIX2PIX_MODEL_ID = "timbrooks/instruct-pix2pix"
PIX2PIX_SIZE = 512
PIX2PIX_INSTRUCTION_COLOR = "Change the color of the selected {name} to {color}."
PIX2PIX_INSTRUCTION_TEXTURE = "Change the material and surface texture of the selected {name} to {texture}."
PIX2PIX_NUM_INFERENCE_STEPS = 30
PIX2PIX_GUIDANCE_SCALE = 7.5
PIX2PIX_IMAGE_GUIDANCE_SCALE = 2.5
# El pipeline se carga una sola vez y se reutiliza entre generaciones (ver src/recolor.py).
# Activa el "attention slicing" solo si te quedas sin memoria en la GPU (frena la difusión).
PIX2PIX_ATTENTION_SLICING = False
PIX2PIX_MASK_CLOSE_SIZE = 7
PIX2PIX_MASK_DILATE_SIZE = 7
PIX2PIX_FEATHER_RADIUS = 2
PIX2PIX_FOLD_PRESERVATION_STRENGTH = 0.75
PIX2PIX_FOLD_PRESERVATION_RADIUS = 12
PIX2PIX_SEED = 42
PIX2PIX_NO_TEXTURE = "none"

PIX2PIX_IP_ADAPTER_REPO = "h94/IP-Adapter"
PIX2PIX_IP_ADAPTER_WEIGHTS = "ip-adapter_sd15.bin"
PIX2PIX_IP_ADAPTER_SCALE = 0.75

PIX2PIX_NUM_CHOICES = 10
PIX2PIX_THUMB_SIZE = 140
PIX2PIX_NUM_TEXTURE_CHOICES = 10
PIX2PIX_TEXTURE_THUMB_SIZE = 96
PIX2PIX_FIGURE_SIZE = (20, 5)
PIX2PIX_SAVE_DPI = 150
PIX2PIX_DEFAULT_COLOR = "black"
PIX2PIX_DEFAULT_TEXTURE = "leather"
PIX2PIX_DEFAULT_EXTRA = ""
PIX2PIX_COLOR_OPTIONS = [
    "black", "white", "red", "blue", "navy blue", "green", "dark green",
    "yellow", "pink", "purple", "orange", "brown", "gray", "beige",
]
# Colores que se generan con el modelo; el resto se recolorea por tono HSV.
PIX2PIX_MODEL_COLORS = ("black", "white")
PIX2PIX_COLOR_HEX = {
    "black": "#252525", "white": "#ffffff", "red": "#c94843", "blue": "#466ca5",
    "navy blue": "#293650", "green": "#66866a", "dark green": "#345343",
    "yellow": "#e9c44e", "pink": "#dc91a4", "purple": "#886c9f",
    "orange": "#e29150", "brown": "#866044", "gray": "#91938e", "beige": "#d5c4a2",
}
PIX2PIX_DEFAULT_HEX = "#c75d3a"
PIX2PIX_TEXTURE_OPTIONS = [
    PIX2PIX_NO_TEXTURE, "leather", "denim", "silk", "velvet", "wool",
    "cotton", "satin",
]
PIX2PIX_TEXTURE_CATEGORY_HINTS = {
    "denim": "fabric", "jeans": "fabric", "silk": "fabric", "satin": "fabric",
    "velvet": "fabric", "cotton": "fabric", "linen": "fabric", "corduroy": "fabric",
    "wool": "fabric", "fabric": "fabric", "lace": "net", "fishnet": "net",
    "latex": "rubber", "rug": "carpet",
}