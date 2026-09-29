"""
Aplica el clasificador ya entrenado a todas las imágenes de una carpeta,
y las copia (no las mueve) a dos subcarpetas nuevas según la predicción:

"""

import os
import shutil

import torch
from PIL import Image

from dataset import get_transforms
from model import get_device, get_model



CHECKPOINT_PATH = "checkpoints/clasificador_best.pth"
INPUT_DIR = "dataset/train"     # carpeta con las imágenes nuevas sin clasificar
OUTPUT_DIR = "dataset/dataset human_classificator/test/train_classified"
IMAGE_SIZE = 224
THRESHOLD = 0.5   # a partir de qué probabilidad se considera "human"

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]  # ej. ["human", "not_human"]

    model = get_model(pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


@torch.no_grad()
def predict_image(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)  # añade dimensión de batch: (1, 3, H, W)

    logits = model(tensor)
    probability = torch.sigmoid(logits).item()  # probabilidad de la clase índice 0

    return probability


def main():
    device = get_device()
    model, class_names = load_model(CHECKPOINT_PATH, device)
    print(f"Clases del modelo: índice 0 = '{class_names[0]}', índice 1 = '{class_names[1]}'")

    transform = get_transforms(IMAGE_SIZE, train=False)

    # creamos las carpetas de salida, una por clase
    for name in class_names:
        os.makedirs(os.path.join(OUTPUT_DIR, name), exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(IMG_EXTENSIONS)]
    print(f"Imágenes encontradas en '{INPUT_DIR}': {len(files)}")

    counts = {name: 0 for name in class_names}

    for filename in files:
        src_path = os.path.join(INPUT_DIR, filename)
        probability = predict_image(model, src_path, transform, device)

        # el modelo devuelve la probabilidad de la clase índice 0
        # (class_names[0] es "human")
        predicted_index = 0 if probability >= THRESHOLD else 1
        predicted_class = class_names[predicted_index]

        dst_path = os.path.join(OUTPUT_DIR, predicted_class, filename)
        shutil.copy2(src_path, dst_path)
        counts[predicted_class] += 1

    print("\nResultado:")
    for name, count in counts.items():
        print(f"  {name}: {count} imágenes")
    print(f"\nRevisa las carpetas dentro de: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
