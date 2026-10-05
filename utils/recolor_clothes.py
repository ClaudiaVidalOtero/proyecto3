"""
Cambia el color de una prenda segmentada por el modelo Mask R-CNN
ya entrenado, modificando solo el canal H (tono) en el espacio de color
HSV, y dejando S y V intactos para conservar pliegues, sombras y brillos.

"""

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import random

# añadimos la raíz del proyecto (carpeta padre de utils/) 
# al path para que funcionen los imports de src
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.dataset import FashionpediaDataset
from src.model import get_model, NUM_CLASSES


def recolor_hsv(image_rgb, binary_mask, new_hue_degrees, target_saturation=200, target_value=220, min_value_floor=40):
    """
    image_rgb: array (H, W, 3), valores 0-255, en RGB
    binary_mask: array (H, W), True/1 donde está la prenda a recolorear
    new_hue_degrees: el tono nuevo deseado, en grados (0-360),(0=rojo, 120=verde, 240=azul...)  
    target_saturation: saturación (0-255) del color elegido. Se aplica
                        FIJA a toda la máscara: así el tono se ve
                        consistente en toda la prenda, no solo "rescatado"
                        en las zonas que ya tenían algo de saturación.
    target_value: brillo (0-255) del color elegido. El píxel MÁS
                   brillante de la prenda pasará a valer exactamente
                   este valor -> ahí es donde verás el color "puro"
                   que elegiste. El resto de píxeles se reescalan
                   proporcionalmente por debajo, conservando pliegues
                   y sombras relativas al nuevo color, en vez de a los
                   originales.
    min_value_floor: brillo mínimo absoluto, para que las sombras más
                      profundas no lleguen a negro puro (donde no hay
                      color que mostrar, como vimos con prendas oscuras).
    
    """
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.int32)

    # en OpenCV, el canal H va de 0 a 179 (no de 0 a 359 como en teoría),
    # por eso hay que dividir entre 2 el valor en grados.
    new_hue_opencv = int(new_hue_degrees / 2)

    current_value = image_hsv[:, :, 2]

    # el punto más brillante dentro de la máscara pasa a valer target_value; 
    # el resto se reescala proporcionalmente respecto a ese máximo, 
    # así que las sombras siguen siendo "más oscuras que el punto más iluminado", 
    # pero ahora medidas sobre el color nuevo, no sobre el color original.
    mask_values = current_value[binary_mask]
    max_value = mask_values.max() if mask_values.size > 0 else 255
    normalized_value = current_value / max_value          # 0.0 a 1.0, 1.0 = el más brillante
    new_value = normalized_value * target_value
    new_value = np.maximum(new_value, min_value_floor)     # evita sombras en negro absoluto

    image_hsv[:, :, 0] = np.where(binary_mask, new_hue_opencv, image_hsv[:, :, 0])
    image_hsv[:, :, 1] = np.where(binary_mask, target_saturation, image_hsv[:, :, 1])
    image_hsv[:, :, 2] = np.where(binary_mask, new_value, current_value)

    image_hsv = image_hsv.astype(np.uint8)
    image_recolored = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    return image_recolored


def load_checkpoint(path):
    model = get_model(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":

    CHECKPOINT = "checkpoints/best.pth"
    IMAGES_DIR = "dataset/provisional_test_no_humans"
    ANNOTATIONS_FILE = "dataset/instances_provisional_test_no_humans.json"
    IMAGE_SIZE = 256
    SCORE_THRESHOLD = 0.5
    NEW_HUE = 120      # 0=rojo, 120=verde, 240=azul...


    dataset = FashionpediaDataset(IMAGES_DIR, ANNOTATIONS_FILE, image_size=IMAGE_SIZE)
    indx = random.randint(0, len(dataset)-1)      # escoge una imagen aleatoria del dataset
    image_tensor, _ = dataset[indx]

    model = load_checkpoint(CHECKPOINT)
    with torch.no_grad():
        prediction = model([image_tensor])[0]

    scores = prediction["scores"].numpy()
    masks = prediction["masks"].squeeze(1).numpy()  # (N, H, W), valores de probabilidad 0-1

    # recoloreamos la imagen en la escala que usó el modelo (256x256)
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # de momento recoloreamos todas las prendas detectadas
    combined_mask = np.zeros(masks.shape[1:], dtype=bool)
    n_detecciones = 0
    for i in range(len(scores)):
        if scores[i] >= SCORE_THRESHOLD:
            combined_mask |= (masks[i] >= 0.5)
            n_detecciones += 1

    print(f"Prendas detectadas (score >= {SCORE_THRESHOLD}): {n_detecciones}")
    if n_detecciones == 0:
        print("Ninguna detección supera el umbral; prueba a bajar SCORE_THRESHOLD o cambiar INDEX.")

    image_recolored = recolor_hsv(image_np, combined_mask, NEW_HUE)

    # mostramos el resultado
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title(f"Original (índice {indx})")
    axes[0].axis("off")

    axes[1].imshow(combined_mask, cmap="gray")
    axes[1].set_title("Máscara predicha por el modelo")
    axes[1].axis("off")

    axes[2].imshow(image_recolored)
    axes[2].set_title(f"Recoloreado (H={NEW_HUE}°)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("prueba_recoloreado.png", dpi=150)
    print("Guardado en prueba_recoloreado.png")
    plt.show()
