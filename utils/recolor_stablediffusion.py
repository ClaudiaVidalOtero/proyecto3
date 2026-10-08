"""
Recolorea una prenda usando Stable Diffusion Inpainting. La máscara que se le pasa al modelo generativo 
es la que produce nuestro pipeline Mask R-CNN + SAM2.

"""

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline

# Añadimos la raíz del proyecto (carpeta padre de utils/)
# al path para que funcionen los imports de src y utils
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.dataset import FashionpediaDataset
from src.model import get_model, NUM_CLASSES
from refine_edges import refine_with_sam2


CHECKPOINT = "checkpoints/best.pth"
IMAGES_DIR = "dataset/provisional_test_no_humans"
ANNOTATIONS_FILE = "dataset/instances_provisional_test_no_humans.json"
IMAGE_SIZE = 256
INDEX = 45

SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
SD_SIZE = 512          # Stable Diffusion trabaja en múltiplos de 8. 512 es el tamaño estándar
# el modelo entiende mejor en los prompts en inglés
PROMPT_TEMPLATE = "a {color} {name}, photorealistic, same fabric texture, studio photo"
NEGATIVE_PROMPT = "blurry, deformed, extra limbs, low quality, cartoon"
COLOR_NAME = "red"     # el color que quieres pedirle al modelo
NUM_INFERENCE_STEPS = 12
GUIDANCE_SCALE = 15


def load_mask_rcnn(checkpoint_path):
    model = get_model(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def mask_to_sd_input(binary_mask, target_size):
    """
    Convierte una máscara booleana (H, W) al formato que espera
    Stable Diffusion. Una imagen en escala de grises,
    blanco (255) = repintar aquí, negro (0) = conservar tal cual.
    Se redimensiona a SD_SIZE con interpolación NEAREST, para no "difuminar" el
    borde de la máscara al cambiar de tamaño.
    """
    mask_uint8 = (binary_mask.astype(np.uint8)) * 255
    mask_img = Image.fromarray(mask_uint8).resize((target_size, target_size), Image.NEAREST)
    return mask_img


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Mask R-CNN
    dataset = FashionpediaDataset(IMAGES_DIR, ANNOTATIONS_FILE, image_size=IMAGE_SIZE)
    image_tensor, _ = dataset[INDEX]
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    mask_rcnn = load_mask_rcnn(CHECKPOINT)
    with torch.no_grad():
        model_output = mask_rcnn([image_tensor])[0]

    label2name = {
        class_index: dataset.coco.loadCats(category_id)[0]["name"]
        for category_id, class_index in dataset.categoryid_to_index.items()
    }

    # refinar bordes con SAM 2 y agrupar (prenda + partes - cierres)
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large",
        device=device.type,
    )

    _, prendas, _ = refine_with_sam2(model_output, image_np, sam2_predictor, label2name)

    if len(prendas) == 0:
        print("SAM2 + Mask R-CNN no encontraron ninguna prenda clara en esta imagen.")
        print("Prueba a cambiar INDEX, o baja score_prenda en refinar_con_sam2.")
        return

    prenda = prendas[0]  # la de mayor score; cambia el índice si quieres otra
    print(f"Prenda elegida: {prenda['name']} (score={prenda['score']:.2f}, "
          f"partes={prenda['partes']}, cierres restados={prenda['cierres']})")

    # transforma imagen y máscara al tamaño que espera Stable Diffusion
    image_pil = Image.fromarray(image_np).resize((SD_SIZE, SD_SIZE), Image.BILINEAR)
    mask_pil = mask_to_sd_input(prenda["mask"], SD_SIZE)

    # cargar el pipeline de inpainting y generar
    print("\nCargando Stable Diffusion Inpainting (puede tardar la primera vez)...")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # reduce memoria, a costa de algo de velocidad

    prompt = PROMPT_TEMPLATE.format(color=COLOR_NAME, name=prenda["name"])
    print(f"Prompt: {prompt}")

    generator = torch.Generator(device=device).manual_seed(42)
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=image_pil,
        mask_image=mask_pil,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]

    # mostramos resultado
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_pil)
    axes[0].set_title(f"Original (índice {INDEX})")
    axes[0].axis("off")

    axes[1].imshow(mask_pil, cmap="gray")
    axes[1].set_title(f"Máscara refinada ({prenda['name']})")
    axes[1].axis("off")

    axes[2].imshow(result)
    axes[2].set_title(f"Recoloreado: '{COLOR_NAME}'")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("prueba_recoloreado_diffusion.png", dpi=150)
    print("Guardado en prueba_recoloreado_diffusion.png")
    plt.show()


if __name__ == "__main__":
    main()