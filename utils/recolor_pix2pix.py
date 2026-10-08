"""
Recolorea una prenda usando InstructPix2Pix. Primero recibe la imagen original 
y una instrucción de texto ("make the jacket red") y edita la imagen entera. 
Para que el cambio afecte solo a la prenda, el resultado se pega sobre
la máscara que produce nuestro pipeline Mask R-CNN + SAM2.

"""

import numpy as np
import torch
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

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
INDEX = 38

IP2P_MODEL_ID = "timbrooks/instruct-pix2pix"
IP2P_SIZE = 512        # tamaño al que se reescala la imagen antes de entrar al modelo
# InstructPix2Pix funciona con instrucciones (órdenes), no con descripciones
INSTRUCTION_TEMPLATE = "make the {name} {color}, change the texture into leather"
COLOR_NAME = "black"     # el color que quieres pedirle al modelo, en inglés
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5          # cuánto obedece a la instrucción de texto
IMAGE_GUIDANCE_SCALE = 1.5    # cuánto se pega a la imagen original (subirlo si alucina)
FEATHER_RADIUS = 3            # difuminado del borde de la máscara al pegar el resultado (0 = borde duro)


def load_mask_rcnn(checkpoint_path):
    model = get_model(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def mask_to_pil(binary_mask, target_size, feather_radius):
    """
    Convierte una máscara booleana (H, W) en una imagen PIL en escala de grises
    del tamaño de trabajo del modelo. Blanco = zona donde se usa el resultado
    del modelo, negro = zona donde se conserva la imagen original.
    El redimensionado se hace con "nearest" (para no mezclar valores) y después
    se difumina el borde para que la transición sea gradual y no se note el corte.
    """
    mask_uint8 = binary_mask.astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_uint8).resize((target_size, target_size), Image.NEAREST)
    if feather_radius > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(feather_radius))
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
        print("Prueba a cambiar INDEX, o baja score_prenda en refine_with_sam2.")
        return

    prenda = prendas[0]  # la de mayor score
    print(f"Prenda elegida: {prenda['name']} (score={prenda['score']:.2f}, "
          f"partes={prenda['partes']}, cierres restados={prenda['cierres']})")

    # imagen y máscara al tamaño de trabajo del modelo
    image_pil = Image.fromarray(image_np).resize((IP2P_SIZE, IP2P_SIZE), Image.BICUBIC)
    mask_pil = mask_to_pil(prenda["mask"], IP2P_SIZE, FEATHER_RADIUS)

    # cargar InstructPix2Pix
    print("\nCargando InstructPix2Pix (puede tardar la primera vez)...")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        IP2P_MODEL_ID, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # reduce memoria, a costa de algo de velocidad

    # los nombres de Fashionpedia pueden ser largos ("top, t-shirt, sweatshirt"),
    # nos quedamos con el primero para que la instrucción sea natural
    garment_name = prenda["name"].split(",")[0].strip()
    instruction = INSTRUCTION_TEMPLATE.format(name=garment_name, color=COLOR_NAME)
    print(f"Instrucción: {instruction}")

    generator = torch.Generator(device=device).manual_seed(42)
    edited = pipe(
        prompt=instruction,
        image=image_pil,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        image_guidance_scale=IMAGE_GUIDANCE_SCALE,
        generator=generator,
    ).images[0]

    # el modelo ha editado la imagen entera, pero nos quedamos con su resultado 
    # solo dentro de la máscara y con la imagen original fuera de ella
    final = Image.composite(edited, image_pil, mask_pil)

    # mostramos resultado
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image_pil)
    axes[0].set_title(f"Original (índice {INDEX})")
    axes[1].imshow(mask_pil, cmap="gray")
    axes[1].set_title(f"Máscara refinada ({prenda['name']})")
    axes[2].imshow(edited)
    axes[2].set_title("Salida del modelo (imagen entera)")
    axes[3].imshow(final)
    axes[3].set_title(f"Resultado final: '{COLOR_NAME}'")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("prueba_recoloreado_pix2pix.png", dpi=150)
    print("Guardado en prueba_recoloreado_pix2pix.png")
    plt.show()


if __name__ == "__main__":
    main()
