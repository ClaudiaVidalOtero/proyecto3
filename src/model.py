import torch

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

 
# CONFIGURACIÓN

# Fashionpedia tiene 46 categorías.
# Mask R-CNN necesita añadir una clase adicional para background.
NUM_CLASSES = 47


import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No se ha detectado GPU. Usando CPU.")

    return device


def get_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    Crea un modelo Mask R-CNN preentrenado y sustituye
    las cabezas de clasificación y segmentación para
    adaptarlo a Fashionpedia.
    """

    # Modelo preentrenado
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None

    model = maskrcnn_resnet50_fpn_v2(
        weights=weights
    )

      
    # Cabeza de clasificación

    # Número de características que recibe el clasificador.
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

      
    # Cabeza de segmentación
    in_features_mask = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def load_model(checkpoint_path, num_classes=NUM_CLASSES):
    """
    Carga un modelo previamente entrenado.

    Parameters
    ----------
    checkpoint_path : str
        Ruta al archivo .pth

    num_classes : int
        Número de clases incluyendo background.
    """

    model = get_model(num_classes)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    # Permitimos tanto un state_dict directamente
    # como un checkpoint con una clave "model_state_dict".
    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    return model


if __name__ == "__main__":

    print("=" * 60)
    print("Fashionpedia - Mask R-CNN")
    print("=" * 60)

    # Detectar dispositivo
    device = get_device()

    # Crear modelo
    model = get_model()

    # Mover modelo al dispositivo
    model.to(device)

    print("\nModelo creado correctamente.")
    print(f"Número de clases: {NUM_CLASSES}")
    print(f"Dispositivo: {device}")
