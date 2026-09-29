"""
Clasificador binario "human" vs "not_human", usando ResNet18 preentrenado.

"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Usando GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("No se ha detectado GPU. Usando CPU.")
    return device


def get_model(pretrained=True, freeze_backbone=True):
    """
    Crea un ResNet18 adaptado a clasificación binaria 
    (1 salida, con sigmoid aplicado luego fuera del modelo).
    freeze_backbone=True congela todas las capas preentrenadas y
    solo entrena la última capa nueva.
    
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        # "Congelar" significa que estos pesos no se van a actualizar
        # durante el entrenamiento: requires_grad=False evita que se
        # calculen gradientes para ellos, ahorrando cómputo y memoria.
        for param in model.parameters():
            param.requires_grad = False

    # la última capa de ResNet18 (model.fc) está pensada para las 1000
    # clases de ImageNet. La sustituimos por una capa nueva de salida 1
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    return model