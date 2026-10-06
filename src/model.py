"""Mask R-CNN (ResNet50-FPN v2) para Fashionpedia."""
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from . import config as C  # aquí están MIN_SIZE y MAX_SIDE

NUM_CLASSES = 47  # 46 categorías + background (el fondo cuenta como una clase más en Mask R-CNN)


def get_device():
    # si hay GPU de NVIDIA la usamos, si no, CPU
    if torch.cuda.is_available():
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0")
    print("No se ha detectado GPU. Usando CPU.")
    return torch.device("cpu")


def get_model(pretrained=True):
    # DEFAULT es los pesos preentrenados que tiene torchvision;
    # con pretrained=False no se descarga nada y el modelo empieza con pesos aleatorios
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    # min_size/max_size controlan a qué tamaño reescala el modelo las imágenes por dentro
    model = maskrcnn_resnet50_fpn_v2(weights=weights, min_size=C.MIN_SIZE, max_size=C.MAX_SIDE)

    # el modelo viene entrenado para 91 clases, así que hay que cambiar las "cabezas" 
    # por otras nuevas con 47 clases, es decir, hacer fine-tunning
    # primero la del clasificador/cajas:
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # número de features que entran a la capa
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # y ahora lo mismo con la cabeza de máscaras (256 = canales de la capa oculta)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)
    return model


def load_model(checkpoint_path, device="cpu"):
    # pretrained=False porque los pesos los vamos a cargar nosotros desde el checkpoint,
    # no tiene sentido descargar los de COCO para luego reescribirlos
    model = get_model(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # soporta los dos formatos: checkpoint completo (dict con "model_state_dict")
    # o un state_dict suelto guardado directamente
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return model.to(device)
