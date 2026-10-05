"""Mask R-CNN (ResNet50-FPN v2) para Fashionpedia."""
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from . import config as C

NUM_CLASSES = 47  # 46 categorías + background


def get_device():
    if torch.cuda.is_available():
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0")
    print("No se ha detectado GPU. Usando CPU.")
    return torch.device("cpu")


def get_model(pretrained=True):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights, min_size=C.MIN_SIZE, max_size=C.MAX_SIDE)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)
    return model


def load_model(checkpoint_path, device="cpu"):
    model = get_model(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return model.to(device)
