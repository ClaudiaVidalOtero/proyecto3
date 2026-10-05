"""Dataset de Fashionpedia (formato COCO) para Mask R-CNN."""
import copy
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch.utils.data import Dataset, Subset
from torchvision.transforms import ColorJitter

from . import config as C


class FashionpediaDataset(Dataset):
    """Conserva la relación de aspecto, recalcula las cajas desde las máscaras y,
    con train=True, aplica flip horizontal + color jitter."""

    def __init__(self, images_dir, annotations_file, max_side=C.MAX_SIDE, train=False,
                 hflip_prob=0.5, color_jitter=True):
        self.images_dir = images_dir
        self.max_side = max_side
        self.train = train
        self.hflip_prob = hflip_prob
        self.jitter = (ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02)
                       if color_jitter else None)

        self.coco = COCO(annotations_file)
        all_ids = list(sorted(self.coco.imgs.keys()))
        self.image_ids = [i for i in all_ids
                          if os.path.isfile(os.path.join(images_dir, self.coco.imgs[i]["file_name"]))]
        if len(self.image_ids) < len(all_ids):
            print(f"Aviso: de {len(all_ids)} imágenes en el JSON, solo se encontraron "
                  f"{len(self.image_ids)} en '{images_dir}'.")

        category_ids = sorted(self.coco.getCatIds())
        self.categoryid_to_index = {cid: i + 1 for i, cid in enumerate(category_ids)}
        self.index_to_categoryid = {v: k for k, v in self.categoryid_to_index.items()}
        self.catid_to_name = {c["id"]: c["name"] for c in self.coco.loadCats(category_ids)}

    def label_map(self):
        """índice de clase del modelo (1..46) -> nombre"""
        return {idx: self.catid_to_name[cid] for idx, cid in self.index_to_categoryid.items()}

    def with_mode(self, train):
        """Misma data pero con/sin augmentation (no duplica el JSON en memoria)."""
        other = copy.copy(self)
        other.train = train
        return other

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        info = self.coco.imgs[image_id]
        width, height = info["width"], info["height"]

        image = Image.open(os.path.join(self.images_dir, info["file_name"])).convert("RGB")
        masks, labels = self._decode_instances(image_id, height, width)

        # 1) Reescalado sin deformar (solo si se pasa de max_side)
        scale = min(1.0, self.max_side / max(width, height))
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        if (image.width, image.height) != (new_w, new_h):
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        if scale < 1.0:
            masks = [(np.array(Image.fromarray(m * 255).resize((new_w, new_h), Image.Resampling.BILINEAR)) > 127)
                     .astype(np.uint8) for m in masks]

        # 2) Augmentation (solo train)
        if self.train:
            if random.random() < self.hflip_prob:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                masks = [np.ascontiguousarray(m[:, ::-1]) for m in masks]
            if self.jitter is not None:
                image = self.jitter(image)

        # 3) Cajas desde las máscaras finales (se descartan las vacías)
        boxes, kept_masks, kept_labels = [], [], []
        for m, label in zip(masks, labels):
            rows, cols = np.where(m)
            if rows.size == 0:
                continue
            boxes.append([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1])
            kept_masks.append(m)
            kept_labels.append(label)

        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        n = len(kept_masks)
        masks_t = (torch.from_numpy(np.stack(kept_masks)).to(torch.uint8) if n
                   else torch.zeros((0, new_h, new_w), dtype=torch.uint8))
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(kept_labels, dtype=torch.int64),
            "masks": masks_t,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": masks_t.flatten(1).sum(dim=1).to(torch.float32),
            "iscrowd": torch.zeros((n,), dtype=torch.int64),
        }
        return image_tensor, target

    def _decode_instances(self, image_id, height, width):
        masks, labels = [], []
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id)):
            seg = ann["segmentation"]
            if isinstance(seg, list):
                rle = coco_mask.merge(coco_mask.frPyObjects(seg, height, width))
            elif isinstance(seg["counts"], list):
                rle = coco_mask.frPyObjects(seg, height, width)
            else:
                rle = seg
            m = coco_mask.decode(rle)
            if m.ndim == 3:
                m = np.any(m, axis=2)
            m = m.astype(np.uint8)
            if m.sum() == 0:
                continue
            masks.append(m)
            labels.append(self.categoryid_to_index[ann["category_id"]])
        return masks, labels


def unwrap(dataset):
    """El FashionpediaDataset que hay debajo de uno o varios Subset."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def build_train_val():
    """(train con augmentation, val sin augmentation), con split reproducible."""
    full_aug = FashionpediaDataset(C.TRAIN_IMAGES_DIR, C.TRAIN_ANNOTATIONS, train=C.AUGMENT)
    full_plain = full_aug.with_mode(train=False)
    n_val = int(C.VAL_FRACTION * len(full_aug))
    g = torch.Generator().manual_seed(C.SPLIT_SEED)
    idx = torch.randperm(len(full_aug), generator=g).tolist()
    return Subset(full_aug, idx[:len(idx) - n_val]), Subset(full_plain, idx[len(idx) - n_val:])


def build_test():
    return FashionpediaDataset(C.TEST_IMAGES_DIR, C.TEST_ANNOTATIONS, train=False)


def load_label_map(annotations_file=C.TEST_ANNOTATIONS):
    """índice -> nombre leyendo solo las categorías del JSON (para inferencia sin cargar el dataset)."""
    with open(annotations_file, "r", encoding="utf-8") as f:
        cats = json.load(f)["categories"]
    return {i + 1: c["name"] for i, c in enumerate(sorted(cats, key=lambda c: c["id"]))}
