"""
Carga dataset de Fashionpedia (formato COCO) para Mask R-CNN.

"""
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

from . import config as C  # aquí están las rutas y constantes (MAX_SIDE, seeds...)


class FashionpediaDataset(Dataset):
    """Conserva la relación de aspecto, recalcula las cajas desde las máscaras y,
    con train=True, aplica augmentation (flip horizontal + color jitter)."""

    def __init__(self, images_dir, annotations_file, max_side=C.MAX_SIDE, train=False,
                 hflip_prob=0.5, color_jitter=True):
        # guardamos los parámetros tal cual para usarlos luego en __getitem__
        self.images_dir = images_dir
        self.max_side = max_side
        self.train = train
        self.hflip_prob = hflip_prob
        # si color_jitter está activo creamos el transform, si no lo dejamos en None
        self.jitter = (ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02)
                       if color_jitter else None)

        # cargamos el JSON de anotaciones con pycocotools
        self.coco = COCO(annotations_file)
        all_ids = list(sorted(self.coco.imgs.keys()))
        # nos quedamos solo con las imágenes que de verdad existen en disco
        # (por si queremos probar con solo un trozo del dataset)
        self.image_ids = [i for i in all_ids
                          if os.path.isfile(os.path.join(images_dir, self.coco.imgs[i]["file_name"]))]
        if len(self.image_ids) < len(all_ids):
            print(f"Aviso: de {len(all_ids)} imágenes en el JSON, solo se encontraron "
                  f"{len(self.image_ids)} en '{images_dir}'.")

        # Mask R-CNN reserva el 0 para el fondo, así que las clases empiezan en 1
        # (por eso el i + 1). Los ids originales de COCO pueden tener huecos
        category_ids = sorted(self.coco.getCatIds())
        self.categoryid_to_index = {cid: i + 1 for i, cid in enumerate(category_ids)}
        self.index_to_categoryid = {v: k for k, v in self.categoryid_to_index.items()}  # el mapa inverso
        self.catid_to_name = {c["id"]: c["name"] for c in self.coco.loadCats(category_ids)}

    def label_map(self):
        """índice de clase del modelo (1..46) -> nombre"""
        # índice del modelo -> id de COCO -> nombre
        return {idx: self.catid_to_name[cid] for idx, cid in self.index_to_categoryid.items()}

    def with_mode(self, train):
        """Misma data pero con/sin augmentation (no duplica el JSON en memoria)."""
        # copy.copy hace una copia que cambia solo el flag train. Así no cargamos el JSON dos veces
        other = copy.copy(self)
        other.train = train
        return other

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # sacamos el id de la imagen y su info (tamaño, nombre de archivo) del JSON
        image_id = self.image_ids[idx]
        info = self.coco.imgs[image_id]
        width, height = info["width"], info["height"]

        # abrimos la imagen y leemos las máscaras anotadas de todas las prendas de esa imagen
        image = Image.open(os.path.join(self.images_dir, info["file_name"])).convert("RGB")
        masks, labels = self._decode_instances(image_id, height, width)

        # 1) Reescalado sin deformar (solo si se pasa de max_side)
        # el min(1.0, ...) hace que nunca se agrande una imagen, solo se reduce
        scale = min(1.0, self.max_side / max(width, height))
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))  # max(1,..) evita tamaño 0
        if (image.width, image.height) != (new_w, new_h):
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        if scale < 1.0:
            # las máscaras hay que reducirlas igual que la imagen. Se pasan a 0/255 para poder interpolar, 
            # y luego se vuelve a binarizar con el umbral de 127
            masks = [(np.array(Image.fromarray(m * 255).resize((new_w, new_h), Image.Resampling.BILINEAR)) > 127)
                     .astype(np.uint8) for m in masks]

        # 2) Augmentation (solo train)
        if self.train:
            if random.random() < self.hflip_prob:  # 50% de probabilidad de voltear la imagen
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                # m[:, ::-1] invierte las columnas (flip horizontal en numpy);
                # ascontiguousarray porque el slice deja un array con strides raros que suele dar problemas en torch
                masks = [np.ascontiguousarray(m[:, ::-1]) for m in masks]
            if self.jitter is not None:
                # el jitter solo toca la imagen, las máscaras no cambian
                image = self.jitter(image)

        # 3) Cajas desde las máscaras finales (se descartan las vacías)
        # las calculamos aquí (después del resize y el flip) para que siempre cuadren
        boxes, kept_masks, kept_labels = [], [], []
        for m, label in zip(masks, labels):
            rows, cols = np.where(m)  # coordenadas de todos los píxeles que valen 1
            if rows.size == 0:
                continue  # tras reducir la imagen una máscara muy pequeña puede quedarse en nada
            # formato [x_min, y_min, x_max, y_max]; el +1 es porque el borde derecho/inferior es exclusivo
            boxes.append([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1])
            kept_masks.append(m)
            kept_labels.append(label)

        # imagen a tensor: HWC -> CHW (lo que espera PyTorch) y de 0-255 a 0-1
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        n = len(kept_masks)
        # si no queda ninguna instancia devolvemos un tensor vacío con la forma correcta,
        # porque np.stack de una lista vacía da error
        masks_t = (torch.from_numpy(np.stack(kept_masks)).to(torch.uint8) if n
                   else torch.zeros((0, new_h, new_w), dtype=torch.uint8))
        # diccionario con el formato que pide torchvision para Mask R-CNN
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),  # reshape para que con 0 cajas siga siendo (0, 4)
            "labels": torch.tensor(kept_labels, dtype=torch.int64),
            "masks": masks_t,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            # el área sale de contar píxeles de la máscara (no del área del JSON, que ya no vale tras el resize)
            "area": masks_t.flatten(1).sum(dim=1).to(torch.float32),
            "iscrowd": torch.zeros((n,), dtype=torch.int64),  # todo a 0: no usamos crowds
        }
        return image_tensor, target

    def _decode_instances(self, image_id, height, width):
        masks, labels = [], []
        # recorremos todas las anotaciones (prendas) de esta imagen
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id)):
            seg = ann["segmentation"]
            # COCO guarda las máscaras de 3 formas distintas, hay que tratar cada una:
            if isinstance(seg, list):
                # polígonos (una prenda puede tener varios). 
                # Los pasamos a RLE y los fusionamos en uno solo
                rle = coco_mask.merge(coco_mask.frPyObjects(seg, height, width))
            elif isinstance(seg["counts"], list):
                # RLE sin comprimir hay que convertirlo a RLE comprimido
                rle = coco_mask.frPyObjects(seg, height, width)
            else:
                rle = seg  # ya viene como RLE comprimido, listo
            m = coco_mask.decode(rle) # array binario de la máscara
            if m.ndim == 3:
                m = np.any(m, axis=2)  # si salen varias capas las juntamos en una
            m = m.astype(np.uint8)
            if m.sum() == 0:
                continue  # máscara vacía, pasamos de ella
            masks.append(m)
            labels.append(self.categoryid_to_index[ann["category_id"]])  # id de COCO -> índice del modelo
        return masks, labels


def unwrap(dataset):
    """El FashionpediaDataset que hay debajo de uno o varios Subset."""
    # un Subset puede envolver a otro Subset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def build_train_val():
    """(train con augmentation, val sin augmentation), con split"""
    full_aug = FashionpediaDataset(C.TRAIN_IMAGES_DIR, C.TRAIN_ANNOTATIONS, train=C.AUGMENT)
    full_plain = full_aug.with_mode(train=False)  # mismo dataset pero sin augmentation, para validar
    n_val = int(C.VAL_FRACTION * len(full_aug))
    # generador con semilla fija para que el split sea siempre el mismo en cada ejecución
    g = torch.Generator().manual_seed(C.SPLIT_SEED)
    idx = torch.randperm(len(full_aug), generator=g).tolist()  # índices barajados
    # los primeros índices van a train y los últimos n_val a validación
    # (mismos índices en ambos, así no hay imágenes repetidas entre train y val)
    return Subset(full_aug, idx[:len(idx) - n_val]), Subset(full_plain, idx[len(idx) - n_val:])


def build_test():
    return FashionpediaDataset(C.TEST_IMAGES_DIR, C.TEST_ANNOTATIONS, train=False)


def load_label_map(annotations_file=C.TEST_ANNOTATIONS):
    """índice -> nombre leyendo solo las categorías del JSON (sin cargar dataset)."""
    # leemos el JSON directamente, sin pasar por COCO, que es más rápido para inferencia
    with open(annotations_file, "r", encoding="utf-8") as f:
        cats = json.load(f)["categories"]
    # importante: ordenar por id y numerar desde 1 igual que en la clase de arriba,
    # si no los índices no coincidirían con los del entrenamiento
    return {i + 1: c["name"] for i, c in enumerate(sorted(cats, key=lambda c: c["id"]))}
