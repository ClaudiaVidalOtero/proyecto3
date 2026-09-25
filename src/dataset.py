import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt
import random



class FashionpediaDataset(Dataset):
    """
      __len__: cuántos elementos tiene el dataset en total
      __getitem__: obtiene el elemento número i (imagen + máscara)

    PyTorch llama a __getitem__ repetidamente
    durante el entrenamiento, así que aquí solo definimos cómo se
    construye un solo par (imagen, máscara) a partir de un índice.
    """


    def __init__(self, images_dir, annotations_file, image_size=256):

        self.images_dir = images_dir    # carpeta donde están las imágenes
        self.image_size = image_size    # tamaño al que haremos resize de todas las imágenes

        self.coco = COCO(annotations_file)    # carga el JSON con las anotaciones

        # lista de todos los image_id del dataset
        self.image_ids = list(sorted(self.coco.imgs.keys()))

        # Fashionpedia tiene category_id que no son consecutivos (hay huecos). Para la red neuronal necesitamos 1, 2, 3... sin huecos
        category_ids = sorted(self.coco.getCatIds())
        self.categoryid_to_index = {category_id: i + 1 for i, category_id in enumerate(category_ids)}


    def __len__(self):
        """
        Devuelve cuántas imágenes tiene el dataset en total.
        """
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Devuelve una imagen y el target con el formato esperado por Mask R-CNN.
        """
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]

        # carga la imagen del disco
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        # Construye una máscara independiente para cada instancia anotada.
        target = self._build_target(img_info, image_id)

        # redimensiona la imagen para que todas tengan el mismo tamaño
        image_np = np.array(Image.fromarray(image_np).resize((self.image_size, self.image_size), Image.BILINEAR))

        # convertimos a tensores de PyTorch
        # la imagen pasa de (alto, ancho, 3canales) a (3canales, alto, ancho), y de valores 0-255 a 0.0-1.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        scale_x = self.image_size / img_info["width"]
        scale_y = self.image_size / img_info["height"]
        target["boxes"][:, [0, 2]] *= scale_x
        target["boxes"][:, [1, 3]] *= scale_y
        target["masks"] = torch.stack([
            torch.from_numpy(
                np.array(
                    Image.fromarray(instance_mask).resize(
                        (self.image_size, self.image_size), Image.NEAREST
                    )
                )
            )
            for instance_mask in target.pop("_masks")
        ]).to(torch.uint8) if target["boxes"].shape[0] else torch.zeros(
            (0, self.image_size, self.image_size), dtype=torch.uint8
        )
        target["area"] = target["masks"].flatten(1).sum(dim=1).to(torch.float32)

        return image_tensor, target


    def _build_target(self, img_info, image_id):
        """
        Construye el target de Mask R-CNN a partir de las anotaciones COCO.
        """
        height, width = img_info["height"], img_info["width"]
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        masks = []
        labels = []
        boxes = []
        areas = []

        for annotation in annotations:
            # la segmentación puede venir como polígono (lista) o como RLE (dict)
            segmentation = annotation["segmentation"]
            if isinstance(segmentation, list):
                rles = coco_mask.frPyObjects(segmentation, height, width)
                rle = coco_mask.merge(rles)
            elif isinstance(segmentation["counts"], list):
                rle = coco_mask.frPyObjects(segmentation, height, width)
            else:
                rle = segmentation

            instance_mask = coco_mask.decode(rle)
            if instance_mask.ndim == 3:
                instance_mask = np.any(instance_mask, axis=2)
            instance_mask = instance_mask.astype(np.uint8)
            rows, columns = np.where(instance_mask)
            if rows.size == 0:
                continue

            masks.append(instance_mask)
            labels.append(self.categoryid_to_index[annotation["category_id"]])
            boxes.append([columns.min(), rows.min(), columns.max() + 1, rows.max() + 1])
            areas.append(float(instance_mask.sum()))

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.empty((0, height, width), dtype=torch.uint8),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(masks),), dtype=torch.int64),
            "_masks": masks,
        }



if __name__ == "__main__":
    ANNOTATIONS_FILE = "dataset/instances_attributes_train2020.json"
    IMAGES_DIR = "dataset/train"

    dataset = FashionpediaDataset(IMAGES_DIR, ANNOTATIONS_FILE)
    print(f"El dataset tiene {len(dataset)} imágenes")

    # seleccionar una imagen aleatoria para mostrar por pantalla
    random_idx = random.randint(0, len(dataset) - 1)
    print(f"Mostrando imagen con índice aleatorio: {random_idx}")
    image_tensor, target = dataset[random_idx]

    print("Clases de prendas presentes:", target["labels"].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # convertimos el tensor de (C, H, W) a (H, W, C) para matplotlib
    image_display = image_tensor.permute(1, 2, 0).numpy()
    mask_display = target["masks"].sum(dim=0).numpy()

    axes[0].imshow(image_display)
    axes[0].set_title(f"Imagen (Índice {random_idx})")
    axes[0].axis("off")

    im_mask = axes[1].imshow(mask_display, cmap="jet")
    axes[1].set_title("Máscara de Segmentación")
    axes[1].axis("off")
    fig.colorbar(im_mask, ax=axes[1], shrink=0.7)

    plt.tight_layout()
    plt.show()