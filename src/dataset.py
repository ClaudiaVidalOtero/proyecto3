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

        # lista de todos los image_id que aparecen en el JSON (46.000 más o menos)
        all_images_ids = list(sorted(self.coco.imgs.keys()))

        # Si "images_dir" no contiene todas las imágenes del JSON (train_no_humans solo tiene 500 de las 45000), 
        # nos quedamos solo con los image_id cuyo archivo existe de verdad en esa carpeta.
        self.image_ids = [
            img_id for img_id in all_images_ids
            if os.path.isfile(
                os.path.join(self.images_dir, self.coco.loadImgs(img_id)[0]["file_name"])
            )
        ]

        n_total = len(all_images_ids)
        n_encontradas = len(self.image_ids)
        if n_encontradas < n_total:
            print(
                f"Aviso: de {n_total} imágenes en el JSON, solo se encontraron "
                f"{n_encontradas} en '{self.images_dir}'. Se usarán solo esas."
            )

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
        A partir de una imagen, construye su máscara. Devuelve la imagen y su máscara ya convertidas ambas a tensores
        """
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]

        # carga la imagen del disco
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        # constuye la mascara
        mask = self._build_mask(img_info, image_id)

        # redimensiona la imagen para que todas tengan el mismo tamaño (necesario para la entrada del modelo)
        image_np = np.array(Image.fromarray(image_np).resize((self.image_size, self.image_size), Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize((self.image_size, self.image_size), Image.NEAREST))

        # convertimos a tensores de PyTorch
        # la imagen pasa de (alto, ancho, 3canales) a (3canales, alto, ancho), y de valores 0-255 a 0.0-1.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor


    def _build_mask(self, img_info, image_id):
        """
        A partir de las anotaciones de segmentación, construimos la máscara de una imagen
        """
        # construimos la máscara sobre una "imagen de zeros"
        height, width = img_info["height"], img_info["width"]
        mask = np.zeros((height, width), dtype=np.uint8)

        # cogemos todas las anotaciones (prendas de ropa) que aparecen en la imagen
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        # ordenamos de prenda más grande a prenda más pequeña por, si se solapan varias, seleccionamos la más grande para la máscara
        annotations = sorted(annotations, key=lambda a: a.get("area", 0), reverse=True)

        for annotation in annotations:
            class_index = self.categoryid_to_index[annotation["category_id"]]

            # la segmentación puede venir como polígono (lista) o como RLE (dict)
            segmentation = annotation["segmentation"]
            if isinstance(segmentation, list):
                rles = coco_mask.frPyObjects(segmentation, height, width)
                rle = coco_mask.merge(rles)
            elif isinstance(segmentation["counts"], list):
                rle = coco_mask.frPyObjects(segmentation, height, width)
            else:
                rle = segmentation

            instance = coco_mask.decode(rle).astype(bool)
            # pintamos esa zona segmentada con el id de la categoría
            mask[instance] = class_index

        return mask



if __name__ == "__main__":
    ANNOTATIONS_FILE = "dataset/instances_attributes_train2020.json"
    IMAGES_DIR = "dataset/train_no_humans"

    dataset = FashionpediaDataset(IMAGES_DIR, ANNOTATIONS_FILE)
    print(f"El dataset tiene {len(dataset)} imágenes")

    # seleccionar una imagen aleatoria para mostrar por pantalla
    random_idx = random.randint(0, len(dataset) - 1)
    print(f"Mostrando imagen con índice aleatorio: {random_idx}")
    image_tensor, mask_tensor = dataset[random_idx]

    print("Clases de prendas presentes en esta máscara:", mask_tensor.unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # convertimos el tensor de (C, H, W) a (H, W, C) para matplotlib
    image_display = image_tensor.permute(1, 2, 0).numpy()
    mask_display = mask_tensor.numpy()

    axes[0].imshow(image_display)
    axes[0].set_title(f"Imagen (Índice {random_idx})")
    axes[0].axis("off")

    im_mask = axes[1].imshow(mask_display, cmap="jet")
    axes[1].set_title("Máscara de Segmentación")
    axes[1].axis("off")
    fig.colorbar(im_mask, ax=axes[1], shrink=0.7)

    plt.tight_layout()
    plt.show()