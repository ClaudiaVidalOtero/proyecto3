"""
Dataset para el clasificador binario "con persona/maniquí" vs "sin persona".

A diferencia de FashionpediaDataset, torchvision.datasets.ImageFolder recorre
una carpeta con subcarpetas, y usa el nombre de cada subcarpeta como
etiqueta automáticamente ("human" y "not human").

"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch


def get_transforms(image_size=224, train=True):
    """
    Transformaciones de preprocesado. image_size=224 es el tamaño
    estándar que esperan los modelos preentrenados tipo ResNet.
    """
    # normalización estándar de ImageNet: los modelos preentrenados
    # esperan que las imágenes estén normalizadas con estos valores
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], )

    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders(data_dir, image_size=224, batch_size=16, val_split=0.2, num_workers=0):
    """
    Carga todas las imágenes de data_dir (con sus subcarpetas
    human/ y not_human/) y las divide en train/val.

    Devuelve: train_loader, val_loader, class_names
    """
    # cargamos el dataset completo con las transformaciones de train.
    # ImageFolder ordena las clases alfabéticamente: human -> índice 0 y not_human -> índice 1
    full_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(image_size, train=True))
    class_names = full_dataset.classes
    print(f"Clases detectadas: {class_names}")
    print(f"Total de imágenes: {len(full_dataset)}")

    # dividimos en train (80%) y validación (20%)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)  # para que la división sea siempre igual
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # la parte de validación no debería tener augmentation (flip, color...), solo el resize y la normalización.
    val_subset.dataset = datasets.ImageFolder(data_dir, transform=get_transforms(image_size, train=False))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, class_names


if __name__ == "__main__":
    DATA_DIR = "dataset_clasificador/train"
    train_loader, val_loader, class_names = get_dataloaders(DATA_DIR)

    print(f"Batches de train: {len(train_loader)}")
    print(f"Batches de val:   {len(val_loader)}")

    images, labels = next(iter(train_loader))
    print("Forma de un batch de imágenes:", images.shape)
    print("Etiquetas del batch:", labels.tolist())
