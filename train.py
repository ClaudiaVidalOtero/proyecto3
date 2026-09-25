"""Pipeline de entrenamiento de Mask R-CNN con las clases del dataset."""

import argparse
import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset import FashionpediaDataset
from src.model import NUM_CLASSES, get_device, get_model


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Mantiene las imágenes y objetivos como listas de tamaño variable."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def move_targets_to_device(
    targets: List[Dict[str, torch.Tensor]], device: torch.device
) -> List[Dict[str, torch.Tensor]]:
    return [
        {key: value.to(device) for key, value in target.items()}
        for target in targets
    ]


def train_one_epoch(model, loader, optimizer, device, epoch: int, log_every: int = 10) -> float:
    model.train()
    total_loss = 0.0

    for step, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % log_every == 0 or step == len(loader):
            print(
                f"Época {epoch} | paso {step}/{len(loader)} "
                f"| pérdida: {loss.item():.4f}"
            )

    return total_loss / max(len(loader), 1)


def train_model(
    model,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    image_size: int = 256,
    workers: int = 0,
    checkpoint_dir: str = "checkpoints",
    max_train_images: int | None = None,
    max_val_images: int | None = None,
) -> None:
    train_dataset = build_dataset(train_dataset, max_train_images)
    val_dataset = build_dataset(val_dataset, max_val_images)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate_loss(model, val_loader, device)
        scheduler.step()
        print(f"Época {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, "last.pth"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best.pth"))


@torch.no_grad()
def evaluate_loss(model, loader, device) -> float:
    # Mask R-CNN solo devuelve pérdidas en modo train; no se calculan gradientes.
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)
        loss_dict = model(images, targets)
        total_loss += sum(loss_dict.values()).item()

    return total_loss / max(len(loader), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    train_dataset = FashionpediaDataset(
        "dataset/train",
        "dataset/instances_attributes_train2020.json",
        image_size=args.image_size,
    )
    val_dataset = FashionpediaDataset(
        "dataset/test",
        "dataset/instances_attributes_val2020.json",
        image_size=args.image_size,
    )

    model = get_model(num_classes=NUM_CLASSES, pretrained=not args.no_pretrained)
    model.to(device)

    train_model(
        model,
        train_dataset,
        val_dataset,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
    )


if __name__ == "__main__":
    main()