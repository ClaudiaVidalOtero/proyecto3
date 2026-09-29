"""
Entrena el clasificador binario "human" vs "not_human".

"""

import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import get_device, get_model


DATA_DIR = "dataset/dataset human_classificator"
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = "checkpoints/clasificador_best.pth"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        # las etiquetas vienen como enteros (0 o 1); BCEWithLogitsLoss
        # espera float, con forma (batch, 1) igual que la salida del modelo
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        # convertimos la salida del modelo a predicción 0/1 para calcular precisión
        predicted = (torch.sigmoid(logits) >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(logits) >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train():
    device = get_device()

    train_loader, val_loader, class_names = get_dataloaders(
        DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
    )
    print(f"Índice 0 = '{class_names[0]}', índice 1 = '{class_names[1]}'")

    model = get_model(pretrained=True, freeze_backbone=True)
    model.to(device)

    # BCEWithLogitsLoss combina sigmoid + binary cross-entropy en una sola operación
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Época {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            import os
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict(), "class_names": class_names},
                CHECKPOINT_PATH,
            )
            print(f"  -> Nuevo mejor modelo guardado (val_acc={val_acc:.2%})")

    print("\nEntrenamiento finalizado.")


if __name__ == "__main__":
    train()
