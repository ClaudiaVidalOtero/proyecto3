import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from src.dataset import FashionpediaDataset
from src.model import get_model, NUM_CLASSES


def load_checkpoint(path):
    model = get_model(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def draw_instances(axis, image, boxes, labels, scores, names, masks=None, threshold=0.0):
    axis.imshow(image)
    for index, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score < threshold:
            continue
        if masks is not None:
            mask = masks[index]
            axis.imshow(
                np.ma.masked_where(mask < 0.5, mask),
                cmap="jet",
                alpha=0.35,
                vmin=0,
                vmax=1,
            )
        x_min, y_min, x_max, y_max = box
        axis.add_patch(
            patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor="lime",
                linewidth=1.5,
            )
        )
        axis.text(
            x_min,
            y_min,
            f"{names.get(int(label), int(label))} ({score:.2f})",
            color="white",
            backgroundcolor="green",
            fontsize=6,
        )
    axis.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Visualiza una prediccion de Mask R-CNN.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--output", default="checkpoints/prediction_example.png")
    args = parser.parse_args()

    dataset = FashionpediaDataset(
        "dataset/test",
        "dataset/instances_attributes_val2020.json",
        image_size=args.image_size,
    )
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"--index debe estar entre 0 y {len(dataset) - 1}")

    image_tensor, target = dataset[args.index]
    model = load_checkpoint(args.checkpoint)
    with torch.no_grad():
        prediction = model([image_tensor])[0]

    names = {
        index: dataset.coco.loadCats(category_id)[0]["name"]
        for category_id, index in dataset.categoryid_to_index.items()
    }
    image = image_tensor.permute(1, 2, 0).numpy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw_instances(
        axes[0],
        image,
        target["boxes"].numpy(),
        target["labels"].numpy(),
        np.ones(len(target["labels"])),
        names,
        masks=target["masks"].numpy(),
    )
    axes[0].set_title("Anotaciones reales")
    draw_instances(
        axes[1],
        image,
        prediction["boxes"].numpy(),
        prediction["labels"].numpy(),
        prediction["scores"].numpy(),
        names,
        masks=prediction["masks"].squeeze(1).numpy(),
        threshold=args.threshold,
    )
    axes[1].set_title(f"Prediccion (score >= {args.threshold})")
    figure.tight_layout()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(f"Imagen guardada en: {args.output}")
    print(f"Detecciones totales: {len(prediction['scores'])}")
    print(f"Detecciones mostradas: {(prediction['scores'] >= args.threshold).sum().item()}")
    plt.show()


if __name__ == "__main__":
    main()
