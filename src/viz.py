"""Gráficas: curvas de entrenamiento, GT vs predicción y overlays de máscaras."""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(history["train_loss"], label="train_loss")
    a1.plot(history["val_loss"], label="val_loss")
    a1.set(xlabel="Época", ylabel="Pérdida", title="Pérdida")
    a2.plot(history["val_map"], label="segm AP@[.5:.95]")
    a2.plot(history["val_map50"], label="segm AP@.5")
    a2.set(xlabel="Época", ylabel="mAP", title="mAP de segmentación (validación)")
    for a in (a1, a2):
        a.legend(); a.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_prediction(image_tensor, target, prediction, ax_gt, ax_pred, label2name, score_threshold=0.5):
    img = image_tensor.permute(1, 2, 0).cpu().numpy()

    ax_gt.imshow(img); ax_gt.set_title("Ground truth"); ax_gt.axis("off")
    for box, label in zip(target["boxes"], target["labels"]):
        x1, y1, x2, y2 = box.tolist()
        ax_gt.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                                          edgecolor="lime", facecolor="none"))
        ax_gt.text(x1, max(y1 - 4, 0), label2name.get(int(label), str(int(label))),
                   color="lime", fontsize=7, backgroundcolor="black")

    ax_pred.imshow(img); ax_pred.set_title("Predicción del modelo"); ax_pred.axis("off")
    overlay_rgba = np.zeros((*img.shape[:2], 4))
    colors = plt.get_cmap("tab20")
    keep = prediction["scores"] >= score_threshold
    for i, (box, label, score, mask) in enumerate(zip(prediction["boxes"][keep].cpu(),
                                                      prediction["labels"][keep].cpu(),
                                                      prediction["scores"][keep].cpu(),
                                                      prediction["masks"][keep].cpu())):
        color = colors(i % 20)
        m = mask[0].numpy() > 0.5
        for c in range(3):
            overlay_rgba[..., c] = np.where(m, color[c], overlay_rgba[..., c])
        overlay_rgba[..., 3] = np.where(m, 0.5, overlay_rgba[..., 3])
        x1, y1, x2, y2 = box.tolist()
        ax_pred.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                                            edgecolor=color, facecolor="none"))
        ax_pred.text(x1, max(y1 - 4, 0), f"{label2name.get(int(label), str(int(label)))} {score:.2f}",
                     color="white", fontsize=7, backgroundcolor=color)
    ax_pred.imshow(overlay_rgba)


def overlay(img, masks, alpha=0.55):
    """Pinta cada máscara booleana con un color distinto sobre la imagen uint8 RGB."""
    out = img.astype(np.float32).copy()
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(masks):
        out[m] = (1 - alpha) * out[m] + alpha * np.array(cmap(i % 10)[:3]) * 255
    return out.astype(np.uint8)
