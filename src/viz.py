"""Gráficas: curvas de entrenamiento, ground truth vs predicción y overlays de máscaras."""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history, path):
    # una figura con 2 gráficas lado a lado: pérdida a la izquierda, mAP a la derecha
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(history["train_loss"], label="train_loss")
    a1.plot(history["val_loss"], label="val_loss")
    a1.set(xlabel="Época", ylabel="Pérdida", title="Pérdida")  # .set() deja poner varios ajustes de golpe
    a2.plot(history["val_map"], label="segm AP@[.5:.95]")
    a2.plot(history["val_map50"], label="segm AP@.5")
    a2.set(xlabel="Época", ylabel="mAP", title="mAP de segmentación (validación)")
    # leyenda y cuadrícula suave en las dos gráficas sin repetir código
    for a in (a1, a2):
        a.legend(); a.grid(alpha=0.3)
    plt.tight_layout()  # evita que se solapen títulos y ejes
    fig.savefig(path, dpi=120)
    plt.close(fig)  # cerramos la figura para no acumular memoria si se llama muchas veces


def plot_prediction(image_tensor, target, prediction, ax_gt, ax_pred, label2name, score_threshold=0.5):
    # el tensor viene en CHW y matplotlib quiere HWC, de ahí el permute
    img = image_tensor.permute(1, 2, 0).cpu().numpy()

    # panel izquierdo: ground truth
    ax_gt.imshow(img); ax_gt.set_title("Ground truth"); ax_gt.axis("off")
    for box, label in zip(target["boxes"], target["labels"]):
        x1, y1, x2, y2 = box.tolist()
        # Bounding box (esquina sup. izq., ancho, alto), no las dos esquinas, por eso se restan
        ax_gt.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                                          edgecolor="lime", facecolor="none"))
        # nombre de la clase encima de la caja 
        # max(..., 0) para que no se salga por arriba de la imagen
        # y el .get(..., str(...)) pone el número si por lo que sea no hay nombre
        ax_gt.text(x1, max(y1 - 4, 0), label2name.get(int(label), str(int(label))),
                   color="lime", fontsize=7, backgroundcolor="black")

    # panel derecho: predicción del modelo
    ax_pred.imshow(img); ax_pred.set_title("Predicción del modelo"); ax_pred.axis("off")
    # imagen RGB transparente donde iremos pintando todas las máscaras (la capa de color encima de la foto)
    overlay_rgba = np.zeros((*img.shape[:2], 4))
    colors = plt.get_cmap("tab20")  # paleta de 20 colores distintos
    # nos quedamos solo con las detecciones con score suficiente
    keep = prediction["scores"] >= score_threshold
    # recorre a la vez cajas, clases, scores y máscaras ya filtrados
    for i, (box, label, score, mask) in enumerate(zip(prediction["boxes"][keep].cpu(),
                                                      prediction["labels"][keep].cpu(),
                                                      prediction["scores"][keep].cpu(),
                                                      prediction["masks"][keep].cpu())):
        color = colors(i % 20)  # el % 20 hace que si hay más de 20 prendas se repitan los colores
        m = mask[0].numpy() > 0.5  # la máscara es (1,H,W) de probabilidades; [0] quita el canal y > 0.5 la binariza
        # pintamos el color de esta prenda solo en los píxeles de su máscara (canales R, G, B)
        for c in range(3):
            overlay_rgba[..., c] = np.where(m, color[c], overlay_rgba[..., c])
        # y ponemos esos píxeles semitransparentes (alpha 0.5) para que se vea la foto debajo
        overlay_rgba[..., 3] = np.where(m, 0.5, overlay_rgba[..., 3])
        x1, y1, x2, y2 = box.tolist()
        ax_pred.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                                            edgecolor=color, facecolor="none"))
        # etiqueta con clase + score (2 decimales) con fondo del mismo color que la caja
        ax_pred.text(x1, max(y1 - 4, 0), f"{label2name.get(int(label), str(int(label)))} {score:.2f}",
                     color="white", fontsize=7, backgroundcolor=color)
    # el overlay se dibuja al final, una sola vez, encima de la imagen
    ax_pred.imshow(overlay_rgba)


def overlay(img, masks, alpha=0.55):
    """Pinta cada máscara booleana con un color distinto sobre la imagen RGB."""
    # pasamos de uint8 a float para poder mezclar colores
    out = img.astype(np.float32).copy()    # hacemos copy para no modificar la original
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(masks):
        # mezcla (1-alpha) de la imagen con alpha del color
        # cmap devuelve valores 0-1 (y RGBA),
        # por eso se coge [:3] y se multiplica por 255
        out[m] = (1 - alpha) * out[m] + alpha * np.array(cmap(i % 10)[:3]) * 255
    return out.astype(np.uint8)  # de vuelta a uint8 para poder mostrarla y guardarla
