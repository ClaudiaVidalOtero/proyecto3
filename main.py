"""
Punto de entrada único.

    python main.py train
    python main.py evaluate
    python main.py visualize --n 4
    python main.py segment                       # como el notebook: imágenes aleatorias de test
    python main.py segment --n 5 --no-sam
    python main.py segment --input foto.jpg      # una imagen concreta
    python main.py segment --input carpeta/ --no-sam
    python main.py recolor --hue 240
"""
# =====================================================================
# CÓMO EJECUTAR (desde la raíz del proyecto, con el .venv activado)
# =====================================================================
#
# Ayuda general / de un comando
#   python main.py --help
#   python main.py segment --help
#
# 1) EVALUAR el modelo en test (pérdida, mAP segm/bbox y AP por categoría)
#   python main.py evaluate
#   python main.py evaluate --checkpoint checkpoints/last.pth
#
# 2) VISUALIZAR predicciones (ground truth vs. modelo)
#   python main.py visualize --n 4
#   -> outputs/predictions.png
#
# 3) SEGMENTAR PRENDAS (Mask R-CNN + SAM 2 + prenda + partes − cierres)
#
#   Igual que en el notebook: SIN pasar ninguna ruta. Coge N imágenes
#   aleatorias del conjunto de test (a resolución original) y guarda la
#   figura comparativa (original | Mask R-CNN | resultado final):
#     python main.py segment               # 3 imágenes aleatorias (--n por defecto)
#     python main.py segment --n 6
#     python main.py segment --no-sam      # solo Mask R-CNN (rápido, no necesita SAM 2)
#
#   Opcionalmente, una imagen o carpeta concreta:
#     python main.py segment --input foto.jpg
#     python main.py segment --input foto.jpg --compare
#     python main.py segment --input dataset/train_classified/not_human_test --no-sam
#
#   -> outputs/segmentation/sam/      (con SAM 2)
#      outputs/segmentation/no_sam/   (con --no-sam)
#      y dentro de cada una:
#      <nombre>_<i>_<prenda>_alpha.png   (recorte RGBA a color de la prenda)
#      <nombre>_overlay.png
#      <nombre>_compare.png               (con --no-sam: solo original | resultado; siempre en modo aleatorio;
#                                                               con --input solo si pasas --compare)
#
#   SAM 2 hay que instalarlo una vez:
#     pip install git+https://github.com/facebookresearch/sam2.git
#
# 4) RECOLOREAR PRENDAS (cambia solo el tono H en HSV; conserva sombras y pliegues)
#   python main.py recolor                      # 3 imágenes aleatorias de test, verde
#   python main.py recolor --hue 240 --n 5      # azul
#   python main.py recolor --pick               # selector visual de color con vista previa en vivo
#   python main.py recolor --prenda dress       # solo vestidos
#   python main.py recolor --input foto.jpg --hue 0 --sat 220 --val 230
#   (siempre usa Mask R-CNN + SAM 2)
#   -> outputs/recolor/
#      <nombre>_h<hue>.png y <nombre>_h<hue>_compare.png
#
# 5) ENTRENAR (¡cuidado!: reanuda desde checkpoints/last.pth y sobrescribe
#    last.pth / best.pth. Haz copia de tus checkpoints antes)
#   python main.py train
#
# NOTAS
#   - Rutas, épocas, batch size, etc.: se cambian en src/config.py
#   - Si en Windows falla el DataLoader: WORKERS = 0 en src/config.py
# =====================================================================
import argparse
import os

import matplotlib

matplotlib.use("Agg")

from src import config as C

DEFAULT_CKPT = os.path.join(C.CHECKPOINT_DIR, "best.pth")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ------------------------------------------------------------------ train
def cmd_train(args):
    """Entrena Mask R-CNN. Reanuda solo desde checkpoints/last.pth si existe."""
    import json

    from src.dataset import build_train_val
    from src.engine import train_model
    from src.model import get_device, get_model
    from src.viz import plot_history

    device = get_device()
    train_ds, val_ds = build_train_val()
    print(f"Train: {len(train_ds)} imágenes | Val: {len(val_ds)} imágenes")

    model = get_model(pretrained=C.PRETRAINED).to(device)
    history = train_model(model, train_ds, val_ds, device)

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(C.OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    if history["train_loss"]:
        plot_history(history, os.path.join(C.OUTPUT_DIR, "curves.png"))
    print("Listo. Checkpoints en:", C.CHECKPOINT_DIR)


# ------------------------------------------------------------------ evaluate
def cmd_evaluate(args):
    """Evaluación en test: pérdida, mAP (segm y bbox) y AP por categoría."""
    import math

    from torch.utils.data import DataLoader

    from src.dataset import build_test
    from src.engine import ap_por_clase, collate_fn, evaluate_loss, evaluate_map
    from src.model import get_device, load_model

    device = get_device()
    test_ds = build_test()
    model = load_model(args.checkpoint, device)

    loader = DataLoader(test_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                        num_workers=C.WORKERS, collate_fn=collate_fn)
    print(f"Pérdida final en test: {evaluate_loss(model, loader, device):.4f}\n")

    metrics, segm_eval = evaluate_map(model, test_ds, device, batch_size=C.BATCH_SIZE, workers=C.WORKERS)
    print("Resumen test:", {k: round(v, 4) for k, v in metrics.items()})

    if segm_eval is not None:
        print("\nAP de segmentación por categoría (de mayor a menor):")
        ap = ap_por_clase(segm_eval, test_ds.label_map())
        for nombre, v in sorted(ap.items(), key=lambda kv: -1 if math.isnan(kv[1]) else kv[1], reverse=True):
            print(f"  {nombre:40s} {v:.3f}")


# ------------------------------------------------------------------ visualize
def cmd_visualize(args):
    """Ground truth vs. predicción sobre imágenes aleatorias de test -> outputs/predictions.png"""
    import random

    import matplotlib.pyplot as plt
    import torch

    from src.dataset import build_test
    from src.model import get_device, load_model
    from src.viz import plot_prediction

    device = get_device()
    test_ds = build_test()
    label2name = test_ds.label_map()
    model = load_model(args.checkpoint, device).eval()

    indices = random.sample(range(len(test_ds)), k=min(args.n, len(test_ds)))
    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 5 * len(indices)), squeeze=False)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, target = test_ds[idx]
            pred = model([image.to(device)])[0]
            plot_prediction(image, target, pred, axes[row, 0], axes[row, 1], label2name, C.SCORE_THRESHOLD)

    plt.tight_layout()
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    out = os.path.join(C.OUTPUT_DIR, "predictions.png")
    fig.savefig(out, dpi=120)
    print("Guardado en", out)


# ------------------------------------------------------------------ segment
def _rutas_aleatorias_test(n):
    """Igual que la celda 11.3 del notebook: n imágenes aleatorias del test, a resolución original."""
    import random

    from src.dataset import build_test

    test_ds = build_test()
    indices = random.sample(range(len(test_ds)), k=min(n, len(test_ds)))
    return [os.path.join(test_ds.images_dir, test_ds.coco.imgs[test_ds.image_ids[i]]["file_name"])
            for i in indices]


def _rutas_desde_input(ruta):
    """Una imagen o una carpeta (recursiva)."""
    if not os.path.exists(ruta):
        raise SystemExit(f"ERROR: no existe la ruta '{ruta}' (se busca desde: {os.getcwd()})")
    if os.path.isfile(ruta):
        return [ruta]
    paths = sorted(os.path.join(d, f) for d, _, fs in os.walk(ruta)
                   for f in fs if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not paths:
        raise SystemExit(f"ERROR: no se encontraron imágenes ({', '.join(sorted(IMG_EXTS))}) en '{ruta}'")
    return paths


def cmd_segment(args):
    """Mask R-CNN + SAM 2 + prenda + partes − cierres. Guarda *_alpha.png y *_overlay.png por imagen.

    Sin --input: elige --n imágenes aleatorias del conjunto de test (como el notebook).
    """
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    from src.dataset import load_label_map
    from src.model import get_device, load_model
    from src.pipeline import check_categories, load_sam, segmentar_prendas
    from src.viz import overlay

    label2name = load_label_map()
    check_categories(label2name)

    device = get_device()
    model = load_model(args.checkpoint, device).eval()
    usar_sam = not args.no_sam
    predictor = load_sam(device.type) if usar_sam else None

    modo_aleatorio = args.input is None
    paths = _rutas_aleatorias_test(args.n) if modo_aleatorio else _rutas_desde_input(args.input)
    comparar = args.compare or modo_aleatorio      # el notebook siempre muestra la comparación
    origen = "aleatorias del test" if modo_aleatorio else "de la ruta indicada"
    print(f"{len(paths)} imagen(es) {origen}")

    # Carpetas separadas según el modo, para que no se mezclen los resultados
    out_dir = os.path.join(C.OUTPUT_DIR, "segmentation", "sam" if usar_sam else "no_sam")
    os.makedirs(out_dir, exist_ok=True)

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        img = np.array(Image.open(path).convert("RGB"))
        prendas = segmentar_prendas(model, predictor, img, device, label2name, usar_sam=usar_sam)

        for i, p in enumerate(prendas):
            slug = re.sub(r"[^a-z0-9]+", "_", p["nombre"].lower()).strip("_")
            # Recorte a color (RGBA): píxeles originales de la prenda + máscara suave en el canal alfa
            recorte = np.dstack([img, (p["alpha"] * 255).astype(np.uint8)])
            Image.fromarray(recorte, "RGBA").save(f"{out_dir}/{stem}_{i}_{slug}_alpha.png")
        Image.fromarray(overlay(img, [p["mask_final"] for p in prendas])).save(f"{out_dir}/{stem}_overlay.png")

        resumen = ", ".join(f'{p["nombre"]} ({p["score"]:.2f}, +{len(p["partes"])} partes, −{p["cierres"]} cierres)'
                            for p in prendas) or "ninguna prenda detectada"

        if comparar:
            if usar_sam:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img); axes[0].set_title("Original")
                axes[1].imshow(overlay(img, [p["mask_rcnn"] for p in prendas])); axes[1].set_title("Mask R-CNN (sin refinar)")
                axes[2].imshow(overlay(img, [p["mask_final"] for p in prendas])); axes[2].set_title("SAM 2 + partes − cierres")
                axes[2].text(0, -0.02, resumen, transform=axes[2].transAxes, va="top", fontsize=8)
            else:   # sin SAM no hay nada que comparar: solo original | resultado
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(img); axes[0].set_title("Original")
                axes[1].imshow(overlay(img, [p["mask_final"] for p in prendas])); axes[1].set_title("Mask R-CNN + partes − cierres")
                axes[1].text(0, -0.02, resumen, transform=axes[1].transAxes, va="top", fontsize=8)
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(f"{out_dir}/{stem}_compare.png", dpi=110)
            plt.close(fig)

        print(f"{os.path.basename(path)}: {resumen}")

    print("Resultados en", out_dir)


# ------------------------------------------------------------------ recolor
def cmd_recolor(args):
    """Segmenta las prendas (Mask R-CNN + SAM 2) y cambia su tono en HSV.

    Sin --input: elige --n imágenes aleatorias del test (como segment).
    """
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    if args.pick:   # main.py arranca con backend "Agg" (sin ventanas): cambiamos a uno interactivo
        try:
            plt.switch_backend("TkAgg")
        except Exception as e:
            raise SystemExit(f"ERROR: no se pudo abrir una ventana interactiva ({e}). "
                             "Instala tkinter o usa --hue sin --pick.")

    from src.dataset import load_label_map
    from src.model import get_device, load_model
    from src.pipeline import check_categories, load_sam, segmentar_prendas
    from src.recolor import SelectorColor, recolor_hsv
    from src.viz import overlay

    label2name = load_label_map()
    check_categories(label2name)

    device = get_device()
    model = load_model(args.checkpoint, device).eval()
    predictor = load_sam(device.type)      # el recolor siempre usa SAM 2

    modo_aleatorio = args.input is None
    paths = _rutas_aleatorias_test(args.n) if modo_aleatorio else _rutas_desde_input(args.input)
    comparar = args.compare or modo_aleatorio
    print(f"{len(paths)} imagen(es) {'aleatorias del test' if modo_aleatorio else 'de la ruta indicada'}")

    out_dir = os.path.join(C.OUTPUT_DIR, "recolor")
    hue, sat, val = args.hue, args.sat, args.val     # con --pick se van recordando entre imágenes
    os.makedirs(out_dir, exist_ok=True)

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        img = np.array(Image.open(path).convert("RGB"))
        prendas = segmentar_prendas(model, predictor, img, device, label2name, usar_sam=True)
        if args.prenda:
            prendas = [p for p in prendas if args.prenda.lower() in p["nombre"].lower()]

        if not prendas:
            print(f"{os.path.basename(path)}: ninguna prenda detectada"
                  + (f" que coincida con '{args.prenda}'" if args.prenda else ""))
            continue

        if args.pick:
            elegido = SelectorColor(img, prendas, hue, sat, val).mostrar()
            if elegido is None:
                print(f"{os.path.basename(path)}: saltada")
                continue
            hue, sat, val = elegido

        # Cada prenda se recolorea con su propia máscara (su punto más brillante
        # se calcula por separado) y se mezcla con la máscara suave (alpha).
        resultado = img
        for p in prendas:
            resultado = recolor_hsv(resultado, p["mask_final"], hue,
                                    target_saturation=sat, target_value=val,
                                    alpha=p["alpha"])

        Image.fromarray(resultado).save(f"{out_dir}/{stem}_h{hue}.png")

        if comparar:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img); axes[0].set_title("Original")
            axes[1].imshow(overlay(img, [p["mask_final"] for p in prendas])); axes[1].set_title("Prendas segmentadas")
            axes[2].imshow(resultado); axes[2].set_title(f"Recoloreado (H={hue}°)")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(f"{out_dir}/{stem}_h{hue}_compare.png", dpi=110)
            plt.close(fig)

        print(f"{os.path.basename(path)}: recoloreadas {', '.join(p['nombre'] for p in prendas)}")

    print("Resultados en", out_dir)


# ------------------------------------------------------------------ CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Entrenar (reanuda desde last.pth)").set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="mAP y AP por categoría en test")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("visualize", help="GT vs. predicción")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--n", type=int, default=4)
    p.set_defaults(func=cmd_visualize)

    p = sub.add_parser("segment", help="Segmentar prendas con SAM 2 (sin --input: imágenes aleatorias de test)")
    p.add_argument("--input", default=None,
                   help="Imagen o carpeta. Si se omite, se eligen imágenes aleatorias del test (como el notebook)")
    p.add_argument("--n", type=int, default=3,
                   help="Nº de imágenes aleatorias de test cuando no se pasa --input (por defecto 3)")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--no-sam", action="store_true", help="Solo Mask R-CNN, sin refinar")
    p.add_argument("--compare", action="store_true",
                   help="Guarda también original | Mask R-CNN | final (siempre activo sin --input)")
    p.set_defaults(func=cmd_segment)

    p = sub.add_parser("recolor", help="Cambiar el color de las prendas (tono HSV)")
    p.add_argument("--input", default=None,
                   help="Imagen o carpeta. Si se omite, imágenes aleatorias del test")
    p.add_argument("--n", type=int, default=3, help="Nº de imágenes aleatorias sin --input (por defecto 3)")
    p.add_argument("--hue", type=int, default=120, help="Tono en grados: 0=rojo, 120=verde, 240=azul (por defecto 120)")
    p.add_argument("--sat", type=int, default=200, help="Saturación 0-255 (por defecto 200)")
    p.add_argument("--val", type=int, default=220, help="Brillo del punto más claro 0-255 (por defecto 220)")
    p.add_argument("--pick", action="store_true",
                   help="Abre un selector visual de color (barra de tonos + sliders + vista previa en vivo) por imagen")
    p.add_argument("--prenda", default=None, help="Solo recolorear prendas cuyo nombre contenga este texto (p. ej. dress)")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--compare", action="store_true", help="Guarda también original | prendas | recoloreado (siempre sin --input)")
    p.set_defaults(func=cmd_recolor)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":   # imprescindible en Windows (los workers del DataLoader reimportan este archivo)
    main()
