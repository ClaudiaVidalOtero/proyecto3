"""
Lista de comandos que ejecutan las diferentes funciones del modelo
(train, evaluate, visualize, recolor...)

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
#   - Rutas, epochs, batch size, etc.: se cambian en src/config.py
#   - Si en Windows falla el DataLoader: WORKERS = 0 en src/config.py
# =====================================================================

import argparse
import os

import matplotlib

# "Agg" = backend sin ventana: solo guarda a archivo. Hay que ponerlo ANTES de importar pyplot,
# si no matplotlib intenta abrir una ventana y puede crashear
matplotlib.use("Agg")

from src import config as C

# por defecto se usa el mejor checkpoint (el de mayor mAP), no el último
DEFAULT_CKPT = os.path.join(C.CHECKPOINT_DIR, "best.pth")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}  # extensiones que consideramos imágenes al leer carpetas


# TRAIN
def cmd_train(args):
    """Entrena Mask R-CNN. Reanuda solo desde checkpoints/last.pth si existe."""
    # los imports van dentro de cada comando para que `python main.py evaluate`
    # no tenga que cargar cosas que no usa y así arrancar más rápido
    import json

    from src.dataset import build_train_val
    from src.engine import train_model
    from src.model import get_device, get_model
    from src.viz import plot_history

    device = get_device()
    train_ds, val_ds = build_train_val()
    print(f"Train: {len(train_ds)} imágenes | Val: {len(val_ds)} imágenes")

    # creamos el modelo (con pesos de COCO si PRETRAINED es True) y lo mandamos a la GPU
    model = get_model(pretrained=C.PRETRAINED).to(device)
    history = train_model(model, train_ds, val_ds, device)

    # guardamos el historial en JSON y, si hay datos, la gráfica de las curvas
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(C.OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    if history["train_loss"]:  # si el entrenamiento ya estaba completo, la lista sale vacía y no hay nada que pintar
        plot_history(history, os.path.join(C.OUTPUT_DIR, "curves.png"))
    print("Listo. Checkpoints en:", C.CHECKPOINT_DIR)


# EVALUATE
def cmd_evaluate(args):
    """Evaluación en test: pérdida, mAP (segm y bbox) y AP por categoría."""
    import math

    from torch.utils.data import DataLoader

    from src.dataset import build_test
    from src.engine import ap_por_clase, collate_fn, evaluate_loss, evaluate_map
    from src.model import get_device, load_model

    device = get_device()
    test_ds = build_test()
    model = load_model(args.checkpoint, device)  # args.checkpoint viene del --checkpoint de la terminal

    # primero la pérdida en test, para eso hace falta un DataLoader
    loader = DataLoader(test_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                        num_workers=C.WORKERS, collate_fn=collate_fn)
    print(f"Pérdida final en test: {evaluate_loss(model, loader, device):.4f}\n")

    # y luego el mAP (esta función crea su propio DataLoader por dentro)
    metrics, segm_eval = evaluate_map(model, test_ds, device, batch_size=C.BATCH_SIZE, workers=C.WORKERS)
    print("Resumen test:", {k: round(v, 4) for k, v in metrics.items()})

    if segm_eval is not None:
        print("\nAP de segmentación por categoría (de mayor a menor):")
        ap = ap_por_clase(segm_eval, test_ds.label_map())
        # ordenamos por AP de mayor a menor. Los NaN (clases sin ejemplos en test) no se pueden
        # comparar con números, así que se les da -1 para que acaben al final de la lista
        for nombre, v in sorted(ap.items(), key=lambda kv: -1 if math.isnan(kv[1]) else kv[1], reverse=True):
            print(f"  {nombre:40s} {v:.3f}")  # :40s alinea los nombres para que las columnas queden rectas


# VISUALIZE
def cmd_visualize(args):
    """
    Ground truth vs. predicción sobre imágenes aleatorias de test. 
    Guarda output en outputs/predictions.png
    """
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

    # elegimos n índices al azar sin repetir (el min evita pedir más imágenes de las que hay)
    indices = random.sample(range(len(test_ds)), k=min(args.n, len(test_ds)))
    # una fila por imagen y dos columnas (GT | predicción). squeeze=False hace que axes sea siempre
    # una matriz 2D, incluso con una sola fila (si no, axes[row, 0] fallaría con n=1)
    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 5 * len(indices)), squeeze=False)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, target = test_ds[idx]
            # el modelo espera una lista de imágenes. [0] es el resultado de la nuestra
            pred = model([image.to(device)])[0]
            plot_prediction(image, target, pred, axes[row, 0], axes[row, 1], label2name, C.SCORE_THRESHOLD)

    plt.tight_layout()
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    out = os.path.join(C.OUTPUT_DIR, "predictions.png")
    fig.savefig(out, dpi=120)
    print("Guardado en", out)


# SEGMENT
def _rutas_aleatorias_test(n):
    """n imágenes aleatorias del test, a resolución original."""
    import random

    from src.dataset import build_test

    test_ds = build_test()
    indices = random.sample(range(len(test_ds)), k=min(n, len(test_ds)))
    # no usamos test_ds[i] porque eso devolvería la imagen ya reducida; queremos la ruta del
    # archivo para abrirlo luego a resolución original. Se saca de índice -> id de imagen -> nombre de archivo
    return [os.path.join(test_ds.images_dir, test_ds.coco.imgs[test_ds.image_ids[i]]["file_name"])
            for i in indices]


def _rutas_desde_input(ruta):
    """Una imagen o una carpeta (recursiva)."""
    if not os.path.exists(ruta):
        # SystemExit corta el programa con un mensaje de error limpio (sin traceback gigante)
        raise SystemExit(f"ERROR: no existe la ruta '{ruta}' (se busca desde: {os.getcwd()})")
    if os.path.isfile(ruta):
        return [ruta]  # es una sola imagen
    # es una carpeta: os.walk recorre todo (subcarpetas incluidas) y nos quedamos con los archivos
    # cuya extensión (en minúsculas) esté en IMG_EXTS
    paths = sorted(os.path.join(d, f) for d, _, fs in os.walk(ruta)
                   for f in fs if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not paths:
        raise SystemExit(f"ERROR: no se encontraron imágenes ({', '.join(sorted(IMG_EXTS))}) en '{ruta}'")
    return paths


def cmd_segment(args):
    """
    Mask R-CNN + SAM 2 + prenda + partes − cierres. Guarda *_alpha.png y *_overlay.png por imagen.
    Sin --input elige --n imágenes aleatorias del conjunto de test.
    """
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    from src.dataset import load_label_map
    from src.model import get_device, load_model
    from src.pipeline import check_categories, load_sam, segmentar_prendas
    from src.viz import overlay

    # aquí usamos load_label_map (lee solo el JSON) en vez de cargar el dataset entero, que es más ligero
    label2name = load_label_map()
    check_categories(label2name)  # avisa si algún nombre de PRENDAS/PARTES/CIERRES está mal escrito

    device = get_device()
    model = load_model(args.checkpoint, device).eval()
    usar_sam = not args.no_sam  # --no-sam apaga SAM
    predictor = load_sam(device.type) if usar_sam else None  # si no usamos SAM ni lo cargamos para ahorrar tiempo

    # si no se pasa --input estamos en modo aleatorio (imágenes de test)
    modo_aleatorio = args.input is None
    paths = _rutas_aleatorias_test(args.n) if modo_aleatorio else _rutas_desde_input(args.input)
    comparar = args.compare or modo_aleatorio      # el notebook siempre muestra la comparación
    origen = "aleatorias del test" if modo_aleatorio else "de la ruta indicada"
    print(f"{len(paths)} imagen(es) {origen}")

    # Carpetas separadas según el modo, para que no se mezclen los resultados
    out_dir = os.path.join(C.OUTPUT_DIR, "segmentation", "sam" if usar_sam else "no_sam")
    os.makedirs(out_dir, exist_ok=True)

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]  # nombre del archivo sin extensión, para nombrar las salidas
        img = np.array(Image.open(path).convert("RGB"))
        prendas = segmentar_prendas(model, predictor, img, device, label2name, usar_sam=usar_sam)

        for i, p in enumerate(prendas):
            # convierte el nombre de la prenda en algo seguro para un nombre de archivo,
            # es decir, sustituye espacios por "_"
            slug = re.sub(r"[^a-z0-9]+", "_", p["nombre"].lower()).strip("_")
            # Recorte a color (RGBA) de píxeles originales de la prenda + máscara suave en el canal alfa
            # dstack pega el alfa (0-255) como 4º canal. Fuera de la prenda queda transparente
            recorte = np.dstack([img, (p["alpha"] * 255).astype(np.uint8)])
            Image.fromarray(recorte, "RGBA").save(f"{out_dir}/{stem}_{i}_{slug}_alpha.png")
        # una sola imagen con todas las prendas coloreadas encima de la foto
        Image.fromarray(overlay(img, [p["mask_final"] for p in prendas])).save(f"{out_dir}/{stem}_overlay.png")

        # texto resumen de lo detectado: nombre (score, +partes, −cierres). Si no hay prendas, un mensaje por defecto
        resumen = ", ".join(f'{p["nombre"]} ({p["score"]:.2f}, +{len(p["partes"])} partes, −{p["cierres"]} cierres)'
                            for p in prendas) or "ninguna prenda detectada"

        if comparar:
            if usar_sam:
                # 3 imágenes: la original, Mask R-CNN sin refinar y elresultado final con SAM
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img); axes[0].set_title("Original")
                axes[1].imshow(overlay(img, [p["mask_rcnn"] for p in prendas])); axes[1].set_title("Mask R-CNN (sin refinar)")
                axes[2].imshow(overlay(img, [p["mask_final"] for p in prendas])); axes[2].set_title("SAM 2 + partes − cierres")
                # el texto va justo debajo del panel (coordenadas relativas al eje y negativo = fuera, por abajo)
                axes[2].text(0, -0.02, resumen, transform=axes[2].transAxes, va="top", fontsize=8)
            else:   # sin SAM no hay nada que comparar: solo original | resultado
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(img); axes[0].set_title("Original")
                axes[1].imshow(overlay(img, [p["mask_final"] for p in prendas])); axes[1].set_title("Mask R-CNN + partes − cierres")
                axes[1].text(0, -0.02, resumen, transform=axes[1].transAxes, va="top", fontsize=8)
            for ax in axes:
                ax.axis("off")  # quitamos los ejes con números, no aportan nada en una foto
            fig.tight_layout()
            fig.savefig(f"{out_dir}/{stem}_compare.png", dpi=110)
            plt.close(fig)  # liberamos la figura, importante dentro de un bucle con muchas imágenes

        print(f"{os.path.basename(path)}: {resumen}")

    print("Resultados en", out_dir)


# RECOLOR
def cmd_recolor(args):
    """
    Segmenta las prendas (Mask R-CNN + SAM 2) y cambia su tono en HSV.
    Sin --input elige --n imágenes aleatorias del test (como segment).
    """
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # main.py arranca con backend "Agg" (sin ventanas), y aquí cambiamos a uno interactivo
    if args.pick:   
        # (con --pick necesitamos una ventana de verdad para el selector de color, y Agg no puede abrirla)
        try:
            plt.switch_backend("TkAgg")
        except Exception as e:
            # si no hay tkinter instalado falla aquí. Mejor paramos con un mensaje claro para que no falle más adelante
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
    predictor = load_sam(device.type)      
    # el recolor siempre usa SAM 2 porque las máscaras refinadas 
    # dan bordes mucho más limpios para recolorear

    # misma lógica que en segment, sin --input coge imágenes aleatorias de test
    modo_aleatorio = args.input is None
    paths = _rutas_aleatorias_test(args.n) if modo_aleatorio else _rutas_desde_input(args.input)
    comparar = args.compare or modo_aleatorio
    print(f"{len(paths)} imagen(es) {'aleatorias del test' if modo_aleatorio else 'de la ruta indicada'}")

    out_dir = os.path.join(C.OUTPUT_DIR, "recolor")
    hue, sat, val = args.hue, args.sat, args.val     # con --pick se van recordando entre imágenes
    # (o sea, el color que se elige en una imagen será el valor de partida en la siguiente)
    os.makedirs(out_dir, exist_ok=True)

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        img = np.array(Image.open(path).convert("RGB"))
        prendas = segmentar_prendas(model, predictor, img, device, label2name, usar_sam=True)
        if args.prenda:
            # filtro por texto, ya que "dress" también pillaría cualquier nombre que lo contenga. El .lower() es para ignorar mayúsculas
            prendas = [p for p in prendas if args.prenda.lower() in p["nombre"].lower()]

        if not prendas:
            # si no hay nada que recolorear pasamos a la siguiente imagen (continue)
            # El mensaje cambia según si el usuario había filtrado por --prenda o no
            print(f"{os.path.basename(path)}: ninguna prenda detectada"
                  + (f" que coincida con '{args.prenda}'" if args.prenda else ""))
            continue

        if args.pick:
            # abre la ventana del selector y espera (es bloqueante) hasta que se pulse "Aplicar" o "Saltar"
            elegido = SelectorColor(img, prendas, hue, sat, val).mostrar()
            if elegido is None:
                print(f"{os.path.basename(path)}: saltada")
                continue
            hue, sat, val = elegido  # desempaquetamos la tupla (tono, saturación, brillo) elegida

        # cada prenda se recolorea con su propia máscara (su punto más brillante
        # se calcula por separado) y se mezcla con la máscara suave (alpha)
        resultado = img
        for p in prendas:
            # el resultado de recolorear una prenda es la entrada de la siguiente,
            # así se acumulan todos los cambios en la misma imagen
            resultado = recolor_hsv(resultado, p["mask_final"], hue,
                                    target_saturation=sat, target_value=val,
                                    alpha=p["alpha"])

        Image.fromarray(resultado).save(f"{out_dir}/{stem}_h{hue}.png")  # el tono va en el nombre del archivo

        if comparar:
            # 3 paneles: original | prendas segmentadas | resultado recoloreado
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


# SISTEMA DE COMANDOS
def main():
    # RawDescriptionHelpFormatter mantiene el formato del docstring de arriba tal cual en el --help
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # un "subparser" por comando (train, evaluate...), cada uno con sus propios argumentos
    sub = parser.add_subparsers(dest="command", required=True)

    # set_defaults(func=...) asocia cada comando con su función, que se llama al final con args.func(args)
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
    # store_true = flag sin valor: si lo escribes vale True, si no, False
    p.add_argument("--no-sam", action="store_true", help="Solo Mask R-CNN, sin refinar")
    p.add_argument("--compare", action="store_true",
                   help="Guarda también original | Mask R-CNN | final (siempre activo sin --input)")
    p.set_defaults(func=cmd_segment)

    # comando nuevo: recolor, con sus propios argumentos (tono, saturación, brillo, selector, filtro de prenda...)
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

    args = parser.parse_args()  # lee lo que se ha escrito en la terminal
    args.func(args)  # y ejecuta la función del comando elegido


if __name__ == "__main__":
    main()
