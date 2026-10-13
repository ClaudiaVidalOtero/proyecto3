"""
Recolorea una prenda usando InstructPix2Pix. Primero recibe la imagen original
y una instrucción de texto ("make the jacket red") y edita la imagen entera.
Para que el cambio afecte solo a la prenda, el resultado se pega sobre
la máscara que produce nuestro pipeline Mask R-CNN + SAM2.

Al ejecutarlo se abre una ventana donde eliges:
  1. una imagen del dataset (o un archivo tuyo),
  2. color, material y una petición extra libre,
  3. opcionalmente, una muestra de textura del dataset de ambientCG
     (créalo antes con:  python download_textures.py).
La muestra elegida se le pasa al modelo como imagen de referencia (IP-Adapter).

Uso:
    python recolor.py                                    # abre la interfaz
    python recolor.py --image 12 --color red --texture denim   # sin interfaz
    python recolor.py --image 12 --texture leather --texture-sample Leather012
    python recolor.py --image 12 --color red --extra "make the letters green"
"""

import argparse
import json
import os
import random
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageTk
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from winotify import Notification

# Añadimos la raíz del proyecto (carpeta padre de utils/)
# al path para que funcionen los imports de src y utils
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.dataset import FashionpediaDataset
from src.model import get_model
from src import config as C
from utils.refine_edges import refine_with_sam2


def parse_args():
    parser = argparse.ArgumentParser(description="Recolorea una prenda con InstructPix2Pix")
    parser.add_argument("--image", help="índice del dataset (ej. 38) o ruta a una imagen (ej. foto.jpg)")
    parser.add_argument("--color", help='color en inglés (ej. "red")')
    parser.add_argument("--texture", help='textura en inglés (ej. "leather"); "none" para no cambiarla')
    parser.add_argument("--extra", help='petición extra en inglés (ej. "make the letters green")')
    parser.add_argument("--texture-sample",
                        help="muestra de textura: id de ambientCG (ej. Leather012) o ruta a una imagen")
    return parser.parse_args()


# ---------------------------------------------------------------- texturas

def load_texture_index(textures_dir):
    """Lee textures/index.json. Devuelve una lista (vacía si todavía no se ha descargado nada)."""
    index_path = pathlib.Path(textures_dir) / "index.json"
    if not index_path.exists():
        return []
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    textures = []
    for entry in entries:
        path = index_path.parent / entry["file"]
        if path.exists():
            textures.append({
                "id": entry["id"],
                "category": entry.get("category", ""),
                "tags": [t.lower() for t in entry.get("tags", [])],
                "path": str(path),
            })
    return textures


def find_textures(word, textures):
    """
    Devuelve (muestras, nota) para el material escrito. Primero busca por etiquetas, id o
    categoría; si no hay nada, usa la categoría más parecida; si tampoco, las enseña todas.
    """
    if not textures:
        return [], "Todavía no hay muestras: ejecuta  python download_textures.py"
    word = (word or "").strip().lower()
    if not word or word == C.PIX2PIX_NO_TEXTURE:
        return textures, "Todas las muestras (escribe un material para filtrarlas)"

    matches = [t for t in textures
               if word in t["tags"] or word in t["id"].lower() or word == t["category"].lower()]
    if matches:
        return matches, f"{len(matches)} muestras para '{word}'"

    hint = C.PIX2PIX_TEXTURE_CATEGORY_HINTS.get(word)
    if hint:
        matches = [t for t in textures if t["category"].lower() == hint]
        if matches:
            return matches, f"'{word}' no tiene etiqueta propia en ambientCG: se muestra la categoría {hint}"

    return textures, f"Sin coincidencias para '{word}': se muestran todas las muestras"


def resolve_texture_path(value, textures):
    """Convierte un id de ambientCG (Leather012) o una ruta en la ruta de la imagen."""
    if not value:
        return None
    if os.path.isfile(value):
        return value
    for texture in textures:
        if texture["id"].lower() == value.lower():
            return texture["path"]
    print(f"Aviso: no encuentro la muestra de textura '{value}'; se ignora.")
    return None


# ---------------------------------------------------------------- interfaz

def tensor_to_thumbnail(image_tensor):
    array = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(array)
    img.thumbnail((C.PIX2PIX_THUMB_SIZE, C.PIX2PIX_THUMB_SIZE))
    return img


def choose_options(dataset, textures, default_source=None,
                   default_color=C.PIX2PIX_DEFAULT_COLOR,
                   default_texture=C.PIX2PIX_DEFAULT_TEXTURE,
                   default_extra=C.PIX2PIX_DEFAULT_EXTRA, default_texture_path=None):
    """
    Abre una ventana con dos pestañas:
      1. imágenes del dataset para elegir una (clic) o un archivo propio.
      2. Color, material, muestra de textura (clic en una miniatura) y petición extra.
    Devuelve (source, color, texture, extra, texture_path) o None si se cierra sin confirmar.
    """
    root = tk.Tk()
    root.title("Recolorear prenda")

    BLUE, GREY = "#2a7de1", "#cccccc"
    state = {"source": None, "texture": None, "result": None}
    photos, texture_photos = [], []       # referencias a las miniaturas (si no, Tk las borra)
    cells, texture_cells = {}, {}         # clave -> Label de la miniatura
    pool = []                             # muestras que encajan con el material escrito

    summary = tk.StringVar()
    texture_note = tk.StringVar()
    color_var = tk.StringVar(value=default_color)
    texture_var = tk.StringVar(value=default_texture)
    extra_var = tk.StringVar(value=default_extra)

    notebook = ttk.Notebook(root)
    notebook.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
    tab_image = ttk.Frame(notebook, padding=8)
    tab_style = ttk.Frame(notebook, padding=8)
    notebook.add(tab_image, text="1. Imagen")
    notebook.add(tab_style, text="2. Color, textura y extras")

    # ---------- pestaña 1: imagen ----------
    ttk.Label(tab_image, text="Elige una imagen", font=("TkDefaultFont", 11, "bold")).grid(
        row=0, column=0, sticky="w")
    gallery = ttk.Frame(tab_image)
    gallery.grid(row=1, column=0, pady=5)
    image_buttons = ttk.Frame(tab_image)
    image_buttons.grid(row=2, column=0, sticky="w")

    # ---------- pestaña 2: color, textura, extra ----------
    options = ttk.Frame(tab_style)
    options.grid(row=0, column=0, sticky="w")
    ttk.Label(options, text="Color:").grid(row=0, column=0, sticky="e")
    ttk.Combobox(options, textvariable=color_var, values=C.PIX2PIX_COLOR_OPTIONS, width=16).grid(
        row=0, column=1, padx=(4, 16))
    ttk.Label(options, text="Textura:").grid(row=0, column=2, sticky="e")
    texture_box = ttk.Combobox(
        options, textvariable=texture_var, values=C.PIX2PIX_TEXTURE_OPTIONS, width=16)
    texture_box.grid(row=0, column=3, padx=4)
    ttk.Label(tab_style, text="(en inglés; puedes escribir el tuyo. El color es opcional)",
              foreground="#666666").grid(row=1, column=0, sticky="w", pady=(2, 8))

    ttk.Label(tab_style, text="Muestra de textura (opcional, ambientCG)",
              font=("TkDefaultFont", 11, "bold")).grid(row=2, column=0, sticky="w")
    ttk.Label(tab_style, textvariable=texture_note, foreground="#666666").grid(
        row=3, column=0, sticky="w")
    texture_gallery = ttk.Frame(tab_style)
    texture_gallery.grid(row=4, column=0, pady=5)
    texture_buttons = ttk.Frame(tab_style)
    texture_buttons.grid(row=5, column=0, sticky="w")

    ttk.Label(tab_style, text='Petición extra (en inglés, opcional). Ej: "make the letters green"',
              font=("TkDefaultFont", 11, "bold")).grid(row=6, column=0, sticky="w", pady=(12, 0))
    ttk.Entry(tab_style, textvariable=extra_var, width=70).grid(row=7, column=0, sticky="w", pady=5)

    # ---------- lógica ----------
    def update_summary():
        image = state["source"] or "ninguna"
        texture = pathlib.Path(state["texture"]).stem if state["texture"] else "ninguna"
        summary.set(f"Imagen: {image}   |   Muestra de textura: {texture}")

    def select(source):
        state["source"] = source
        for key, label in cells.items():
            label.config(highlightbackground=BLUE if key == source else GREY)
        update_summary()

    def set_texture(path):
        state["texture"] = path
        for key, label in texture_cells.items():
            label.config(highlightbackground=BLUE if key == path else GREY)
        update_summary()

    def fill_gallery(indices):
        for widget in gallery.winfo_children():
            widget.destroy()
        photos.clear()
        cells.clear()
        for k, idx in enumerate(indices):
            tensor, _ = dataset[idx]
            photo = ImageTk.PhotoImage(tensor_to_thumbnail(tensor))
            photos.append(photo)
            label = tk.Label(gallery, image=photo, text=f"#{idx}", compound="top",
                             highlightthickness=4, highlightbackground=GREY, cursor="hand2")
            label.grid(row=k // 5, column=k % 5, padx=4, pady=4)
            label.bind("<Button-1>", lambda event, s=str(idx): select(s))
            cells[str(idx)] = label

    def new_random_set(force_index=None):
        indices = random.sample(range(len(dataset)), min(C.PIX2PIX_NUM_CHOICES, len(dataset)))
        if force_index is not None and force_index not in indices:
            indices[0] = force_index
        fill_gallery(indices)
        select(str(force_index) if force_index is not None else str(indices[0]))

    def pick_file():
        path = filedialog.askopenfilename(
            title="Elige una imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.webp *.bmp"), ("Todos", "*.*")])
        if path:
            select(path)

    def fill_texture_gallery(entries):
        for widget in texture_gallery.winfo_children():
            widget.destroy()
        texture_photos.clear()
        texture_cells.clear()
        for k, entry in enumerate(entries):
            try:
                img = Image.open(entry["path"]).convert("RGB")
            except Exception:
                continue
            img.thumbnail((C.PIX2PIX_TEXTURE_THUMB_SIZE, C.PIX2PIX_TEXTURE_THUMB_SIZE))
            photo = ImageTk.PhotoImage(img)
            texture_photos.append(photo)
            label = tk.Label(texture_gallery, image=photo, text=entry["id"], compound="top",
                             font=("TkDefaultFont", 8), highlightthickness=4,
                             highlightbackground=GREY, cursor="hand2")
            label.grid(row=k // 5, column=k % 5, padx=3, pady=3)
            # clic = elegir; clic otra vez sobre la misma = quitarla
            label.bind("<Button-1>", lambda event, p=entry["path"]:
                       set_texture(None if state["texture"] == p else p))
            texture_cells[entry["path"]] = label

    def show_texture_page(forced_path=None):
        chosen = random.sample(pool, min(C.PIX2PIX_NUM_TEXTURE_CHOICES, len(pool)))
        forced = next((t for t in textures if t["path"] == forced_path), None) if forced_path else None
        if forced and forced not in chosen:
            if chosen:
                chosen[0] = forced
            else:
                chosen = [forced]
        fill_texture_gallery(chosen)
        set_texture(forced["path"] if forced else None)

    def refresh_textures(event=None, forced_path=None):
        matches, note = find_textures(texture_var.get(), textures)
        texture_note.set(note)
        pool[:] = matches
        show_texture_page(forced_path)

    ttk.Button(options, text="Ver muestras", command=refresh_textures).grid(row=0, column=4, padx=4)
    texture_box.bind("<<ComboboxSelected>>", refresh_textures)
    texture_box.bind("<Return>", refresh_textures)

    ttk.Button(
        image_buttons, text=f"Otras {C.PIX2PIX_NUM_CHOICES} al azar",
        command=new_random_set).pack(side="left")
    ttk.Button(image_buttons, text="Usar un archivo mío...", command=pick_file).pack(
        side="left", padx=6)

    ttk.Button(texture_buttons, text="Otras muestras", command=show_texture_page).pack(side="left")
    ttk.Button(texture_buttons, text="Sin muestra", command=lambda: set_texture(None)).pack(
        side="left", padx=6)

    def confirm():
        color = color_var.get().strip()
        texture = texture_var.get().strip()
        extra = extra_var.get().strip()
        if not state["source"]:
            messagebox.showwarning("Falta la imagen", "Elige una imagen.")
            return
        nothing_to_change = (not color and (not texture or texture.lower() == C.PIX2PIX_NO_TEXTURE)
                             and not extra and not state["texture"])
        if nothing_to_change:
            messagebox.showwarning("Falta algo que cambiar",
                                   "Indica un color, una textura, una muestra o una petición extra.")
            return
        state["result"] = (
            state["source"], color, texture or C.PIX2PIX_NO_TEXTURE, extra, state["texture"])
        root.destroy()

    bottom = ttk.Frame(root)
    bottom.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
    ttk.Label(bottom, textvariable=summary).pack(side="left")
    ttk.Button(bottom, text="Recolorear", command=confirm).pack(side="right")

    # ---------- arranque ----------
    # si venía una imagen por argumento y es del dataset, la incluimos
    forced = int(default_source) if (default_source or "").isdigit() and int(default_source) < len(dataset) else None
    new_random_set(force_index=forced)
    if default_source and forced is None:
        select(default_source)  # ruta a un archivo pasada por argumento
    refresh_textures(forced_path=default_texture_path)

    root.mainloop()
    return state["result"]


# ---------------------------------------------------------------- pipeline

def load_image_tensor(source, dataset):
    """Si source es un número, usa esa imagen del dataset; si no, lo trata como ruta a un archivo."""
    if source.isdigit():
        image_tensor, _ = dataset[int(source)]
        return image_tensor

    img = Image.open(source).convert("RGB")
    scale = C.PIX2PIX_IMAGE_SIZE / max(img.size)  # misma escala que el dataset
    new_size = (round(img.width * scale), round(img.height * scale))
    return TF.to_tensor(img.resize(new_size, Image.BICUBIC))


def load_mask_rcnn(checkpoint_path):
    model = get_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def mask_to_pil(binary_mask, target_size, feather_radius, close_size=0):
    """
    Convierte una máscara booleana (H, W) en una imagen PIL en escala de grises
    del tamaño de trabajo del modelo. Blanco = zona donde se usa el resultado
    del modelo, negro = zona donde se conserva la imagen original.
    El redimensionado se hace con "nearest" (para no mezclar valores) y después
    se difumina el borde para que la transición sea gradual y no se note el corte.
    """
    mask_uint8 = binary_mask.astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_uint8)
    if close_size > 1:
        # cierre morfológico: rellena huecos más estrechos que close_size píxeles
        # (a la resolución de trabajo de Mask R-CNN) sin tocar los más anchos, como el escote
        mask_img = mask_img.filter(ImageFilter.MaxFilter(close_size)).filter(ImageFilter.MinFilter(close_size))
    mask_img = mask_img.resize((target_size, target_size), Image.NEAREST)
    if feather_radius > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(feather_radius))
    return mask_img


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataset = FashionpediaDataset(
        C.PIX2PIX_IMAGES_DIR, C.PIX2PIX_ANNOTATIONS_FILE, max_side=C.PIX2PIX_IMAGE_SIZE)
    textures = load_texture_index(C.PIX2PIX_TEXTURES_DIR)
    print(f"Muestras de textura disponibles: {len(textures)}")

    # Datos elegidos por el usuario: por argumentos (sin interfaz) o con la ventana
    args = parse_args()
    texture_path_arg = resolve_texture_path(args.texture_sample, textures)
    if args.image and (args.color or args.texture or args.extra or args.texture_sample):
        source = args.image
        color = args.color or ""
        texture = args.texture or C.PIX2PIX_NO_TEXTURE
        extra = args.extra or ""
        texture_path = texture_path_arg
    else:
        choice = choose_options(
            dataset,
            textures,
            default_source=args.image,
            default_color=args.color or C.PIX2PIX_DEFAULT_COLOR,
            default_texture=args.texture or C.PIX2PIX_DEFAULT_TEXTURE,
            default_extra=args.extra or C.PIX2PIX_DEFAULT_EXTRA,
            default_texture_path=texture_path_arg,
        )
        if choice is None:
            print("Cancelado.")
            return
        source, color, texture, extra, texture_path = choice

    # Mask R-CNN
    image_tensor = load_image_tensor(source, dataset)
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    mask_rcnn = load_mask_rcnn(C.PIX2PIX_CHECKPOINT).to(device)
    with torch.no_grad():
        model_output = mask_rcnn([image_tensor.to(device)])[0]

    label2name = {
        class_index: dataset.coco.loadCats(category_id)[0]["name"]
        for category_id, class_index in dataset.categoryid_to_index.items()
    }

    # refinar bordes con SAM 2 y agrupar (prenda + partes - cierres)
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_predictor = SAM2ImagePredictor.from_pretrained(
        C.SAM2_MODEL_ID,
        device=device.type,
    )

    _, prendas, _ = refine_with_sam2(model_output, image_np, sam2_predictor, label2name)

    if len(prendas) == 0:
        print("SAM2 + Mask R-CNN no encontraron ninguna prenda clara en esta imagen.")
        print("Prueba con otra imagen, o baja score_prenda en refine_with_sam2.")
        return

    prenda = prendas[0]  # la de mayor score
    print(f"Prenda elegida: {prenda['name']} (score={prenda['score']:.2f}, "
          f"partes={prenda['partes']}, cierres restados={prenda['cierres']})")

    # imagen y máscara al tamaño de trabajo del modelo
    image_pil = Image.fromarray(image_np).resize(
        (C.PIX2PIX_SIZE, C.PIX2PIX_SIZE), Image.BICUBIC)
    mask_pil = mask_to_pil(
        prenda["mask"], C.PIX2PIX_SIZE, C.PIX2PIX_FEATHER_RADIUS,
        C.PIX2PIX_MASK_CLOSE_SIZE)

    # cargar InstructPix2Pix
    print("\nCargando InstructPix2Pix (puede tardar la primera vez)...")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        C.PIX2PIX_MODEL_ID, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # reduce memoria, a costa de algo de velocidad

    # muestra de textura como imagen de referencia (IP-Adapter). Si falla, se sigue solo con el texto.
    texture_image = None
    if texture_path:
        try:
            print("Cargando IP-Adapter (la primera vez descarga un codificador de imagen grande)...")
            pipe.load_ip_adapter(
                C.PIX2PIX_IP_ADAPTER_REPO, subfolder="models",
                weight_name=C.PIX2PIX_IP_ADAPTER_WEIGHTS)
            pipe.set_ip_adapter_scale(C.PIX2PIX_IP_ADAPTER_SCALE)
            texture_image = Image.open(texture_path).convert("RGB")
            print(f"Muestra de textura: {texture_path} (escala {C.PIX2PIX_IP_ADAPTER_SCALE})")
        except Exception as exc:
            print(f"No se pudo usar la muestra de textura ({exc}).")
            print("Se sigue solo con el texto. Comprueba que diffusers sea >= 0.27.")

    # los nombres de Fashionpedia pueden ser largos ("top, t-shirt, sweatshirt"),
    # nos quedamos con el primero para que la instrucción sea natural
    garment_name = prenda["name"].split(",")[0].strip()
    use_texture = bool(texture) and texture.lower() != C.PIX2PIX_NO_TEXTURE
    parts = []
    if color:
        parts.append(C.PIX2PIX_INSTRUCTION_COLOR.format(name=garment_name, color=color))
    if use_texture:
        parts.append(C.PIX2PIX_INSTRUCTION_TEXTURE.format(texture=texture))
    if extra:
        parts.append(extra)
    instruction = ", ".join(parts)
    print(f"Instrucción: {instruction}")

    call_kwargs = dict(
        prompt=instruction or "keep the image as it is",
        image=image_pil,
        num_inference_steps=C.PIX2PIX_NUM_INFERENCE_STEPS,
        guidance_scale=C.PIX2PIX_GUIDANCE_SCALE,
        image_guidance_scale=C.PIX2PIX_IMAGE_GUIDANCE_SCALE,
        generator=torch.Generator(device=device).manual_seed(C.PIX2PIX_SEED),
    )
    if texture_image is not None:
        call_kwargs["ip_adapter_image"] = texture_image
    edited = pipe(**call_kwargs).images[0]

    # el modelo ha editado la imagen entera, pero nos quedamos con su resultado
    # solo dentro de la máscara y con la imagen original fuera de ella
    final = Image.composite(edited, image_pil, mask_pil)

    # mostramos resultado
    sample_id = pathlib.Path(texture_path).stem if texture_image is not None else ""
    label = ", ".join(p for p in [color, texture if use_texture else "", extra, sample_id] if p)
    fig, axes = plt.subplots(1, 4, figsize=C.PIX2PIX_FIGURE_SIZE)
    axes[0].imshow(image_pil)
    axes[0].set_title(f"Original ({source})")
    axes[1].imshow(mask_pil, cmap="gray")
    axes[1].set_title(f"Máscara refinada ({prenda['name']})")
    axes[2].imshow(edited)
    axes[2].set_title("Salida del modelo (imagen entera)")
    axes[3].imshow(final)
    axes[3].set_title(f"Resultado final: '{label}'")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    out_dir = pathlib.Path(C.PIX2PIX_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"\W+", "_", label).strip("_")[:60]  # nombre de archivo seguro y corto
    out_path = out_dir / f"{C.PIX2PIX_OUTPUT_PREFIX}{safe_label}.png"
    plt.savefig(out_path, dpi=C.PIX2PIX_SAVE_DPI)
    print(f"Guardado en {out_path}")
    Notification(
        app_id="Recolor de prendas",
        title="Prenda creada",
        msg=f"El resultado se guardó en:\n{out_path.resolve()}",
        duration="long",
    ).show()
    plt.show()


if __name__ == "__main__":
    main()
