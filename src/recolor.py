"""
Recolorea una prenda usando InstructPix2Pix. Primero recibe la imagen original
y una instrucción de texto ("make the jacket red") y edita la imagen entera.
Para que el cambio afecte solo a la prenda, el resultado se pega sobre
la máscara que produce nuestro pipeline Mask R-CNN + SAM2.

Al ejecutarlo se abre una ventana con interfaz donde eliges:
  1. una imagen del dataset (o un archivo tuyo),
  2. color, material y una petición extra libre,
  3. opcionalmente, una muestra de textura del dataset de ambientCG
     (hay que crearlo antes con:  python download_textures.py).
La muestra elegida se le pasa al modelo como imagen de referencia (IP-Adapter).
Mientras se genera la imagen, la ventana se queda abierta con un mensaje de "Cargando".

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
import threading          # para que la interfaz no se congele mientras se genera la imagen
import traceback
import tkinter as tk      # para la interfaz gráfica
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


# TEXTURAS

def load_texture_index(textures_dir):
    """Lee textures/index.json con las texturas descargadas de AmbientCG
    Devuelve una lista (vacía si todavía no se ha descargado nada)."""
    index_path = pathlib.Path(textures_dir) / "index.json"
    if not index_path.exists():
        return []
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    textures = []
    for entry in entries:
        path = index_path.parent / entry["file"]
        # por si alguien borró algún archivo a mano, así no da error luego
        if path.exists():
            textures.append({
                "id": entry["id"],
                "category": entry.get("category", ""),
                "tags": [t.lower() for t in entry.get("tags", [])],  # todo en minúsculas para comparar fácil
                "path": str(path),
            })
    return textures


def find_textures(word, textures):
    """
    Devuelve (muestras, nota) para el material escrito. Primero busca por etiquetas, id o
    categoría. Si no hay nada, usa la categoría más parecida, si no, las enseña todas.
    """
    if not textures:
        return [], "Todavía no hay muestras: ejecuta  python download_textures.py"
    word = (word or "").strip().lower()
    if not word or word == C.PIX2PIX_NO_TEXTURE:
        return textures, "Todas las muestras (escribe un material para filtrarlas)"

    # búsqueda directa para ver si la palabra está en las tags, en el id o es la categoría
    matches = [t for t in textures
               if word in t["tags"] or word in t["id"].lower() or word == t["category"].lower()]
    if matches:
        return matches, f"{len(matches)} muestras para '{word}'"

    # si no hay nada probamos con la categoría más parecida que hay en el config
    hint = C.PIX2PIX_TEXTURE_CATEGORY_HINTS.get(word)
    if hint:
        matches = [t for t in textures if t["category"].lower() == hint]
        if matches:
            return matches, f"'{word}' no tiene etiqueta propia en ambientCG: se muestra la categoría {hint}"

    # último recurso: devolvemos todas
    return textures, f"Sin coincidencias para '{word}': se muestran todas las muestras"


def resolve_texture_path(value, textures):
    """Convierte un id de ambientCG (Leather012) o una ruta en la ruta de la imagen."""
    if not value:
        return None
    if os.path.isfile(value):
        return value
    # si no es un archivo, miramos si coincide con algún id de las muestras
    for texture in textures:
        if texture["id"].lower() == value.lower():
            return texture["path"]
    print(f"Aviso: no encuentro la muestra de textura '{value}'; se ignora.")
    return None


# INTERFAZ

def tensor_to_thumbnail(image_tensor):
    # el tensor viene como (C, H, W) en [0,1] y hay que pasarlo a (H, W, C) en [0,255] para PIL
    array = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(array)
    img.thumbnail((C.PIX2PIX_THUMB_SIZE, C.PIX2PIX_THUMB_SIZE))
    return img


def choose_options(dataset, textures, default_source=None,
                   default_color=C.PIX2PIX_DEFAULT_COLOR,
                   default_texture=C.PIX2PIX_DEFAULT_TEXTURE,
                   default_extra=C.PIX2PIX_DEFAULT_EXTRA, default_texture_path=None,
                   worker=None):
    """
    Abre una ventana con dos pestañas seleccionables:
      1. imágenes del dataset para elegir una (clic) o un archivo propio.
      2. Color, material, muestra de textura (clic en una miniatura) y petición extra.

    Si se pasa 'worker', al pulsar "Recolorear" la ventana NO se cierra: se tapa con un
    mensaje de "Cargando" y worker(choice, progress) se ejecuta en un hilo aparte
    (choice = (source, color, texture, extra, texture_path), progress(texto) actualiza el mensaje).
    Cuando el worker termina, la ventana se cierra y se devuelve lo que devolvió el worker.
    Si el worker devuelve None o falla, se avisa y se vuelve al formulario.

    Sin 'worker' devuelve (source, color, texture, extra, texture_path) al confirmar.
    En los dos casos devuelve None si se cierra la ventana sin confirmar.
    """
    root = tk.Tk()
    root.title("Recolorear prenda")

    BLUE, GREY = "#175DB3", "#cccccc"
    # "busy" = hay un hilo trabajando, "done" = ya terminó, "output"/"error" = lo que salió de él
    state = {"source": None, "texture": None, "result": None,
             "busy": False, "done": False, "output": None, "error": None}
    status = {"msg": ""}                  # texto que escribe el hilo y que lee la interfaz
    photos, texture_photos = [], []       # referencias a las miniaturas (si no, Tkinter las borra)
    cells, texture_cells = {}, {}         # la clave es la label de la miniatura
    pool = []                             # muestras que encajan con el material escrito

    summary = tk.StringVar()
    texture_note = tk.StringVar()
    color_var = tk.StringVar(value=default_color)
    texture_var = tk.StringVar(value=default_texture)
    extra_var = tk.StringVar(value=default_extra)
    loading_title = tk.StringVar(value="Cargando")
    loading_detail = tk.StringVar()

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

    # ---------- lógica de la interfaz ----------
    def update_summary():
        image = state["source"] or "ninguna"
        texture = pathlib.Path(state["texture"]).stem if state["texture"] else "ninguna"
        summary.set(f"Imagen: {image}   |   Muestra de textura: {texture}")

    def select(source):
        state["source"] = source
        # resaltamos en azul la miniatura elegida y el resto en gris
        for key, label in cells.items():
            label.config(highlightbackground=BLUE if key == source else GREY)
        update_summary()

    def set_texture(path):
        state["texture"] = path
        for key, label in texture_cells.items():
            label.config(highlightbackground=BLUE if key == path else GREY)
        update_summary()

    def fill_gallery(indices):
        # primero borramos lo que hubiera de antes
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
            label.grid(row=k // 5, column=k % 5, padx=4, pady=4)  # 5 columnas
            # el s=str(idx) del lambda es para que cada botón se quede con SU índice
            label.bind("<Button-1>", lambda event, s=str(idx): select(s))
            cells[str(idx)] = label

    def new_random_set(force_index=None):
        indices = random.sample(range(len(dataset)), min(C.PIX2PIX_NUM_CHOICES, len(dataset)))
        # si nos pasan una imagen concreta, la metemos sí o sí en la primera posición
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
                continue  # si una imagen está corrupta nos la saltamos y ya
            img.thumbnail((C.PIX2PIX_TEXTURE_THUMB_SIZE, C.PIX2PIX_TEXTURE_THUMB_SIZE))
            photo = ImageTk.PhotoImage(img)
            texture_photos.append(photo)
            label = tk.Label(texture_gallery, image=photo, text=entry["id"], compound="top",
                             font=("TkDefaultFont", 8), highlightthickness=4,
                             highlightbackground=GREY, cursor="hand2")
            label.grid(row=k // 5, column=k % 5, padx=3, pady=3)
            # 1 click sobre una imagen la elige. 2 clicks sobre la misma la elimina
            label.bind("<Button-1>", lambda event, p=entry["path"]:
                       set_texture(None if state["texture"] == p else p))
            texture_cells[entry["path"]] = label

    def show_texture_page(forced_path=None):
        chosen = random.sample(pool, min(C.PIX2PIX_NUM_TEXTURE_CHOICES, len(pool)))
        forced = next((t for t in textures if t["path"] == forced_path), None) if forced_path else None
        # si venía una muestra por argumento la colamos en la lista aunque no saliera al azar
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
        pool[:] = matches  # [:] para modificar la lista en el sitio y no crear otra nueva
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

    # ---------- pantalla de "Cargando" ----------
    # es un frame que se coloca encima de toda la ventana (con place) para tapar el
    # formulario y que no se pueda tocar nada mientras el modelo trabaja
    loading = ttk.Frame(root)
    loading_inner = ttk.Frame(loading)
    loading_inner.place(relx=0.5, rely=0.5, anchor="center")  # centrado en la ventana
    ttk.Label(loading_inner, textvariable=loading_title,
              font=("TkDefaultFont", 18, "bold")).pack(pady=(0, 8))
    ttk.Label(loading_inner, textvariable=loading_detail, wraplength=380,
              justify="center").pack(pady=(0, 12))
    # barra "indeterminate" va de un lado a otro porque no sabemos cuánto falta
    loading_bar = ttk.Progressbar(loading_inner, mode="indeterminate", length=320)
    loading_bar.pack()
    ttk.Label(loading_inner, text="Puede tardar un rato, no cierres la ventana.",
              foreground="#666666").pack(pady=(12, 0))

    def show_loading():
        loading.place(relx=0, rely=0, relwidth=1, relheight=1)
        loading.lift()  # por si algún widget se quedara por encima
        loading_bar.start(12)

    def hide_loading():
        loading_bar.stop()
        loading.place_forget()

    def progress(text):
        # esto lo llama el hilo. No toca widgets (Tkinter no lo aguanta bien desde otro hilo),
        # solo guarda el texto y ya lo lee poll() desde el hilo principal
        status["msg"] = text
        print(text)

    def run_worker(choice):
        # esto corre en el hilo aparte
        # si falla, guardamos el error para enseñarlo luego
        try:
            state["output"] = worker(choice, progress)
        except Exception as exc:
            traceback.print_exc()
            state["error"] = exc
        finally:
            state["done"] = True

    def poll(tick=0):
        # se llama cada 300 ms desde el hilo principal para ver cómo va el otro hilo
        if not state["done"]:
            dots = "." * (tick % 4)  # animación de Cargando, Cargando., Cargando.., Cargando...
            loading_title.set("Cargando" + dots)
            loading_detail.set(status["msg"])
            root.after(300, poll, tick + 1)
            return

        # ya terminó el hilo
        state["busy"] = False
        if state["error"] is not None:
            hide_loading()
            messagebox.showerror("Error", f"Algo ha fallado generando la imagen:\n\n{state['error']}")
        elif state["output"] is None:
            hide_loading()
            messagebox.showinfo(
                "Sin prenda", "No se ha encontrado ninguna prenda clara en esta imagen.\n"
                              "Prueba con otra imagen.")
        else:
            loading_bar.stop()
            root.destroy()  # todo bien, así que cerramos todo y main() enseña el resultado

    def confirm():
        color = color_var.get().strip()
        texture = texture_var.get().strip()
        extra = extra_var.get().strip()
        if not state["source"]:
            messagebox.showwarning("Falta la imagen", "Elige una imagen.")
            return
        # comprobamos que haya algo que cambiar (color, textura, muestra o extra)
        nothing_to_change = (not color and (not texture or texture.lower() == C.PIX2PIX_NO_TEXTURE)
                             and not extra and not state["texture"])
        if nothing_to_change:
            messagebox.showwarning("Falta algo que cambiar",
                                   "Indica un color, una textura, una muestra o una petición extra.")
            return
        choice = (
            state["source"], color, texture or C.PIX2PIX_NO_TEXTURE, extra, state["texture"])
        state["result"] = choice

        # modo sin "Cargando". Mo sale la pantalla de carga mientras se genera la imagen
        if worker is None:
            root.destroy()
            return

        # modo con "Cargando". Reseteamos las flags, tapamos la ventana y lanzamos el hilo
        state.update(busy=True, done=False, output=None, error=None)
        status["msg"] = "Preparando..."
        loading_title.set("Cargando")
        loading_detail.set(status["msg"])
        show_loading()
        # daemon=True para que el hilo no impida cerrar el programa si algo sale mal
        threading.Thread(target=run_worker, args=(choice,), daemon=True).start()
        poll()

    def on_close():
        # si se cierra la ventana en pleno proceso se quedaría el hilo colgado, mejor avisar
        if state["busy"]:
            messagebox.showinfo("Espera", "Todavía se está generando la imagen, espera un momento.")
            return
        state["result"] = None
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    bottom = ttk.Frame(root)
    bottom.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
    ttk.Label(bottom, textvariable=summary).pack(side="left")
    ttk.Button(bottom, text="Recolorear", command=confirm).pack(side="right")

    # ARRANQUE DE LA INTERFAZ
    # si venía una imagen por argumento y es del dataset, la incluimos
    forced = int(default_source) if (default_source or "").isdigit() and int(default_source) < len(dataset) else None
    new_random_set(force_index=forced)
    if default_source and forced is None:
        select(default_source)  # ruta a un archivo pasada por argumento
    refresh_textures(forced_path=default_texture_path)

    root.mainloop()
    # con worker devolvemos lo que sacó el hilo. Si se cerró la ventana sin confirmar, None
    if worker is not None:
        return state["output"]
    return state["result"]


# PIPELINE

def load_image_tensor(source, dataset):
    """Si source es un número, usa esa imagen del dataset, si no, lo trata como ruta a un archivo."""
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
    # según cómo se guardara el checkpoint, los pesos pueden venir dentro de una clave o sueltos
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
        # cierre morfológico. rellena huecos más estrechos que close_size píxeles
        # (a la resolución de trabajo de Mask R-CNN) sin tocar los más anchos, como el escote
        # (primero dilata con MaxFilter y luego erosiona con MinFilter)
        mask_img = mask_img.filter(ImageFilter.MaxFilter(close_size)).filter(ImageFilter.MinFilter(close_size))
    dilate_size = C.PIX2PIX_MASK_DILATE_SIZE
    if dilate_size > 1:
        mask_img = mask_img.filter(ImageFilter.MaxFilter(dilate_size))
    mask_img = mask_img.resize((target_size, target_size), Image.NEAREST)
    if feather_radius > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(feather_radius))
    return mask_img


def preserve_garment_folds(original, edited, strength=None, radius=None):
    """Transfers source luminance detail (folds and seams) onto the edited garment."""
    strength = (
        C.PIX2PIX_FOLD_PRESERVATION_STRENGTH if strength is None else strength
    )
    radius = C.PIX2PIX_FOLD_PRESERVATION_RADIUS if radius is None else radius
    edited_rgb = np.asarray(edited.convert("RGB"), dtype=np.float32)
    original_luma_image = original.convert("L")
    original_luma = np.asarray(original_luma_image, dtype=np.float32)
    smooth_luma = np.asarray(
        original_luma_image.filter(ImageFilter.GaussianBlur(radius)),
        dtype=np.float32,
    )
    fold_detail = (original_luma - smooth_luma) * strength
    edited_luma = (
        edited_rgb[..., 0] * 0.299
        + edited_rgb[..., 1] * 0.587
        + edited_rgb[..., 2] * 0.114
    )
    target_luma = np.clip(edited_luma + fold_detail, 0, 255)
    scale = target_luma / np.maximum(edited_luma, 1)
    result = np.clip(edited_rgb * scale[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def detect_garments(source, dataset, device, progress=print):
    """Detecta y refina todas las prendas de una imagen para previsualizarlas o recolorearlas."""
    progress("Detectando prendas (Mask R-CNN)...")
    image_tensor = load_image_tensor(source, dataset)
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    mask_rcnn = load_mask_rcnn(C.PIX2PIX_CHECKPOINT).to(device)
    with torch.no_grad():
        model_output = mask_rcnn([image_tensor.to(device)])[0]

    label2name = {
        class_index: dataset.coco.loadCats(category_id)[0]["name"]
        for category_id, class_index in dataset.categoryid_to_index.items()
    }

    progress("Refinando las prendas con SAM 2...")
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_predictor = SAM2ImagePredictor.from_pretrained(C.SAM2_MODEL_ID, device=device.type)
    _, garments, _ = refine_with_sam2(model_output, image_np, sam2_predictor, label2name)
    return {"image": image_np, "garments": garments}


def build_edit_instruction(garment_name, color, texture, extra, has_texture_reference):
    """Construye una instrucción explícita para la prenda y la referencia visual."""
    parts = []
    if color:
        parts.append(C.PIX2PIX_INSTRUCTION_COLOR.format(name=garment_name, color=color))
    if texture:
        parts.append(C.PIX2PIX_INSTRUCTION_TEXTURE.format(name=garment_name, texture=texture))
    if has_texture_reference:
        parts.append(
            "Faithfully reproduce the distinctive surface texture, stitching, and "
            "repeating pattern visible in the supplied reference image across the "
            "selected garment. Treat the reference as a texture swatch and tile its "
            "pattern naturally over the garment. Do not copy the swatch background "
            "or its shape. Keep the garment color specified above."
        )
    if extra:
        parts.append(f"Additional instruction: {extra}")
    if parts:
        parts.append(
            "Preserve the selected garment's shape and all details not explicitly "
            "mentioned in these instructions."
        )
    return "\n".join(parts)


def generate(source, color, texture, extra, texture_path, dataset, device, progress=print,
             segmentation=None, garment_indices=None):
    """
    Ejecuta lo correspondiente a Mask R-CNN, SAM2 e InstructPix2Pix.
    No ejecuta nada de la interfaz (matplotlip, tkinter), así que se puede ejecutar 
    en un hilo aparte. Devuelve un diccionario con lo que hace falta para dibujar 
    el resultado, o None si no se encontró ninguna prenda.
    """
    color = color.strip()
    if color.lower() == C.PIX2PIX_NO_TEXTURE:
        color = ""
    segmentation = segmentation or detect_garments(source, dataset, device, progress)
    image_np = segmentation["image"]
    prendas = segmentation["garments"]
    if not prendas:
        print("SAM2 + Mask R-CNN no encontraron ninguna prenda clara en esta imagen.")
        print("Prueba con otra imagen, o baja score_prenda en refine_with_sam2.")
        return None

    indices = garment_indices if garment_indices is not None else [0]
    if not indices or any(index < 0 or index >= len(prendas) for index in indices):
        raise ValueError("Selecciona prendas detectadas válidas antes de generar.")
    selected_garments = [prendas[index] for index in indices]
    garment_names = list(dict.fromkeys(garment["name"] for garment in selected_garments))
    garment_name = " y ".join(name.split(",")[0].strip() for name in garment_names)
    binary_mask = np.logical_or.reduce([garment["mask"] for garment in selected_garments])
    print("Prendas elegidas: " + ", ".join(
        f"{garment['name']} (score={garment['score']:.2f})" for garment in selected_garments
    ))

    # imagen y máscara al tamaño de trabajo del modelo
    image_pil = Image.fromarray(image_np).resize(
        (C.PIX2PIX_SIZE, C.PIX2PIX_SIZE), Image.BICUBIC)
    mask_pil = mask_to_pil(
        binary_mask, C.PIX2PIX_SIZE, C.PIX2PIX_FEATHER_RADIUS,
        C.PIX2PIX_MASK_CLOSE_SIZE)

    # cargar InstructPix2Pix
    progress("Cargando InstructPix2Pix (puede tardar la primera vez)...")
    # float16 en GPU va mejor, mientras que en CPU hace falta float32 porque float16 va muy lento
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        C.PIX2PIX_MODEL_ID, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if not texture_path:
        pipe.enable_attention_slicing()

    # La muestra se aplica como referencia visual; no continuar en silencio si falla.
    texture_image = None
    if texture_path:
        texture_stage = "cargar IP-Adapter"
        try:
            progress("Cargando IP-Adapter (la primera vez descarga un codificador de imagen grande)...")
            pipe.load_ip_adapter(
                C.PIX2PIX_IP_ADAPTER_REPO, subfolder="models",
                weight_name=C.PIX2PIX_IP_ADAPTER_WEIGHTS)
            texture_stage = "configurar la escala de referencia"
            pipe.set_ip_adapter_scale(C.PIX2PIX_IP_ADAPTER_SCALE)
            texture_stage = "abrir la muestra seleccionada"
            with Image.open(texture_path) as reference:
                texture_image = reference.convert("RGB")
            print(f"Muestra de textura: {texture_path} (escala {C.PIX2PIX_IP_ADAPTER_SCALE})")
        except Exception as exc:
            raise RuntimeError(
                f"Falló el paso «{texture_stage}» "
                f"({type(exc).__name__}): {exc}. "
                "La generación se ha detenido para no ignorar la muestra seleccionada."
            ) from exc

    use_texture = bool(texture) and texture.lower() != C.PIX2PIX_NO_TEXTURE
    instruction = build_edit_instruction(
        garment_name,
        color,
        texture if use_texture else "",
        extra,
        has_texture_reference=texture_image is not None,
    )
    print(f"Instrucción: {instruction}")

    call_kwargs = dict(
        # si no hay instrucción ponemos una de relleno para que de error
        prompt=instruction or "keep the image as it is",
        image=image_pil,
        num_inference_steps=C.PIX2PIX_NUM_INFERENCE_STEPS,
        guidance_scale=C.PIX2PIX_GUIDANCE_SCALE,
        image_guidance_scale=C.PIX2PIX_IMAGE_GUIDANCE_SCALE,
        generator=torch.Generator(device=device).manual_seed(C.PIX2PIX_SEED),  # semilla fija
    )
    if texture_image is not None:
        call_kwargs["ip_adapter_image"] = texture_image

    # esta es la parte que más tarda
    progress("Generando la imagen con InstructPix2Pix...")
    edited = pipe(**call_kwargs).images[0].convert("RGB")
    edited = preserve_garment_folds(image_pil, edited)

    # el modelo ha editado la imagen entera, pero nos quedamos con su resultado
    # solo dentro de la máscara y con la imagen original fuera de ella
    final = Image.composite(edited, image_pil, mask_pil)

    # etiqueta para el título y el nombre del archivo
    sample_id = pathlib.Path(texture_path).stem if texture_image is not None else ""
    label = ", ".join(p for p in [color, texture if use_texture else "", extra, sample_id] if p)

    return {
        "source": source, "prenda_name": ", ".join(garment_names), "label": label,
        "image": image_pil, "mask": mask_pil, "edited": edited, "final": final,
    }


def show_result(result):
    """Dibuja las 4 imágenes, guarda la figura y la enseña. Esto va siempre en el hilo principal."""
    fig, axes = plt.subplots(1, 4, figsize=C.PIX2PIX_FIGURE_SIZE)
    axes[0].imshow(result["image"])
    axes[0].set_title(f"Original ({result['source']})")
    axes[1].imshow(result["mask"], cmap="gray")
    axes[1].set_title(f"Máscara refinada ({result['prenda_name']})")
    axes[2].imshow(result["edited"])
    axes[2].set_title("Salida del modelo (imagen entera)")
    axes[3].imshow(result["final"])
    axes[3].set_title(f"Resultado final: '{result['label']}'")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    out_dir = pathlib.Path(C.PIX2PIX_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    # limpiamos el label para que sirva de nombre de archivo (sin espacios ni símbolos raros)
    safe_label = re.sub(r"\W+", "_", result["label"]).strip("_")[:60]
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataset = FashionpediaDataset(
        C.PIX2PIX_IMAGES_DIR, C.PIX2PIX_ANNOTATIONS_FILE, max_side=C.PIX2PIX_IMAGE_SIZE)
    textures = load_texture_index(C.PIX2PIX_TEXTURES_DIR)
    print(f"Muestras de textura disponibles: {len(textures)}")

    # Datos elegidos por el usuario, bien por argumentos (sin interfaz) o bien con la ventana
    args = parse_args()
    texture_path_arg = resolve_texture_path(args.texture_sample, textures)
    if args.image and (args.color or args.texture or args.extra or args.texture_sample):
        # modo sin interfaz, todo viene de la línea de comandos
        result = generate(
            args.image, args.color or "", args.texture or C.PIX2PIX_NO_TEXTURE,
            args.extra or "", texture_path_arg, dataset, device)
    else:
        # modo con interfaz, le pasamos a la ventana una función que hace todo el trabajo
        # para que lo ejecute en un hilo mientras enseña el "Cargando"
        def worker(choice, progress):
            source, color, texture, extra, texture_path = choice
            return generate(source, color, texture, extra, texture_path,
                            dataset, device, progress)

        result = choose_options(
            dataset,
            textures,
            default_source=args.image,
            default_color=args.color or C.PIX2PIX_DEFAULT_COLOR,
            default_texture=args.texture or C.PIX2PIX_DEFAULT_TEXTURE,
            default_extra=args.extra or C.PIX2PIX_DEFAULT_EXTRA,
            default_texture_path=texture_path_arg,
            worker=worker,
        )

    if result is None:
        print("Cancelado o sin resultado.")
        return

    # mostramos resultado (la ventana de Tkinter ya se cerró en este punto)
    show_result(result)


if __name__ == "__main__":
    main()
