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
from PIL import Image, ImageChops, ImageFilter, ImageTk
import matplotlib.pyplot as plt
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)
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


_PIPELINE_CACHE = {}
_PIPELINE_LOCK = threading.Lock()


def get_pipeline(device):
    """
    Carga InstructPix2Pix una sola vez y lo deja cacheado para reutilizarlo en las
    siguientes generaciones. Antes se recargaba entero desde disco en cada llamada,
    lo que añadía una espera grande (y descarga el modelo la primera vez).
    """
    # float16 en GPU va mejor, mientras que en CPU hace falta float32 porque float16 va muy lento
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    key = (C.PIX2PIX_MODEL_ID, str(device), str(dtype))
    with _PIPELINE_LOCK:
        pipe = _PIPELINE_CACHE.get(key)
        if pipe is None:
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                C.PIX2PIX_MODEL_ID, torch_dtype=dtype, safety_checker=None
            )
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            _PIPELINE_CACHE[key] = pipe
        _activate_pipeline(key, pipe, device)
        if C.PIX2PIX_ATTENTION_SLICING:
            pipe.enable_attention_slicing()
        return pipe


def _activate_pipeline(key, pipe, device):
    for cached_key, cached_pipe in _PIPELINE_CACHE.items():
        if cached_key != key and getattr(cached_pipe, "device", torch.device("cpu")).type != "cpu":
            cached_pipe.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        pipe = pipe.to(device)
    elif getattr(pipe, "device", torch.device("cpu")) != device:
        pipe = pipe.to(device)
    _PIPELINE_CACHE[key] = pipe


def add_placed_detail(image, garment_mask, placement, device, progress):
    """Adds a textured pocket or a shaded button at the selected garment point."""

    mask_height, mask_width = garment_mask.shape
    ys, xs = np.nonzero(garment_mask)
    if not len(xs):
        raise ValueError("No se puede añadir un detalle a una máscara de prenda vacía.")
    garment_left, garment_right = int(xs.min()), int(xs.max())
    garment_top, garment_bottom = int(ys.min()), int(ys.max())
    garment_width = garment_right - garment_left + 1
    garment_height = garment_bottom - garment_top + 1
    image_width, image_height = image.size
    center_x = round(
        (garment_left + placement["x"] / 100 * max(garment_width - 1, 0))
        * image_width / mask_width
    )
    center_y = round(
        (garment_top + placement["y"] / 100 * max(garment_height - 1, 0))
        * image_height / mask_height
    )
    garment_width_px = garment_width * image_width / mask_width
    garment_height_px = garment_height * image_height / mask_height
    if placement["type"] == "pocket":
        half_width = max(11, round(garment_width_px * 0.06))
        half_height = max(11, round(garment_height_px * 0.075))
        bounds = (
            max(0, center_x - half_width),
            max(0, center_y - half_height),
            min(image_width - 1, center_x + half_width),
            min(image_height - 1, center_y + half_height),
        )
        progress("Añadiendo bolsillo con textura y pespuntes...")
        return draw_pocket_detail(image, bounds, garment_mask)
    radius = max(5, round(garment_width_px * 0.022))
    progress("Añadiendo botón con relieve y agujeros de costura...")
    return draw_button_detail(image, center_x, center_y, radius, garment_mask)


def draw_pocket_detail(image, bounds, garment_mask=None):
    """Composites a small sewn pocket from matching local fabric and a clipped shadow."""
    from PIL import ImageDraw

    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    if width < 8 or height < 8:
        raise ValueError("La zona seleccionada es demasiado pequeña para dibujar un bolsillo.")

    inset_x = max(1, round(width * 0.035))
    inset_y = max(1, round(height * 0.035))
    left += inset_x
    right -= inset_x
    top += inset_y
    bottom -= inset_y
    width, height = right - left, bottom - top
    corners = [
        (left, top),
        (right, top),
        (right, bottom - height // 6),
        (left + width // 2, bottom),
        (left, bottom - height // 6),
    ]
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    fabric = image_array[top:bottom + 1, left:right + 1].copy()
    local_y, local_x = np.mgrid[0:height + 1, 0:width + 1]
    edge = np.minimum.reduce((
        np.ones_like(local_x, dtype=np.float32),
        (local_x + 1) / max(width * 0.16, 1),
        (width - local_x + 1) / max(width * 0.16, 1),
        (local_y + 1) / max(height * 0.12, 1),
        (height - local_y + 1) / max(height * 0.12, 1),
    ))
    light = 0.88 + 0.28 * edge + 0.1 * (1 - local_y / max(height, 1))
    fabric = np.clip(fabric * light[..., None], 0, 255).astype(np.uint8)

    patch_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    patch = Image.fromarray(fabric).convert("RGBA")
    patch_alpha = Image.new("L", patch.size, 0)
    local_corners = [(x - left, y - top) for x, y in corners]
    ImageDraw.Draw(patch_alpha).polygon(local_corners, fill=255)
    patch.putalpha(patch_alpha)
    patch_layer.alpha_composite(patch, (left, top))

    shadow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    shadow_draw.polygon([(x, y + max(1, height // 24)) for x, y in corners], fill=(0, 0, 0, 75))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(max(1, width / 35)))
    if garment_mask is not None:
        garment_alpha = Image.fromarray((garment_mask.astype(np.uint8) * 255)).resize(
            image.size, Image.Resampling.NEAREST
        )
        shadow_alpha = ImageChops.multiply(shadow_layer.getchannel("A"), garment_alpha)
        shadow_layer.putalpha(shadow_alpha)
    result = Image.alpha_composite(image.convert("RGBA"), shadow_layer)
    result = Image.alpha_composite(result, patch_layer).convert("RGB")

    luminance = np.asarray(image.convert("RGB"), dtype=np.float32)
    sample_y = min(max((top + bottom) // 2, 0), image.height - 1)
    sample_x = min(max((left + right) // 2, 0), image.width - 1)
    brightness = float(luminance[sample_y, sample_x].mean())
    thread = (236, 230, 210) if brightness < 125 else (55, 52, 46)
    seam = (min(thread[0] + 15, 255), min(thread[1] + 15, 255), min(thread[2] + 15, 255))
    edge_thread = (28, 40, 58) if brightness < 125 else (75, 70, 62)
    draw = ImageDraw.Draw(result)
    # The top opening is reinforced; the other three sides use short, fabric-scale stitches.
    draw.line((left, top, right, top), fill=edge_thread, width=max(1, round(width / 35)))
    draw.line(
        (left + 2, top + max(1, height // 18), right - 2, top + max(1, height // 18)),
        fill=seam,
        width=max(1, round(width / 70)),
    )
    stitch = max(2, round(min(width, height) / 18))
    gap = max(2, stitch)
    for start, end in zip(corners[1:], corners[2:] + corners[:1]):
        draw.line((start, end), fill=edge_thread, width=max(1, round(width / 100)))
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(1, round((dx * dx + dy * dy) ** 0.5))
        for offset in range(stitch, length - stitch, stitch + gap):
            x1 = round(start[0] + dx * offset / length)
            y1 = round(start[1] + dy * offset / length)
            x2 = round(start[0] + dx * min(offset + stitch, length) / length)
            y2 = round(start[1] + dy * min(offset + stitch, length) / length)
            draw.line((x1, y1, x2, y2), fill=thread, width=max(1, round(width / 60)))

    if garment_mask is not None:
        clipped = ImageChops.multiply(
            Image.new("L", image.size, 255),
            Image.fromarray((garment_mask.astype(np.uint8) * 255)).resize(image.size, Image.Resampling.NEAREST),
        )
        result = Image.composite(result, image.convert("RGB"), clipped)
    return result


def draw_button_detail(image, center_x, center_y, radius, garment_mask=None):
    """Draws an anti-aliased, shaded four-hole button with a cast shadow."""
    from PIL import ImageDraw

    scale = 4
    diameter = radius * 2
    size = diameter + 10
    high = size * scale
    center = high // 2
    r = radius * scale
    layer = Image.new("RGBA", (high, high), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    sample_x = min(max(center_x, 0), image.width - 1)
    sample_y = min(max(center_y, 0), image.height - 1)
    garment_brightness = sum(image.convert("RGB").getpixel((sample_x, sample_y))) / 3
    base = (35, 36, 38) if garment_brightness > 125 else (202, 195, 177)
    draw.ellipse((center - r, center - r + 2 * scale, center + r, center + r + 2 * scale),
                 fill=(0, 0, 0, 75))
    draw.ellipse((center - r, center - r, center + r, center + r), fill=(*base, 245))
    inner_r = r * 0.82
    for step in range(10):
        fraction = step / 10
        ring_r = inner_r * (1 - fraction * 0.78)
        shade = round(0.55 + 0.3 * fraction)
        draw.ellipse(
            (center - ring_r, center - ring_r - fraction * r * 0.12,
             center + ring_r, center + ring_r - fraction * r * 0.12),
            fill=tuple(round(channel * shade) for channel in base) + (255,),
        )
    hole_r = max(1.2 * scale, r * 0.105)
    offset = r * 0.32
    for dx, dy in ((-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)):
        draw.ellipse((center + dx - hole_r, center + dy - hole_r,
                      center + dx + hole_r, center + dy + hole_r),
                     fill=(30, 30, 30, 255) if garment_brightness > 125 else (100, 97, 89, 255))
    thread_width = max(1, round(scale * 0.55))
    draw.line((center - offset, center + offset, center + offset, center - offset),
              fill=(205, 195, 170, 255), width=thread_width)
    draw.line((center - offset, center - offset, center + offset, center + offset),
              fill=(205, 195, 170, 255), width=thread_width)
    draw.ellipse((center - r * 0.48, center - r * 0.67,
                  center + r * 0.16, center - r * 0.35), fill=(255, 255, 255, 90))
    layer = layer.resize((size, size), Image.Resampling.LANCZOS)
    left, top = center_x - size // 2, center_y - size // 2
    canvas = Image.new("RGBA", image.size, (0, 0, 0, 0))
    crop_left = max(0, -left)
    crop_top = max(0, -top)
    crop_right = min(size, image.width - left)
    crop_bottom = min(size, image.height - top)
    if crop_left >= crop_right or crop_top >= crop_bottom:
        return image.convert("RGB")
    canvas.alpha_composite(
        layer.crop((crop_left, crop_top, crop_right, crop_bottom)),
        (max(0, left), max(0, top)),
    )
    if garment_mask is not None:
        garment_alpha = Image.fromarray(garment_mask.astype(np.uint8) * 255).resize(
            image.size, Image.Resampling.NEAREST
        )
        button_alpha = ImageChops.multiply(canvas.getchannel("A"), garment_alpha)
        canvas.putalpha(button_alpha)
    return Image.alpha_composite(image.convert("RGBA"), canvas).convert("RGB")


def prepare_ip_adapter(pipe, progress=print):
    """Carga el IP-Adapter sobre el pipeline cacheado. Solo la primera vez."""
    if getattr(pipe, "_atelier_ip_adapter_loaded", False):
        return
    texture_stage = "cargar IP-Adapter"
    try:
        progress("Cargando IP-Adapter (la primera vez descarga un codificador de imagen grande)...")
        pipe.load_ip_adapter(
            C.PIX2PIX_IP_ADAPTER_REPO, subfolder="models",
            weight_name=C.PIX2PIX_IP_ADAPTER_WEIGHTS)
        texture_stage = "configurar la escala de referencia"
        pipe.set_ip_adapter_scale(C.PIX2PIX_IP_ADAPTER_SCALE)
        pipe._atelier_ip_adapter_loaded = True
    except Exception as exc:
        raise RuntimeError(
            f"Falló el paso «{texture_stage}» "
            f"({type(exc).__name__}): {exc}. "
            "La generación se ha detenido para no ignorar la muestra seleccionada."
        ) from exc


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
             segmentation=None, garment_indices=None, garment_edits=None):
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
    if (not indices or len(indices) != len(set(indices))
            or any(index < 0 or index >= len(prendas) for index in indices)):
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

    edit_texture_paths = [
        edit.get("texture_path")
        for edit in (garment_edits or {}).values()
    ]
    needs_pix2pix = garment_edits is None or any(
        (
            bool(edit["color"].strip())
            and edit["color"].strip().lower() != C.PIX2PIX_NO_TEXTURE
        )
        or (
            bool(edit["texture"].strip())
            and edit["texture"].strip().lower() != C.PIX2PIX_NO_TEXTURE
        )
        or edit["extra"].strip()
        for edit in garment_edits.values()
    )
    pipe = None
    if needs_pix2pix:
        progress("Cargando InstructPix2Pix (puede tardar la primera vez)...")
        pipe = get_pipeline(device)
    needs_ip_adapter = bool(texture_path or any(edit_texture_paths))

    # Carga el adaptador (solo la primera vez); cada prenda puede recibir una referencia diferente.
    texture_image = None
    if needs_ip_adapter:
        if pipe is None:
            raise ValueError("La muestra de textura necesita también una instrucción de material.")
        prepare_ip_adapter(pipe, progress)
        if texture_path:
            try:
                with Image.open(texture_path) as reference:
                    texture_image = reference.convert("RGB")
                print(f"Muestra de textura: {texture_path} (escala {C.PIX2PIX_IP_ADAPTER_SCALE})")
            except Exception as exc:
                raise RuntimeError(
                    f"Falló el paso «abrir la muestra seleccionada» "
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
    if garment_edits is None:
        print(f"Instrucción: {instruction}")

    if garment_edits is not None:
        if set(garment_edits) != set(indices):
            raise ValueError("Cada prenda incluida debe tener su propia configuración.")
        final = image_pil.copy()
        edited_composite = image_pil.copy()
        selected_mask = np.zeros_like(binary_mask, dtype=bool)
        edit_labels = []
        for position, index in enumerate(indices, start=1):
            garment = prendas[index]
            edit = garment_edits[index]
            edit_color = edit["color"].strip()
            if edit_color.lower() == C.PIX2PIX_NO_TEXTURE:
                edit_color = ""
            edit_texture = edit["texture"].strip()
            use_garment_texture = bool(edit_texture) and edit_texture.lower() != C.PIX2PIX_NO_TEXTURE
            edit_texture_path = edit.get("texture_path")
            reference_image = None
            if edit_texture_path:
                try:
                    with Image.open(edit_texture_path) as reference:
                        reference_image = reference.convert("RGB")
                except Exception as exc:
                    raise RuntimeError(
                        f"No se pudo abrir la muestra seleccionada para "
                        f"{garment['name']}: {exc}"
                    ) from exc
            garment_prompt = build_edit_instruction(
                garment["name"].split(",")[0].strip(),
                edit_color,
                edit_texture if use_garment_texture else "",
                edit["extra"].strip(),
                has_texture_reference=reference_image is not None,
            )
            if garment_prompt:
                print(f"Instrucción para {garment['name']}: {garment_prompt}")
                call_kwargs = dict(
                    prompt=garment_prompt,
                    image=image_pil,
                    num_inference_steps=C.PIX2PIX_NUM_INFERENCE_STEPS,
                    guidance_scale=C.PIX2PIX_GUIDANCE_SCALE,
                    image_guidance_scale=C.PIX2PIX_IMAGE_GUIDANCE_SCALE,
                    generator=torch.Generator(device=device).manual_seed(C.PIX2PIX_SEED + index),
                )
                if reference_image is not None:
                    call_kwargs["ip_adapter_image"] = reference_image
                progress(f"Generando {garment['name']} ({position}/{len(indices)})...")
                garment_result = pipe(**call_kwargs).images[0].convert("RGB")
                garment_result = preserve_garment_folds(image_pil, garment_result)
            else:
                garment_result = image_pil.copy()
            garment_mask = mask_to_pil(
                garment["mask"], C.PIX2PIX_SIZE, C.PIX2PIX_FEATHER_RADIUS,
                C.PIX2PIX_MASK_CLOSE_SIZE,
            )
            final = Image.composite(garment_result, final, garment_mask)
            edited_composite = Image.composite(garment_result, edited_composite, garment_mask)
            selected_mask = np.logical_or(selected_mask, garment["mask"])
            sample_id = pathlib.Path(edit_texture_path).stem if reference_image is not None else ""
            edit_labels.append(", ".join(
                part for part in [edit_color, edit_texture if use_garment_texture else "",
                                  edit["extra"].strip(), sample_id] if part
            ))
        mask_pil = mask_to_pil(
            selected_mask, C.PIX2PIX_SIZE, C.PIX2PIX_FEATHER_RADIUS,
            C.PIX2PIX_MASK_CLOSE_SIZE,
        )
        edited = edited_composite
        label = " | ".join(
            f"{prendas[index]['name']}: {edit_labels[position]}"
            for position, index in enumerate(indices)
        )
    else:
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

        progress("Generando la imagen con InstructPix2Pix...")
        edited = pipe(**call_kwargs).images[0].convert("RGB")
        edited = preserve_garment_folds(image_pil, edited)
        final = Image.composite(edited, image_pil, mask_pil)
        sample_id = pathlib.Path(texture_path).stem if texture_image is not None else ""
        label = ", ".join(p for p in [color, texture if use_texture else "", extra, sample_id] if p)

    if garment_edits is not None:
        for index in indices:
            for placement in garment_edits[index].get("placements", []):
                final = add_placed_detail(
                    final,
                    prendas[index]["mask"],
                    placement,
                    device,
                    progress,
                )

    return {
        "source": source, "prenda_name": ", ".join(garment_names), "label": label,
        "image": image_pil, "mask": mask_pil, "edited": edited, "final": final,
        "garments": prendas,
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
