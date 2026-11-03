"""Interfaz web local para el recoloreado de prendas con InstructPix2Pix.

Uso desde la raíz del proyecto:
    python interfaz.py
"""

import json
import logging
import random
import re
import shutil
import tempfile
import threading
import uuid
import webbrowser
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

try:
    from winotify import Notification  # solo funciona en Windows
except ImportError:
    Notification = None  # sin winotify la interfaz va igual, solo que sin notificación

from src import config as C


logger = logging.getLogger("interfaz")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
STATIC_DIR = Path(__file__).resolve().parent / "static"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
job_lock = threading.Lock()
jobs = {}
worker = None


@asynccontextmanager
async def lifespan(app):
    global worker
    with tempfile.TemporaryDirectory(prefix="recolor-ui-") as temp_dir:
        app.state.temp_dir = Path(temp_dir)
        app.state.sources = {}
        app.state.segmentations = {}
        app.state.segmenting = False
        app.state.dataset = None
        yield
        if worker is not None:
            await run_in_threadpool(worker.shutdown, wait=True)
            worker = None


app = FastAPI(title="Atelier — recolore prendas", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class HsvValue(BaseModel):
    hue: int = Field(ge=0, le=359)
    saturation: int = Field(ge=0, le=255)
    value: int = Field(ge=0, le=255)


class GarmentEditRequest(BaseModel):
    index: int = Field(ge=0)
    color: str = Field(default="", max_length=80)
    texture: str = Field(default=C.PIX2PIX_NO_TEXTURE, max_length=80)
    extra: str = Field(default="", max_length=2000)
    hsv: HsvValue | None = None
    placements: list[dict] = Field(default_factory=list)


class GenerationRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=100)
    color: str = Field(default="", max_length=80)
    texture: str = Field(default=C.PIX2PIX_NO_TEXTURE, max_length=80)
    extra: str = Field(default="", max_length=2000)
    segmentation_id: str | None = Field(default=None, max_length=100)
    garment_indices: list[int] | None = None
    garment_edits: list[GarmentEditRequest] | None = None


class SegmentationRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=100)


class HsvGarmentEdit(BaseModel):
    index: int = Field(ge=0)
    hue: int = Field(ge=0, le=359)
    saturation: int = Field(ge=0, le=255)
    value: int = Field(ge=0, le=255)


class HsvRecolorRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=100)
    segmentation_id: str = Field(min_length=1, max_length=100)
    garment_indices: list[int] | None = Field(default=None, max_length=50)
    hue: int | None = Field(default=None, ge=0, le=359)
    saturation: int | None = Field(default=None, ge=0, le=255)
    value: int | None = Field(default=None, ge=0, le=255)
    garments: list[HsvGarmentEdit] | None = None


class RenameHistoryRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


def get_dataset(app_state):
    if app_state.dataset is None:
        from src.dataset import FashionpediaDataset

        app_state.dataset = FashionpediaDataset(
            C.PIX2PIX_IMAGES_DIR,
            C.PIX2PIX_ANNOTATIONS_FILE,
            max_side=C.PIX2PIX_IMAGE_SIZE,
        )
    return app_state.dataset


def source_path(app_state, source_id):
    source = app_state.sources.get(source_id)
    if source is None:
        try:
            index = int(source_id)
            dataset = get_dataset(app_state)
        except (TypeError, ValueError):
            raise HTTPException(status_code=404, detail="La imagen seleccionada ya no está disponible.")
        if index < 0 or index >= len(dataset):
            raise HTTPException(status_code=404, detail="La imagen seleccionada no existe en el catálogo.")
        image_info = dataset.coco.imgs[dataset.image_ids[index]]
        source = Path(dataset.images_dir) / image_info["file_name"]
    return Path(source) if not isinstance(source, int) else str(source)


@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
async def config():
    return {
        "colors": C.PIX2PIX_COLOR_OPTIONS,
        "textures": C.PIX2PIX_TEXTURE_OPTIONS,
    }


HISTORY_JOB_SUFFIX = re.compile(r"^[0-9a-f]{8}$")


def history_directory():
    return Path(C.PIX2PIX_OUTPUT_DIR)


def history_assets_directory():
    return history_directory() / ".atelier-history"


def history_label(stem):
    """Nombre editable del diseño, sin el prefijo ni el identificador del trabajo."""
    label = stem.removeprefix(C.PIX2PIX_OUTPUT_PREFIX)
    head, _, suffix = label.rpartition("_")
    if head and HISTORY_JOB_SUFFIX.fullmatch(suffix):
        label = head
    return label.replace("_", " ") or "diseño"


def history_job_suffix(stem):
    """Sufijo del trabajo (p. ej. ``_1a2b3c4d``) que se conserva al renombrar."""
    label = stem.removeprefix(C.PIX2PIX_OUTPUT_PREFIX)
    head, _, suffix = label.rpartition("_")
    return f"_{suffix}" if head and HISTORY_JOB_SUFFIX.fullmatch(suffix) else ""


def history_label_key(label):
    """Clave para comparar nombres sin depender de mayúsculas ni separadores."""
    return label.replace("_", " ").strip().casefold()


def unique_history_label(base_label):
    """Añade un número al nombre base si ya existe otro diseño con ese nombre."""
    used = set()
    output_dir = history_directory()
    if output_dir.is_dir():
        for path in output_dir.glob(f"{C.PIX2PIX_OUTPUT_PREFIX}*.png"):
            if path.is_file():
                used.add(history_label_key(history_label(path.stem)))
    candidate = base_label
    index = 2
    while history_label_key(candidate) in used:
        candidate = f"{base_label}_{index}"
        index += 1
    return candidate


def save_history_assets(final_path, result):
    import numpy as np
    from PIL import Image

    assets_dir = history_assets_directory() / final_path.stem
    assets_dir.mkdir(parents=True, exist_ok=False)
    result["mask"].save(assets_dir / "mask.png")

    palette = ((199, 93, 58), (70, 108, 165), (102, 134, 106),
               (136, 108, 159), (226, 145, 80), (41, 54, 80))
    garments = result["garments"]
    width, height = result["image"].size
    segmentation = np.zeros((height, width, 4), dtype=np.uint8)
    garment_names = []
    for index, garment in enumerate(garments):
        red, green, blue = palette[index % len(palette)]
        mask = np.asarray(garment["mask"], dtype=bool)
        if mask.shape != (height, width):
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (width, height), Image.Resampling.NEAREST
                ),
                dtype=np.uint8,
            ) > 0
        garment_overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        garment_overlay[mask] = (red, green, blue, 175)
        Image.fromarray(garment_overlay).save(assets_dir / f"garment-{index}.png")
        segmentation[mask] = (red, green, blue, 175)
        garment_names.append(garment["name"])

    Image.fromarray(segmentation).save(assets_dir / "segmentation.png")
    manifest = {"garments": garment_names}
    (assets_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )


def safe_download_filename(value, default="atelier.png"):
    name = Path(str(value or default).replace("\\", "/")).name
    name = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._")
    if not name:
        name = default
    if not name.lower().endswith(".png"):
        name = f"{name[:96]}.png"
    return name[:100]


@app.get("/api/history")
async def history():
    output_dir = history_directory()
    if not output_dir.is_dir():
        return {"items": []}
    files = sorted(
        (path for path in output_dir.glob(f"{C.PIX2PIX_OUTPUT_PREFIX}*.png") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    items = []
    for path in files[:100]:
        item = {
            "filename": path.name,
            "label": history_label(path.stem),
            "modified": path.stat().st_mtime,
            "preview_url": f"/api/history/{path.name}",
            "assets": None,
        }
        manifest_path = history_assets_directory() / path.stem / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                asset_url = f"/api/history/{path.name}/assets"
                item["assets"] = {
                    "mask_url": f"{asset_url}/mask",
                    "segmentation_url": f"{asset_url}/segmentation",
                    "garment_masks": [
                        {"name": name, "url": f"{asset_url}/garment-{index}"}
                        for index, name in enumerate(manifest["garments"])
                    ],
                }
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                logger.exception("No se pudieron leer las máscaras guardadas para %s", path)
                raise HTTPException(
                    status_code=500, detail="No se pudieron leer las máscaras de una generación."
                ) from exc
        items.append(item)
    return {"items": items}


@app.get("/api/history/{filename}")
async def history_image(filename: str, download: bool = False, name: str | None = None):
    output_dir = history_directory().resolve()
    path = (output_dir / filename).resolve()
    if (path.parent != output_dir or not filename.startswith(C.PIX2PIX_OUTPUT_PREFIX)
            or path.suffix.lower() != ".png" or not path.is_file()):
        raise HTTPException(status_code=404, detail="No se encontró ese diseño guardado.")
    download_name = safe_download_filename(name, path.name) if download else None
    return FileResponse(path, media_type="image/png", filename=download_name)


@app.post("/api/history/{filename}/rename")
async def rename_history_image(filename: str, request: RenameHistoryRequest):
    output_dir = history_directory().resolve()
    path = (output_dir / filename).resolve()
    if (path.parent != output_dir or not filename.startswith(C.PIX2PIX_OUTPUT_PREFIX)
            or path.suffix.lower() != ".png" or not path.is_file()):
        raise HTTPException(status_code=404, detail="No se encontró ese diseño guardado.")

    requested_name = request.name.strip().removesuffix(".png")
    safe_label = re.sub(r"\W+", "_", requested_name).strip("_")[:45]
    if not safe_label:
        raise HTTPException(status_code=422, detail="Indica un nombre válido para el diseño.")

    requested_key = history_label_key(safe_label)
    if requested_key != history_label_key(history_label(path.stem)):
        for other in output_dir.glob(f"{C.PIX2PIX_OUTPUT_PREFIX}*.png"):
            if (other.is_file() and other.resolve() != path
                    and history_label_key(history_label(other.stem)) == requested_key):
                raise HTTPException(status_code=409, detail="Ya existe un diseño con ese nombre.")

    suffix = history_job_suffix(path.stem)
    new_path = output_dir / f"{C.PIX2PIX_OUTPUT_PREFIX}{safe_label}{suffix}.png"
    if new_path != path and new_path.exists():
        raise HTTPException(status_code=409, detail="Ya existe un diseño con ese nombre.")
    if new_path != path:
        assets_dir = history_assets_directory() / path.stem
        new_assets_dir = history_assets_directory() / new_path.stem
        image_renamed = False
        try:
            path.rename(new_path)
            image_renamed = True
            if assets_dir.is_dir():
                assets_dir.rename(new_assets_dir)
        except OSError as exc:
            if image_renamed:
                try:
                    new_path.rename(path)
                except OSError:
                    logger.exception("No se pudo restaurar el nombre anterior de %s", path)
            logger.exception("No se pudo renombrar el diseño guardado %s", path)
            raise HTTPException(
                status_code=500, detail="No se pudo guardar el nuevo nombre del diseño."
            ) from exc

        with job_lock:
            for job in jobs.values():
                if job.get("saved_path") and Path(job["saved_path"]).resolve() == path:
                    job["saved_path"] = str(new_path)
                    job["history_filename"] = new_path.name

    return {
        "filename": new_path.name,
        "label": safe_label.replace("_", " "),
    }


@app.get("/api/history/{filename}/assets/{asset_name}")
async def history_asset(filename: str, asset_name: str):
    output_dir = history_directory().resolve()
    history_path = (output_dir / filename).resolve()
    if (history_path.parent != output_dir or not filename.startswith(C.PIX2PIX_OUTPUT_PREFIX)
            or history_path.suffix.lower() != ".png" or not history_path.is_file()):
        raise HTTPException(status_code=404, detail="No se encontró ese diseño guardado.")

    if asset_name in {"mask", "segmentation"}:
        asset_filename = f"{asset_name}.png"
    else:
        match = re.fullmatch(r"garment-(\d+)", asset_name)
        if match is None:
            raise HTTPException(status_code=404, detail="No se encontró esa máscara.")
        asset_filename = f"garment-{int(match.group(1))}.png"
    assets_root = history_assets_directory().resolve()
    assets_dir = (assets_root / history_path.stem).resolve()
    path = (assets_dir / asset_filename).resolve()
    if assets_dir.parent != assets_root or path.parent != assets_dir or not path.is_file():
        raise HTTPException(status_code=404, detail="No se encontró esa máscara guardada.")
    return FileResponse(path, media_type="image/png")


@app.delete("/api/history/{filename}")
async def delete_history_image(filename: str):
    output_dir = history_directory().resolve()
    path = (output_dir / filename).resolve()
    if (path.parent != output_dir or not filename.startswith(C.PIX2PIX_OUTPUT_PREFIX)
            or path.suffix.lower() != ".png" or not path.is_file()):
        raise HTTPException(status_code=404, detail="No se encontró ese diseño guardado.")
    try:
        path.unlink()
        assets_dir = history_assets_directory() / path.stem
        if assets_dir.is_dir():
            shutil.rmtree(assets_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Ese diseño ya no existe.") from exc
    except OSError as exc:
        logger.exception("No se pudo eliminar el diseño guardado %s", path)
        raise HTTPException(status_code=500, detail="No se pudo eliminar el diseño guardado.") from exc
    return {"deleted": filename}


@app.delete("/api/history")
async def clear_history():
    output_dir = history_directory().resolve()
    deleted = 0
    try:
        if output_dir.is_dir():
            for path in output_dir.glob(f"{C.PIX2PIX_OUTPUT_PREFIX}*.png"):
                if not path.is_file():
                    continue
                path.unlink()
                deleted += 1
        assets_root = history_assets_directory().resolve()
        if assets_root.is_dir() and assets_root.parent == output_dir:
            shutil.rmtree(assets_root)
    except OSError as exc:
        logger.exception("No se pudo borrar el historial")
        raise HTTPException(status_code=500, detail="No se pudo borrar el historial.") from exc
    return {"deleted": deleted}


@app.get("/api/images")
async def images(request: Request):
    dataset = await run_in_threadpool(get_dataset, request.app.state)
    count = min(C.PIX2PIX_NUM_CHOICES, len(dataset))
    indices = random.sample(range(len(dataset)), count)
    items = []
    for index in indices:
        info = dataset.coco.imgs[dataset.image_ids[index]]
        items.append({
            "id": str(index),
            "label": info["file_name"],
            "preview_url": f"/api/images/{index}/preview",
        })
    return {"items": items}


@app.get("/api/images/{source_id}/preview")
async def image_preview(source_id: str, request: Request):
    path = source_path(request.app.state, source_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No se encontró la imagen seleccionada.")
    return FileResponse(path)


@app.post("/api/segment")
async def segment_image(request: Request, options: SegmentationRequest):
    app_state = request.app.state
    source = source_path(app_state, options.source_id)
    with job_lock:
        if app_state.segmenting or any(job["status"] in {"queued", "running"} for job in jobs.values()):
            raise HTTPException(status_code=409, detail="Espera a que termine la tarea actual antes de detectar prendas.")
        app_state.segmenting = True
    try:
        import torch
        from src.recolor import detect_garments

        dataset = await run_in_threadpool(get_dataset, app_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        segmentation = await run_in_threadpool(
            detect_garments, str(source), dataset, device, logger.info
        )
    finally:
        with job_lock:
            app_state.segmenting = False

    garments = segmentation["garments"]
    if not garments:
        raise HTTPException(status_code=422, detail="No se detectaron prendas en esta imagen. Prueba con otra foto.")

    segmentation_id = uuid.uuid4().hex
    app_state.segmentations[segmentation_id] = {
        "source_id": options.source_id,
        **segmentation,
    }
    while len(app_state.segmentations) > 8:
        app_state.segmentations.pop(next(iter(app_state.segmentations)))
    height, width = segmentation["image"].shape[:2]
    return {
        "id": segmentation_id,
        "width": width,
        "height": height,
        "items": [
            {
                "index": index,
                "name": garment["name"],
                "score": garment["score"],
                "mask_url": f"/api/segmentations/{segmentation_id}/mask/{index}",
            }
            for index, garment in enumerate(garments)
        ],
    }


@app.get("/api/segmentations/{segmentation_id}/mask/{index}")
async def segmentation_mask(segmentation_id: str, index: int, request: Request):
    segmentation = request.app.state.segmentations.get(segmentation_id)
    if segmentation is None or index < 0 or index >= len(segmentation["garments"]):
        raise HTTPException(status_code=404, detail="La máscara de la prenda ya no está disponible.")

    import numpy as np
    from PIL import Image

    palette = ((199, 93, 58), (70, 108, 165), (102, 134, 106),
               (136, 108, 159), (226, 145, 80), (41, 54, 80))
    red, green, blue = palette[index % len(palette)]
    mask = segmentation["garments"][index]["mask"]
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask] = (red, green, blue, 175)
    image = BytesIO()
    Image.fromarray(overlay).save(image, format="PNG")
    return Response(content=image.getvalue(), media_type="image/png")


@app.post("/api/upload")
async def upload_image(request: Request, filename: str = "imagen"):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="La imagen supera el límite de 20 MB.")
    content_buffer = bytearray()
    async for chunk in request.stream():
        if len(content_buffer) + len(chunk) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="La imagen supera el límite de 20 MB.")
        content_buffer.extend(chunk)
    content = bytes(content_buffer)
    if not content:
        raise HTTPException(status_code=413, detail="La imagen está vacía o supera el límite de 20 MB.")
    try:
        from PIL import Image

        with Image.open(BytesIO(content)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                raise HTTPException(status_code=415, detail="Formato no compatible. Usa JPG, PNG, WEBP o BMP.")
            image.verify()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.") from exc

    source_id = uuid.uuid4().hex
    destination = request.app.state.temp_dir / f"{source_id}.png"
    with Image.open(BytesIO(content)) as image:
        image.convert("RGB").save(destination, format="PNG")
    request.app.state.sources[source_id] = destination
    safe_name = Path(filename).name or "imagen"
    return {
        "id": source_id,
        "label": safe_name,
        "preview_url": f"/api/images/{source_id}/preview",
    }


def notify_done(saved_path):
    """Notificación de Windows al terminar. Si falla no debe romper la generación, por eso el try."""
    if Notification is None:
        logger.warning("winotify no está instalado: no se envía notificación (pip install winotify)")
        return
    try:
        Notification(
            app_id="Recolor de prendas",
            title="Prenda creada",
            msg=f"El resultado se guardó en:\n{Path(saved_path).resolve()}",
            duration="long",
        ).show()
    except Exception:
        logger.exception("No se pudo enviar la notificación")


def original_size(source, dataset):
    """Tamaño (ancho, alto) de la imagen original, para devolver el resultado sin deformar."""
    from PIL import Image

    source = str(source)
    if source.isdigit():  # índice del dataset en vez de ruta
        tensor, _ = dataset[int(source)]  # tensor (C, H, W)
        return tensor.shape[2], tensor.shape[1]
    with Image.open(source) as img:
        return img.size


def apply_hsv_garments(image, segmentation, garment_settings):
    """
    Aplica recolor_hsv a las prendas indicadas sobre `image` (array RGB uint8). Devuelve
    (imagen recoloreada, unión de las máscaras suaves en 0-1, máscaras booleanas de cada prenda).
    """
    import cv2
    import numpy as np
    from utils.recolor_hsv import recolor_hsv

    height, width = image.shape[:2]
    result = image
    combined = np.zeros((height, width), np.float32)  # unión de las máscaras suaves (vista "Máscara")
    garment_masks = []
    for settings in garment_settings:
        garment = segmentation["garments"][settings["index"]]
        mask = np.asarray(garment["mask"]).astype(np.uint8)
        soft = garment.get("alpha")
        if soft is not None:
            alpha = np.asarray(soft, dtype=np.float32)
        else:
            alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)  # borde suave, sin escalón
        if mask.shape != (height, width):
            # la detección se hizo a otra resolución: escalamos las máscaras a la de la imagen original
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
        result = recolor_hsv(result, mask.astype(bool), settings["hue"], settings["saturation"], settings["value"], alpha=alpha)
        combined = np.maximum(combined, np.clip(alpha, 0, 1))
        garment_masks.append(mask.astype(bool))
    return result, combined, garment_masks


def hsv_names(segmentation, garment_indices):
    return ", ".join(segmentation["garments"][i]["name"].split(",")[0].strip() for i in garment_indices)


def hsv_label(segmentation, garment_settings):
    if len(garment_settings) == 1:
        s0 = garment_settings[0]
        return f"HSV tono {s0['hue']} sat {s0['saturation']} brillo {s0['value']}"
    label_parts = [f"{segmentation['garments'][s['index']]['name'].split(',')[0].strip()} H{s['hue']}°" for s in garment_settings]
    return " | ".join(label_parts)


def hsv_only_result(source, segmentation, hsv_settings):
    """Resultado con el mismo formato que generate_recolor cuando todas las prendas van solo con recolor_hsv."""
    import numpy as np
    from PIL import Image

    with Image.open(source) as img:
        original = img.convert("RGB")
    array, combined, masks = apply_hsv_garments(np.array(original), segmentation, hsv_settings)
    final = Image.fromarray(array)
    indices = [s["index"] for s in hsv_settings]
    return {
        "image": original,
        "mask": Image.fromarray((np.clip(combined, 0, 1) * 255).astype(np.uint8)),
        "edited": final,
        "final": final.copy(),
        "label": hsv_label(segmentation, hsv_settings),
        "prenda_name": hsv_names(segmentation, indices),
        "garments": [
            {"name": segmentation["garments"][i]["name"], "mask": mask}
            for i, mask in zip(indices, masks)
        ],
    }


def add_hsv_to_result(result, segmentation, hsv_settings):
    """Aplica recolor_hsv por encima del resultado de InstructPix2Pix (ya a tamaño original)."""
    import numpy as np
    from PIL import Image

    combined = None
    masks = []
    for key in ("edited", "final"):
        array, combined, masks = apply_hsv_garments(np.array(result[key].convert("RGB")), segmentation, hsv_settings)
        result[key] = Image.fromarray(array)
    base_mask = np.asarray(result["mask"].convert("L"), dtype=np.float32) / 255
    result["mask"] = Image.fromarray((np.maximum(base_mask, np.clip(combined, 0, 1)) * 255).astype(np.uint8))
    indices = [s["index"] for s in hsv_settings]
    result["garments"] = list(result["garments"]) + [
        {"name": segmentation["garments"][i]["name"], "mask": mask} for i, mask in zip(indices, masks)
    ]
    result["label"] = f"{result['label']} | {hsv_label(segmentation, hsv_settings)}"
    result["prenda_name"] = f"{result['prenda_name']}, {hsv_names(segmentation, indices)}"
    return result


def run_hsv_recolor(app_state, job_id, source, segmentation, garment_settings):
    """
    Aplica recolor_hsv a las prendas elegidas a la resolución de la imagen original, la guarda en el
    historial (con sus archivos auxiliares, como el resto de resultados) y registra un trabajo ya
    completado para que el front la muestre igual que cualquier otro. No usa el modelo ni la GPU,
    así que no pasa por la cola.
    """
    import cv2
    import numpy as np
    from PIL import Image
    from utils.recolor_hsv import recolor_hsv

    with Image.open(source) as img:
        original = np.array(img.convert("RGB"))
    height, width = original.shape[:2]

    garment_indices = [s["index"] for s in garment_settings]
    result, combined, garment_masks = apply_hsv_garments(original, segmentation, garment_settings)
    names = hsv_names(segmentation, garment_indices)
    label = hsv_label(segmentation, garment_settings)

    output_dir = Path(C.PIX2PIX_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = unique_history_label(re.sub(r"\W+", "_", label).strip("_")[:55] or "diseno")
    final_path = output_dir / f"{C.PIX2PIX_OUTPUT_PREFIX}{safe_label}_{job_id[:8]}.png"
    final_image = Image.fromarray(result)
    mask_image = Image.fromarray((combined * 255).astype(np.uint8))
    original_image = Image.fromarray(original)
    save_history_assets(final_path, {
        "mask": mask_image,
        "image": original_image,
        "garments": [
            {"name": segmentation["garments"][i]["name"], "mask": garment_mask}
            for i, garment_mask in zip(garment_indices, garment_masks)
        ],
    })
    final_image.save(final_path)

    job_dir = app_state.temp_dir / job_id
    job_dir.mkdir()
    paths = {
        "original": job_dir / "original.png",
        "mask": job_dir / "mask.png",
        "edited": job_dir / "edited.png",
        "final": job_dir / "final.png",
    }
    original_image.save(paths["original"])
    mask_image.save(paths["mask"])
    final_image.save(paths["edited"])
    final_image.save(paths["final"])
    with job_lock:
        jobs[job_id] = {
            "status": "completed",
            "message": "Diseño listo",
            "error": None,
            "garment": names,
            "label": history_label(final_path.stem),
            "images": paths,
            "saved_path": str(final_path),
            "history_filename": final_path.name,
        }


def run_generation(app_state, job_id, source, color, texture, extra, texture_path,
                  segmentation, garment_indices, garment_edits=None, hsv_settings=None):
    with job_lock:
        if jobs[job_id]["status"] == "cancelled" or jobs[job_id].get("cancel_requested"):
            jobs[job_id].update({"status": "cancelled", "message": "Generación cancelada"})
            return
        jobs[job_id]["status"] = "running"

    def progress(message):
        with job_lock:
            jobs[job_id]["message"] = message

    try:
        # Negro y blanco (y texturas, instrucciones o elementos) → InstructPix2Pix.
        # Resto de colores de la paleta y colores personalizados HSV/HEX → recolor_hsv.
        # Si hay prendas de ambos tipos, primero se ejecuta el modelo y después se aplica el color HSV.
        hsv_settings = hsv_settings or []
        hsv_only = bool(hsv_settings) and not garment_indices
        if hsv_only:
            progress("Recoloreando con HSV…")
            result = hsv_only_result(source, segmentation, hsv_settings)
        else:
            import torch
            from src.recolor import generate as generate_recolor

            dataset = get_dataset(app_state)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            result = generate_recolor(
                source, color, texture, extra, texture_path, dataset, device, progress,
                segmentation=segmentation, garment_indices=garment_indices,
                garment_edits=garment_edits,
            )
        with job_lock:
            if jobs[job_id].get("cancel_requested"):
                jobs[job_id].update({"status": "cancelled", "message": "Generación cancelada"})
                return
        if result is None:
            raise RuntimeError("No se detectó ninguna prenda clara. Prueba con otra imagen.")

        if not hsv_only:
            # el modelo trabaja en cuadrado (PIX2PIX_SIZE x PIX2PIX_SIZE) y eso deforma la imagen,
            # así que devolvemos al tamaño de la original para recuperar su proporción
            from PIL import Image
            import numpy as np

            size = original_size(source, dataset)
            for key in ("image", "mask", "edited", "final"):
                if result[key].size != size:
                    result[key] = result[key].resize(size, Image.LANCZOS)

            if segmentation and garment_indices:
                combined = np.zeros(segmentation["image"].shape[:2], dtype=bool)
                for index in garment_indices:
                    combined |= segmentation["garments"][index]["mask"]
                if combined.any():
                    original_pil = Image.fromarray(segmentation["image"]).convert("RGB")
                    clamp_mask = Image.fromarray(combined.astype(np.uint8) * 255)
                    if clamp_mask.size != result["final"].size:
                        clamp_mask = clamp_mask.resize(result["final"].size, Image.NEAREST)
                    if original_pil.size != result["final"].size:
                        original_pil = original_pil.resize(result["final"].size, Image.LANCZOS)
                    result["final"] = Image.composite(result["final"], original_pil, clamp_mask)

            if hsv_settings:   # después del recorte, para que no borre las prendas recoloreadas con HSV
                progress("Aplicando los colores HSV…")
                result = add_hsv_to_result(result, segmentation, hsv_settings)

        output_dir = Path(C.PIX2PIX_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"\W+", "_", result["label"]).strip("_")[:45] or "diseno"
        safe_label = unique_history_label(safe_label)
        final_path = output_dir / f"{C.PIX2PIX_OUTPUT_PREFIX}{safe_label}_{job_id[:8]}.png"
        save_history_assets(final_path, result)
        result["final"].save(final_path)

        job_dir = app_state.temp_dir / job_id
        job_dir.mkdir()
        paths = {
            "original": job_dir / "original.png",
            "mask": job_dir / "mask.png",
            "edited": job_dir / "edited.png",
            "final": job_dir / "final.png",
        }
        result["image"].save(paths["original"])
        result["mask"].save(paths["mask"])
        result["edited"].save(paths["edited"])
        result["final"].save(paths["final"])
        with job_lock:
            jobs[job_id].update({
                "status": "completed",
                "message": "Diseño listo",
                "garment": result["prenda_name"],
                "label": history_label(final_path.stem),
                "images": paths,
                "saved_path": str(final_path),
                "history_filename": final_path.name,
            })
        notify_done(final_path)  # fuera del lock y con su propio try, no puede marcar el trabajo como fallido
    except Exception as exc:
        with job_lock:
            if jobs[job_id].get("cancel_requested"):
                jobs[job_id].update({"status": "cancelled", "message": "Generación cancelada"})
                return
        logger.exception("Falló la generación del trabajo %s", job_id)
        with job_lock:
            jobs[job_id].update({
                "status": "failed",
                "message": "No se pudo completar la generación",
                "error": str(exc),
            })


@app.post("/api/generate", status_code=202)
async def generate(request: Request, options: GenerationRequest):
    global worker
    try:
        source = source_path(request.app.state, options.source_id)
    except HTTPException:
        raise
    color = options.color.strip()
    if color.lower() == C.PIX2PIX_NO_TEXTURE:
        color = ""
    texture_name = options.texture.strip() or C.PIX2PIX_NO_TEXTURE
    extra = options.extra.strip()
    if (options.garment_edits is None
            and not color and texture_name.lower() == C.PIX2PIX_NO_TEXTURE
            and not extra):
        raise HTTPException(
            status_code=400,
            detail="Indica un color, material o instrucción antes de generar.",
        )
    texture_path = None  # ya no se usan muestras de AmbientCG; generate_recolor sigue esperando el argumento

    segmentation = None
    garment_indices = None
    garment_edits = None
    hsv_settings = []
    if options.segmentation_id:
        segmentation = request.app.state.segmentations.get(options.segmentation_id)
        if segmentation is None or segmentation["source_id"] != options.source_id:
            raise HTTPException(status_code=400, detail="Vuelve a detectar las prendas de la imagen seleccionada.")
        garment_indices = options.garment_indices
        if not garment_indices:
            raise HTTPException(status_code=400, detail="Selecciona al menos una prenda detectada.")
        if any(index < 0 or index >= len(segmentation["garments"]) for index in garment_indices):
            raise HTTPException(status_code=400, detail="Una de las prendas seleccionadas ya no está disponible.")
        if len(garment_indices) != len(set(garment_indices)):
            raise HTTPException(status_code=400, detail="La lista de prendas incluidas contiene duplicados.")
        if options.garment_edits is not None:
            edit_indices = [edit.index for edit in options.garment_edits]
            if len(edit_indices) != len(set(edit_indices)) or set(edit_indices) != set(garment_indices):
                raise HTTPException(
                    status_code=400,
                    detail="Debe haber una configuración distinta para cada prenda incluida.",
                )
            garment_edits = {}
            for edit in options.garment_edits:
                edit_color = edit.color.strip()
                if edit_color.lower() == C.PIX2PIX_NO_TEXTURE:
                    edit_color = ""
                edit_texture = edit.texture.strip() or C.PIX2PIX_NO_TEXTURE
                edit_extra = edit.extra.strip()
                if (not edit_color and edit.hsv is None and edit_texture.lower() == C.PIX2PIX_NO_TEXTURE
                        and not edit_extra and not edit.placements):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Indica un cambio para la prenda {segmentation['garments'][edit.index]['name']}.",
                    )
                if edit.hsv is not None:
                    # color de la paleta (salvo negro/blanco) o personalizado: lo aplica recolor_hsv, no el modelo
                    hsv_settings.append({
                        "index": edit.index,
                        "hue": edit.hsv.hue,
                        "saturation": edit.hsv.saturation,
                        "value": edit.hsv.value,
                    })
                    edit_color = ""
                needs_model = bool(
                    edit_color or edit_texture.lower() != C.PIX2PIX_NO_TEXTURE
                    or edit_extra or edit.placements
                )
                if needs_model:
                    garment_edits[edit.index] = {
                        "color": edit_color,
                        "texture": edit_texture,
                        "extra": edit_extra,
                        "hsv": None,
                        "placements": edit.placements,
                        "texture_path": None,
                    }
            # al modelo solo van las prendas que lo necesitan; el resto se recolorea con HSV
            garment_indices = [index for index in garment_indices if index in garment_edits]
    elif options.garment_indices is not None:
        raise HTTPException(status_code=400, detail="Primero detecta las prendas de la imagen.")
    elif options.garment_edits is not None:
        raise HTTPException(status_code=400, detail="Primero detecta las prendas de la imagen.")

    with job_lock:
        if request.app.state.segmenting:
            raise HTTPException(status_code=409, detail="Espera a que termine la detección de prendas.")
        queue_position = 1 + sum(
            job["status"] in {"queued", "running"} for job in jobs.values()
        )
        job_id = uuid.uuid4().hex
        jobs[job_id] = {
            "status": "queued",
            "message": "Esperando al modelo…",
            "error": None,
        }
        if worker is None:
            from concurrent.futures import ThreadPoolExecutor

            worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recolor")
        worker.submit(
            run_generation,
            request.app.state,
            job_id,
            str(source),
            color,
            texture_name,
            extra,
            texture_path,
            segmentation,
            garment_indices,
            garment_edits,
            hsv_settings,
        )
    return {"id": job_id, "status": "queued", "queue_position": queue_position}


@app.post("/api/recolor-hsv")
async def recolor_with_hsv(request: Request, options: HsvRecolorRequest):
    app_state = request.app.state
    source = source_path(app_state, options.source_id)
    segmentation = app_state.segmentations.get(options.segmentation_id)
    if segmentation is None or segmentation["source_id"] != options.source_id:
        raise HTTPException(status_code=400, detail="Vuelve a detectar las prendas de la imagen seleccionada.")

    garment_settings = []
    if options.garments:
        seen = set()
        for edit in options.garments:
            if edit.index < 0 or edit.index >= len(segmentation["garments"]):
                raise HTTPException(status_code=400, detail="Una de las prendas seleccionadas ya no está disponible.")
            if edit.index in seen:
                raise HTTPException(status_code=400, detail="La lista de prendas contiene duplicados.")
            seen.add(edit.index)
            garment_settings.append({
                "index": edit.index,
                "hue": edit.hue,
                "saturation": edit.saturation,
                "value": edit.value,
            })
    else:
        indices = options.garment_indices or []
        if not indices:
            raise HTTPException(status_code=400, detail="Selecciona al menos una prenda para recolorear.")
        if any(index < 0 or index >= len(segmentation["garments"]) for index in indices):
            raise HTTPException(status_code=400, detail="Una de las prendas seleccionadas ya no está disponible.")
        if len(indices) != len(set(indices)):
            raise HTTPException(status_code=400, detail="La lista de prendas contiene duplicados.")
        if options.hue is None or options.saturation is None or options.value is None:
            raise HTTPException(status_code=400, detail="Faltan los valores de tono, saturación y brillo.")
        for index in indices:
            garment_settings.append({
                "index": index,
                "hue": options.hue,
                "saturation": options.saturation,
                "value": options.value,
            })

    job_id = uuid.uuid4().hex
    try:
        await run_in_threadpool(
            run_hsv_recolor, app_state, job_id, source, segmentation, garment_settings,
        )
    except Exception as exc:
        logger.exception("Falló el recoloreado HSV %s", job_id)
        raise HTTPException(status_code=500, detail=f"No se pudo recolorear con HSV: {exc}")
    return {"id": job_id, "status": "completed"}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    with job_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No se encontró esta generación.")
        queue_position = None
        if job["status"] == "running":
            queue_position = 1
        elif job["status"] == "queued":
            queue_position = 1
            for earlier_id, earlier_job in jobs.items():
                if earlier_id == job_id:
                    break
                if earlier_job["status"] in {"queued", "running"}:
                    queue_position += 1
        return {
            "id": job_id,
            "queue_position": queue_position,
            **{key: value for key, value in job.items() if key not in {"images", "saved_path"}},
        }


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    with job_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No se encontró esta generación.")
        if job["status"] in {"completed", "failed", "cancelled"}:
            return {"id": job_id, "status": job["status"]}
        job["cancel_requested"] = True
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["message"] = "Generación cancelada"
        else:
            job["message"] = "Cancelando la generación…"
        return {"id": job_id, "status": job["status"]}


@app.get("/api/jobs/{job_id}/image/{image_name}")
async def job_image(job_id: str, image_name: str, download: bool = False, name: str | None = None):
    if image_name not in {"original", "mask", "edited", "final"}:
        raise HTTPException(status_code=404, detail="Vista de resultado desconocida.")
    with job_lock:
        job = jobs.get(job_id)
        path = job.get("images", {}).get(image_name) if job else None
        status = job.get("status") if job else None
    if status != "completed" or path is None or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="La imagen todavía no está disponible.")
    filename = None
    if image_name == "final" and download:
        filename = safe_download_filename(name, Path(job["saved_path"]).name)
    return FileResponse(path, media_type="image/png", filename=filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    threading.Timer(
        1,
        webbrowser.open,
        args=("http://127.0.0.1:8000",),
        kwargs={"new": 2},
    ).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)