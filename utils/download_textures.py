"""
Descarga muestras de textura de ambientCG (https://ambientcg.com, de dominio público)
y las guarda como un pequeño dataset para elegirlas visualmente en recolor.py.

De cada material se guarda solo el mapa de color (el "_Color.jpg" del zip de 1K),
recortado a cuadrado y reducido a TEXTURE_SIZE píxeles:

    textures/
        index.json                 id, categoría, etiquetas, archivo y enlace de origen
        leather/Leather012.jpg
        fabric/Fabric045.jpg
        ...

Uso:
    python download_textures.py
    python download_textures.py --max-per-category 20
    python download_textures.py --categories Leather Fabric Carpet

Es reanudable, es decir, lo que ya está descargado se salta. 
Cada textura son más o menos 5MB de descarga (el zip completo), 
aunque en disco solo se queda una imagen pequeña.
"""

import argparse
import io
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request
import zipfile

from PIL import Image, ImageOps

API_URL = "https://ambientcg.com/api/v2/full_json"
OUTPUT_DIR = "textures"
# Categorías de ambientCG que tienen sentido para ropa. No existen categorías propias para
# denim, seda, terciopelo, etc. Todo eso está dentro de "Fabric" (y se distingue por etiquetas).
DEFAULT_CATEGORIES = ["Leather", "Fabric", "Carpet", "Net", "Rubber"]
MAX_PER_CATEGORY = 40
RESOLUTION = "1K-JPG"      # versión del zip que se descarga
TEXTURE_SIZE = 512         # tamaño final de la imagen guardada
USER_AGENT = "Mozilla/5.0 (personal texture dataset downloader)"  # sin esto la web a veces nos bloquea
PAUSE_BETWEEN_DOWNLOADS = 0.5   # segundos, para no saturar el servidor


def http_get(url, retries=3):
    # hacemos la petición con un User-Agent "de navegador" para que no nos rechacen
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except Exception:
            # si es el último intento dejamos que el error suba. Si no, esperamos y reintentamos
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))  # espera cada vez más larga. 2s, 4s...


def list_assets(category, limit):
    """Devuelve hasta 'limit' materiales de la categoría, los más populares primero."""
    assets, offset = [], 0
    while len(assets) < limit:
        # la API da como máximo 100 por página, así que pedimos por trozos
        page_size = min(100, limit - len(assets))
        params = {
            "type": "Material",
            "category": category,
            "sort": "Popular",
            "limit": page_size,
            "offset": offset,   # desde qué resultado empieza esta página
            "include": "downloadData,previewData",  # sin esto no vienen los enlaces de descarga
        }
        data = json.loads(http_get(f"{API_URL}?{urllib.parse.urlencode(params)}"))
        found = data.get("foundAssets", [])
        if not found:
            break
        assets.extend(found)
        offset += len(found)
        # si llegaron menos de los pedidos es que ya no quedan más en la categoría
        if len(found) < page_size:
            break
    return assets[:limit]


def find_zip_url(asset):
    # el enlace está metido muy dentro del json, por eso tanto corchete
    try:
        downloads = asset["downloadFolders"]["default"]["downloadFiletypeCategories"]["zip"]["downloads"]
    except KeyError:
        return None  # si falta alguna clave, este material no tiene zip y lo saltamos
    # de todos los zips disponibles nos quedamos con el de la resolución que queremos (1K)
    for download in downloads:
        if download.get("attribute") == RESOLUTION:
            return download.get("downloadLink") or download.get("fullDownloadPath")
    return None


def extract_color_map(zip_bytes):
    """Saca el mapa de color (…_Color.jpg/png) de dentro del zip, sin escribir el zip a disco."""
    # io.BytesIO hace que los bytes descargados se comporten como un archivo en memoria
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            # el zip trae muchos mapas (normal, rugosidad...) y solo queremos el de color
            if re.search(r"_Color\.(jpg|jpeg|png)$", name, re.IGNORECASE):
                return Image.open(io.BytesIO(zf.read(name))).convert("RGB")
    return None


def load_index(index_path):
    # si ya existe un index.json lo cargamos como diccionario {id: entrada} para poder reanudar
    if index_path.exists():
        return {entry["id"]: entry for entry in json.loads(index_path.read_text(encoding="utf-8"))}
    return {}


def save_index(index_path, entries):
    # lo guardamos ordenado por categoría e id para que el archivo quede siempre igual y se lea bien
    ordered = sorted(entries.values(), key=lambda e: (e["category"], e["id"]))
    index_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    # argumentos de la línea de comandos (todos opcionales, tienen valor por defecto)
    parser = argparse.ArgumentParser(description="Descarga muestras de textura de ambientCG")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                        help="categorías de ambientCG (por defecto: %(default)s)")
    parser.add_argument("--max-per-category", type=int, default=MAX_PER_CATEGORY,
                        help="máximo de texturas por categoría (por defecto: %(default)s)")
    parser.add_argument("--output", default=OUTPUT_DIR, help="carpeta de salida (por defecto: %(default)s)")
    args = parser.parse_args()

    # creamos la carpeta de salida (si ya existe no pasa nada) y cargamos lo que ya hubiera
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    entries = load_index(index_path)

    # contadores para el resumen del final
    downloaded = skipped = failed = 0
    for category in args.categories:
        print(f"\n=== {category} ===")
        try:
            assets = list_assets(category, args.max_per_category)
        except Exception as exc:
            print(f"  No se pudo consultar la categoría: {exc}")
            continue
        if not assets:
            print("  Sin resultados (¿el nombre de la categoría es correcto?)")
            continue

        # una subcarpeta por categoría, en minúsculas (leather/, fabric/...)
        category_dir = out_dir / category.lower()
        category_dir.mkdir(exist_ok=True)

        for number, asset in enumerate(assets, start=1):
            asset_id = asset["assetId"]
            file_rel = f"{category.lower()}/{asset_id}.jpg"  # ruta relativa, la que va en el index.json
            file_path = out_dir / file_rel

            # si ya la teníamos descargada y apuntada en el índice, no la bajamos otra vez
            if file_path.exists() and asset_id in entries:
                skipped += 1
                continue

            zip_url = find_zip_url(asset)
            if not zip_url:
                print(f"  [{number}/{len(assets)}] {asset_id}: sin zip {RESOLUTION}, se salta")
                failed += 1
                continue

            try:
                color_map = extract_color_map(http_get(zip_url))
                if color_map is None:
                    raise ValueError("el zip no contiene mapa de color")
                # ImageOps.fit recorta al centro y redimensiona a la vez, así queda cuadrada sin deformarse
                ImageOps.fit(color_map, (TEXTURE_SIZE, TEXTURE_SIZE), Image.LANCZOS).save(
                    file_path, quality=92)
            except Exception as exc:
                # si una textura falla no paramos todo, la contamos y pasamos a la siguiente
                print(f"  [{number}/{len(assets)}] {asset_id}: ERROR ({exc})")
                failed += 1
                continue

            # apuntamos los datos de la textura en el índice (recolor.py lee estos campos)
            entries[asset_id] = {
                "id": asset_id,
                "category": category,
                "name": asset.get("displayName") or asset_id,
                "tags": asset.get("tags") or [],
                "file": file_rel,
                "source": asset.get("shortLink") or f"https://ambientcg.com/a/{asset_id}",
            }
            save_index(index_path, entries)   # se guarda tras cada una, así si se corta no se pierde nada
            downloaded += 1
            print(f"  [{number}/{len(assets)}] {asset_id} ✓")
            time.sleep(PAUSE_BETWEEN_DOWNLOADS)  # pausita entre descargas para no pasarnos con el servidor

    print(f"\nListo: {downloaded} nuevas, {skipped} ya estaban, {failed} con error.")
    print(f"Total en {index_path}: {len(entries)} texturas.")


if __name__ == "__main__":
    main()
