"""
Crea un JSON de anotaciones reducido, quedándose solo con las imágenes
que existen en una carpeta (por ejemplo train_no_humans).

El JSON original no se modifica, se escribe uno nuevo.
"""

import json
import os


def filter_json(ann_file, images_dir, output_file):
    print(f"\nCargando {ann_file} ...")
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # nombres de archivo que hay realmente en la carpeta
    files_in_dir = set(os.listdir(images_dir))

    # nos quedamos solo con las imágenes cuyo archivo está en la carpeta
    kept_images = [img for img in data["images"] if img["file_name"] in files_in_dir]
    kept_ids = {img["id"] for img in kept_images}

    # nos quedamos solo con las anotaciones (prendas) de esas imágenes
    kept_annotations = [ann for ann in data["annotations"] if ann["image_id"] in kept_ids]

    # copiamos el resto de bloques tal cual (categories, attributes, info...)
    new_data = dict(data)
    new_data["images"] = kept_images
    new_data["annotations"] = kept_annotations

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f)

    # resumen y comprobaciones
    print(f"Archivos en la carpeta:      {len(files_in_dir)}")
    print(f"Imágenes conservadas:        {len(kept_images)} de {len(data['images'])}")
    print(f"Anotaciones conservadas:     {len(kept_annotations)} de {len(data['annotations'])}")
    print(f"JSON reducido guardado en:   {output_file}")

    names_in_json = {img["file_name"] for img in data["images"]}
    not_found = files_in_dir - names_in_json
    if not_found:
        print(
            f"ERROR: {len(not_found)} archivos de la carpeta NO aparecen en este JSON "
        )


if __name__ == "__main__":

    filter_json(
        ann_file="dataset/instances_attributes_train2020.json",
        images_dir="dataset/test_no_humans",
        output_file="dataset/instances_test_no_humans.json",
    )