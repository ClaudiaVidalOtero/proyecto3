from pathlib import Path
from collections import Counter

import json
import matplotlib.pyplot as plt


# path del directorio base y del dataset
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

ANNOTATIONS_FILE = DATASET_DIR / "instances_attributes_train2020.json"


# Cargamos las anotaciones del dataset
print(f"Cargando: {ANNOTATIONS_FILE}")

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


# Extraemos info del dataset
images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

print("\n" + "=" * 60)
print("INFORMACIÓN GENERAL")
print("=" * 60)

print(f"Número de imágenes:       {len(images):,}")
print(f"Número de anotaciones:    {len(annotations):,}")
print(f"Número de categorías:     {len(categories):,}")


 
# Categorías por ID
category_names = { category["id"]: category["name"] for category in categories}


# Contamos el número de objetos por categoría
counter = Counter()

for annotation in annotations:
    category_id = annotation["category_id"]
    counter[category_id] += 1


 
# Mostramos el número de objetos por categoría
print("\n" + "=" * 60)
print("OBJETOS POR CATEGORÍA")
print("=" * 60)

results = []

total_objects = len(annotations)

for category_id, count in counter.most_common():

    name = category_names.get( category_id, f"Unknown ({category_id})")

    percentage = (count / total_objects) * 100

    results.append((name, count, percentage))

    print(
        f"{name:<40} "
        f"{count:>8,} "
        f"({percentage:>6.2f}%)"
    )


 
# Categorías vacías
print("\n" + "=" * 60)
print("CATEGORÍAS SIN OBJETOS")
print("=" * 60)

categories_without_objects = []

for category_id, name in category_names.items():

    if counter[category_id] == 0:
        categories_without_objects.append(name)

if categories_without_objects:

    for name in categories_without_objects:
        print(f"- {name}")

else:
    print("Todas las categorías tienen al menos un objeto.")


 
# Balanceo del dataset
counts = list(counter.values())

if counts:

    max_count = max(counts)
    min_count = min(counts)

    max_category_id = counter.most_common(1)[0][0]
    min_category_id = min(counter, key=counter.get)

    max_category = category_names[max_category_id]
    min_category = category_names[min_category_id]

    print("\n" + "=" * 60)
    print("BALANCEO DEL DATASET")
    print("=" * 60)

    print(f"Categoría más frecuente: "f"{max_category} ({max_count:,})")

    print(f"Categoría menos frecuente: "f"{min_category} ({min_count:,})")

    print(f"Ratio máximo/mínimo: "f"{max_count / min_count:.2f}x")


 
# GRÁFICA
names = [category_names[category_id] for category_id, _ in counter.most_common()]

values = [count for _, count in counter.most_common()]

plt.figure(figsize=(14, 8))

plt.barh(names[::-1], values[::-1])

plt.xlabel("Número de objetos")
plt.ylabel("Categoría")
plt.title("Distribución de categorías - Fashionpedia")

plt.tight_layout()

plt.show()