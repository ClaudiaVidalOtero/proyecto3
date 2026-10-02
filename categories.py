"""
Qué categorías de Fashionpedia se recolorean, se suman o se restan (lógica prenda + partes − cierres).

- PRENDAS : las prendas que se van a recolorear.
- PARTES  : se SUMAN a la prenda a la que pertenecen.
- CIERRES : se RESTAN (no se quiere recolorear una cremallera metálica ni unas lentejuelas).
- Lo que no está en ningún conjunto (neckline, bow, ribbon, flower, fringe, tassel y los
  accesorios) ni se suma ni se resta. `neckline` se deja fuera a propósito: su anotación suele
  incluir la abertura del cuello (piel).
"""

PRENDAS = {
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest",
    "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape",
}
PARTES = {"hood", "collar", "lapel", "epaulette", "sleeve", "pocket", "ruffle"}
CIERRES = {"zipper", "buckle", "rivet", "bead", "sequin", "applique"}
RELEVANTES = PRENDAS | PARTES | CIERRES

# Prioridad cuando dos prendas se solapan: la de mayor número se queda con la zona común.
PRIORIDAD = {
    "coat": 6, "cape": 6,
    "jacket": 5, "cardigan": 5,
    "vest": 4,
    "sweater": 3, "shirt, blouse": 3, "top, t-shirt, sweatshirt": 3,
    "dress": 2, "jumpsuit": 2,
    "pants": 1, "shorts": 1, "skirt": 1,
}


def check_names(label2name):
    """
    Devuelve (no_encontrados, sin_usar): nombres escritos aquí que no existen en el JSON, y
    categorías del JSON que ni se suman ni se restan.
    """
    nombres_dataset = set(label2name.values())
    return RELEVANTES - nombres_dataset, nombres_dataset - RELEVANTES


def print_check(label2name):
    no_encontrados, sin_usar = check_names(label2name)
    if no_encontrados:
        print("Estos nombres no existen en tu JSON, corrígelos en categories.py:", sorted(no_encontrados))
    else:
        print("✓ Todos los nombres coinciden con las categorías del dataset")
    print("Categorías que ni se suman ni se restan:", sorted(sin_usar))
