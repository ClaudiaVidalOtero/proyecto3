# proyecto3

## Entrenamiento de segmentacion

El pipeline conecta `src/dataset.py` con `src/model.py`. El dataset devuelve
objetivos por instancia (`masks`, `boxes` y `labels`) compatibles directamente
con Mask R-CNN y el entrenamiento guarda `best.pth` y `last.pth` en `checkpoints`.

Desde la raiz del proyecto:

```bash
.venv/Scripts/python.exe train_pipeline.py --epochs 10 --batch-size 2
```

Para comprobar el flujo rapidamente sin descargar pesos preentrenados:

```bash
.venv/Scripts/python.exe train_pipeline.py --epochs 1 --batch-size 1 --image-size 64 --max-train-images 2 --max-val-images 2 --no-pretrained
```

Las imagenes de validacion se leen desde `dataset/test/`, usando sus
anotaciones de `instances_attributes_val2020.json`.