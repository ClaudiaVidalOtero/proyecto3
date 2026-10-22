# proyecto3

## Interfaz web de recoloreado

Desde la raíz del proyecto, activa el entorno e inicia la interfaz:

```bash
.venv/Scripts/python.exe interfaz.py
```

El script abre Atelier en el navegador predeterminado de Windows, fuera de VS Code.
Mantén abierta la ventana de la terminal mientras uses la aplicación; al detener el
servidor, el navegador ya no podrá conectar con el modelo. También puedes abrir
[http://127.0.0.1:8000](http://127.0.0.1:8000) manualmente. Permite
elegir imágenes del catálogo o subir una propia, seleccionar color y textura (o
**none** para conservar el color o material original),
añadir una instrucción, consultar el historial de diseños y descargar el resultado
con un nombre PNG personalizado. Usa **Detectar prendas** para superponer las
máscaras del modelo sobre la foto; pasa el cursor para identificar las prendas,
haz clic para seleccionar una o varias y genera con el color y la textura elegidos.
Al detectar varias prendas, aparece una pestaña de configuración por cada una;
puedes asignarles colores, materiales, muestras e instrucciones independientes y
generarlas juntas en una sola imagen.
En la configuración de cada prenda también puedes añadir bolsillos o botones:
elige el elemento y pulsa sobre la prenda para marcar su ubicación; puedes quitar
los puntos añadidos antes de generar. Los bolsillos se componen como un parche
pequeño con textura local y pespuntes; los botones se dibujan con relieve,
agujeros y sombra. Así se ven sin depender de que el modelo los invente. La forma
de la prenda determina parte del aspecto; la colocación sigue siendo orientativa.
La instrucción adicional se envía junto con esos cambios. Si eliges una muestra
visual, se usa como referencia de textura; si no se puede cargar, la generación
se detiene y muestra un error en vez de omitir la muestra silenciosamente.
Las generaciones guardan la máscara aplicada y la segmentación por prenda junto
al resultado. Al activar el modo desarrollador y abrir una generación pasada,
puedes consultar esas máscaras; se eliminan junto al diseño desde el historial.
El montaje final amplía ligeramente la máscara para cubrir bordes sin teñir y
recupera parte del contraste de pliegues y costuras desde la imagen original.
La generación reutiliza los modelos Mask R-CNN, SAM 2 e InstructPix2Pix del
proyecto. También se puede seguir usando la interfaz de escritorio con
`python src/recolor.py`.

### PyTorch con CUDA (Windows, NVIDIA)

Para usar la GPU NVIDIA en lugar de CPU, instala las ruedas CUDA fijadas en
`requirements.txt` desde el índice oficial de PyTorch:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

Verifica que PyTorch detecta la GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

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