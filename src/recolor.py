"""
Recolorea prendas segmentadas cambiando solo el tono (H) en HSV.
Saturación y valor se reescalan para conservar pliegues, sombras y brillos.
"""
import cv2
import numpy as np


def recolor_hsv(image_rgb, binary_mask, new_hue_degrees,
                target_saturation=200, target_value=220, min_value_floor=40,
                alpha=None):
    """
    image_rgb:  (H, W, 3) uint8, RGB.
    binary_mask: (H, W) bool, True donde está la prenda.
    new_hue_degrees: tono nuevo en grados (0=rojo, 120=verde, 240=azul...).
    target_saturation: saturación (0-255) fija en toda la prenda.
    target_value: brillo (0-255) del píxel más brillante de la prenda; el resto
                  se reescala proporcionalmente (conserva pliegues y sombras).
    min_value_floor: brillo mínimo, para que las sombras no lleguen a negro puro.
    alpha: (H, W) float 0-1 opcional. Si se pasa, el resultado se mezcla con la
           imagen original usando esta máscara suave (bordes sin escalón).
    """
    binary_mask = binary_mask.astype(bool)
    if not binary_mask.any():
        return image_rgb.copy()

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.int32)
    new_hue_opencv = int(new_hue_degrees / 2) % 180      # OpenCV: H en 0-179

    current_value = hsv[:, :, 2]
    max_value = max(int(current_value[binary_mask].max()), 1)
    new_value = np.maximum(current_value / max_value * target_value, min_value_floor)

    hsv[:, :, 0] = np.where(binary_mask, new_hue_opencv, hsv[:, :, 0])
    hsv[:, :, 1] = np.where(binary_mask, target_saturation, hsv[:, :, 1])
    hsv[:, :, 2] = np.where(binary_mask, new_value, current_value)

    recolored = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    if alpha is None:
        return recolored
    a = np.clip(alpha, 0, 1)[..., None]
    return (recolored * a + image_rgb * (1 - a)).astype(np.uint8)


class SelectorColor:
    """
    Ventana interactiva con vista previa en vivo para elegir el color.

      - Barra de tonos: haz clic o arrastra para elegir el tono (H). Si la saturación
        estaba en ~0 (blanco/gris) o el brillo en ~0 (negro), se restauran automáticamente.
      - Sliders de saturación y brillo.
      - "Aplicar" devuelve (hue, sat, val); "Saltar" o cerrar la ventana devuelve None.

    La vista previa se calcula sobre una copia reducida para que sea fluida;
    el recoloreado final se hace después a resolución completa.
    """

    def __init__(self, img, prendas, hue=120, sat=200, val=220, preview_max=700):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        self.plt = plt
        self.hue, self.sat, self.val = float(hue), int(sat), int(val)
        self.resultado = None
        self._arrastrando = False

        h, w = img.shape[:2]
        k = min(1.0, preview_max / max(h, w))
        size = (max(1, int(w * k)), max(1, int(h * k)))
        self.small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        self.masks = [
            (cv2.resize(p["mask_final"].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool),
             cv2.resize(np.asarray(p["alpha"], dtype=np.float32), size, interpolation=cv2.INTER_LINEAR))
            for p in prendas
        ]

        self.fig = plt.figure(figsize=(9, 8))
        try:
            self.fig.canvas.manager.set_window_title("Selector de color")
        except Exception:
            pass
        self.ax_img = self.fig.add_axes([0.05, 0.30, 0.90, 0.66])
        self.ax_hue = self.fig.add_axes([0.05, 0.20, 0.90, 0.07])
        ax_sat = self.fig.add_axes([0.15, 0.12, 0.70, 0.03])
        ax_val = self.fig.add_axes([0.15, 0.07, 0.70, 0.03])
        ax_ok = self.fig.add_axes([0.55, 0.01, 0.20, 0.05])
        ax_skip = self.fig.add_axes([0.77, 0.01, 0.18, 0.05])

        self.im = self.ax_img.imshow(self._render())
        self.ax_img.axis("off")

        self.strip = self.ax_hue.imshow(self._strip(), extent=[0, 360, 0, 1], aspect="auto")
        self.ax_hue.set_xticks(range(0, 361, 60))
        self.ax_hue.set_yticks([])
        self.ax_hue.set_xlabel("Tono (clic o arrastra)", fontsize=8)
        self.marker_bg = self.ax_hue.axvline(self.hue, color="black", lw=5)
        self.marker = self.ax_hue.axvline(self.hue, color="white", lw=2)

        self.s_sat = Slider(ax_sat, "Saturación", 0, 255, valinit=self.sat, valstep=1)
        self.s_val = Slider(ax_val, "Brillo", 0, 255, valinit=self.val, valstep=1)
        self.s_sat.on_changed(self._on_sliders)
        self.s_val.on_changed(self._on_sliders)

        self.b_ok = Button(ax_ok, "Aplicar")
        self.b_skip = Button(ax_skip, "Saltar")
        self.b_ok.on_clicked(self._aplicar)
        self.b_skip.on_clicked(self._saltar)

        self.fig.canvas.mpl_connect("button_press_event", self._press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._motion)
        self.fig.canvas.mpl_connect("button_release_event", self._release)
        self._titulo()

    # -- dibujo
    def _render(self):
        out = self.small
        for m, a in self.masks:
            out = recolor_hsv(out, m, self.hue, self.sat, self.val, alpha=a)
        return out

    def _strip(self):
        hsv = np.zeros((1, 360, 3), np.uint8)
        hsv[0, :, 0] = (np.arange(360) // 2).astype(np.uint8)
        hsv[0, :, 1] = 255      # siempre colores puros: la barra se ve y se puede
        hsv[0, :, 2] = 255      # usar aunque los sliders estén en blanco/negro/gris
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _titulo(self):
        self.ax_img.set_title(f"Tono {self.hue:.0f}°  ·  Saturación {self.sat}  ·  Brillo {self.val}")

    def _refresh(self):
        self.im.set_data(self._render())
        self.marker_bg.set_xdata([self.hue, self.hue])
        self.marker.set_xdata([self.hue, self.hue])
        self._titulo()
        self.fig.canvas.draw_idle()

    # -- eventos
    def _set_hue_from(self, event):
        if event.inaxes is self.ax_hue and event.xdata is not None:
            self.hue = float(np.clip(event.xdata, 0, 359))
            # Con saturación ~0 (blanco/gris) o brillo ~0 (negro) el tono no tiene efecto:
            # al elegir un color de la barra se restauran valores que lo dejen ver.
            if self.sat < 40:
                self.s_sat.set_val(200)
            if self.val < 60:
                self.s_val.set_val(220)
            self._refresh()

    def _press(self, event):
        if event.button == 1 and event.inaxes is self.ax_hue:
            self._arrastrando = True
            self._set_hue_from(event)

    def _motion(self, event):
        if self._arrastrando:
            self._set_hue_from(event)

    def _release(self, event):
        self._arrastrando = False

    def _on_sliders(self, _):
        self.sat, self.val = int(self.s_sat.val), int(self.s_val.val)
        self._refresh()

    def _aplicar(self, _):
        self.resultado = (int(round(self.hue)), self.sat, self.val)
        self.plt.close(self.fig)

    def _saltar(self, _):
        self.resultado = None
        self.plt.close(self.fig)

    def mostrar(self):
        """Abre la ventana (bloqueante) y devuelve (hue, sat, val) o None."""
        self.plt.show()
        return self.resultado
