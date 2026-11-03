const $ = (selector) => document.querySelector(selector);
const state = { source: null, activeJob: null, pendingJobs: [], pollingJobs: false, submittingJob: false, cancelling: false, detectingGarments: false, segmentation: null, garmentEdits: new Map(), colorLayers: new Map(), activeGarmentTab: null, activePlacement: null, galleryHistory: [], galleryIndex: -1, devMode: false, historyItem: null, historyFilename: null, savedName: "", lastResultGarment: "", allowReload: false };
// modo desarrollador: se guarda en el navegador para que se recuerde al recargar la página
const DEV_MODE_KEY = "atelier-dev-mode";
const THEME_KEY = "atelier-theme";
function readDevMode() {
  try { return localStorage.getItem(DEV_MODE_KEY) === "1"; } catch (error) { return false; }
}
function readTheme() {
  try { return localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light"; }
  catch (error) { return "light"; }
}
function setTheme(theme) {
  const selectedTheme = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = selectedTheme;
  $("#themeLight").checked = selectedTheme === "light";
  $("#themeDark").checked = selectedTheme === "dark";
  $('meta[name="theme-color"]').content = selectedTheme === "dark" ? "#191a18" : "#f7f7f4";
  try { localStorage.setItem(THEME_KEY, selectedTheme); } catch (error) { /* sin almacenamiento: no se recuerda */ }
}
function setDevMode(enabled) {
  state.devMode = enabled;
  try { localStorage.setItem(DEV_MODE_KEY, enabled ? "1" : "0"); } catch (error) { /* sin almacenamiento: no se recuerda */ }
  document.body.classList.toggle("dev-mode", enabled);
  $("#devModeToggle").checked = enabled;
  if (state.segmentation) updateGarmentOptions();  // repinta los botones de prenda con o sin porcentaje
  renderHistorySegmentation();
}
function closeSettings() {
  $("#settingsMenu").hidden = true;
  $("#settingsButton").setAttribute("aria-expanded", "false");
}
function garmentLabel(item) {
  return state.devMode ? `${item.name} · ${Math.round(item.score * 100)}%` : item.name;
}
const colorHex = {
  black:"#252525", white:"#ffffff", red:"#c94843", blue:"#466ca5", "navy blue":"#293650",
  green:"#66866a", "dark green":"#345343", yellow:"#e9c44e", pink:"#dc91a4",
  purple:"#886c9f", orange:"#e29150", brown:"#866044", gray:"#91938e", beige:"#d5c4a2"
};
async function request(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error("No se puede conectar con Atelier. Inicia interfaz.py y mantén abierta la ventana del servidor.");
    }
    throw error;
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || "No se pudo completar la solicitud.");
  return body;
}
function showError(message) {
  const box = $("#errorMessage");
  box.textContent = message;
  box.classList.add("visible");
}
function clearError() { $("#errorMessage").classList.remove("visible"); }
function setSource(source, previewUrl) {
  if (state.detectingGarments) return;
  if (state.source !== source) clearSegmentation();
  state.source = source;
  $("#sourceImage").src = previewUrl;
  $("#sourcePreview").classList.add("has-image");
  document.querySelectorAll(".thumb").forEach((el) => el.classList.toggle("selected", el.dataset.id === source));
  clearError();
}
function setImageSelectionLocked(locked) {
  state.detectingGarments = locked;
  $("#uploadButton").disabled = locked;
  $("#refreshImages").disabled = locked;
  $("#fileInput").disabled = locked;
  document.querySelectorAll(".thumb").forEach((button) => { button.disabled = locked; });
  updateGalleryNav();
}
function updateStylePanel() {
  // sin prendas detectadas la tarjeta queda en blanco (solo con un aviso) y sin el botón de crear;
  // si hay una generación en marcha el botón y la cola se siguen mostrando para poder cancelarla
  const detected = Boolean(state.segmentation);
  const selected = Boolean(state.segmentation && state.segmentation.selected.size > 0);
  const busy = state.submittingJob || state.pendingJobs.length > 0;
  $("#styleCard").classList.toggle("blank", !detected);
  document.querySelector("#styleCard .card-head").hidden = !selected;
  document.querySelector(".generate-actions").hidden = !(selected || busy);
}
function clearSegmentation(message = "") {
  state.segmentation = null;
  state.garmentEdits.clear();
  state.colorLayers.clear();
  state.activeGarmentTab = null;
  $("#defaultStyle").hidden = true;  // el panel manual de color/textura ya no se muestra: solo el de cada prenda detectada
  $("#garmentStyleEditor").classList.remove("visible");
  updateStylePanel();
  const canvas = $("#segmentationOverlay");
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  $("#sourcePreview").classList.remove("has-segmentation");
  $("#sourcePreview").classList.remove("placing");
  $("#garmentOptions").replaceChildren();
  const status = $("#segmentationStatus");
  status.textContent = message;
  status.classList.toggle("visible", Boolean(message));
}
function drawSegmentationOverlay() {
  const segmentation = state.segmentation;
  if (!segmentation) return;
  const canvas = $("#segmentationOverlay");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  // Sin tinte de máscara: la foto se ve tal cual y solo se dibuja encima el color elegido de cada prenda.
  segmentation.items.forEach((item, index) => {
    const layer = state.colorLayers.get(index);
    if (layer && segmentation.selected.has(index)) context.drawImage(layer, 0, 0, canvas.width, canvas.height);
  });
  segmentation.items.forEach((item, index) => {
    const edit = state.garmentEdits.get(index);
    if (!edit) return;
    edit.points.forEach((point) => {
      const x = item.bounds.left + point.x / 100 * Math.max(item.bounds.width - 1, 0);
      const y = item.bounds.top + point.y / 100 * Math.max(item.bounds.height - 1, 0);
      context.beginPath();
      context.arc(x, y, Math.max(10, canvas.width / 80), 0, Math.PI * 2);
      context.fillStyle = point.type === "pocket" ? "#4a7728" : "#862d67";
      context.fill();
      context.lineWidth = Math.max(2, canvas.width / 400);
      context.strokeStyle = "#fff";
      context.stroke();
      context.fillStyle = "#fff";
      context.font = `bold ${Math.max(12, canvas.width / 55)}px sans-serif`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(point.type === "pocket" ? "P" : "B", x, y);
    });
  });
}
function updateGarmentOptions() {
  const list = $("#garmentOptions");
  list.replaceChildren();
  const segmentation = state.segmentation;
  if (!segmentation) return;
  segmentation.items.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "garment-choice";
    button.classList.toggle("selected", segmentation.selected.has(index));
    button.setAttribute("aria-pressed", String(segmentation.selected.has(index)));
    button.textContent = garmentLabel(item);
    button.addEventListener("click", () => toggleGarment(index));
    list.append(button);
  });
  updateGarmentTabs();
}
function createGarmentEditor(index, item, config) {
  const edit = state.garmentEdits.get(index);
  const pane = document.createElement("section");
  pane.className = "garment-editor-pane";
  pane.id = `garment-pane-${index}`;
  pane.setAttribute("role", "tabpanel");
  const colorField = document.createElement("div");
  colorField.className = "field";
  const colorLabel = document.createElement("label");
  colorLabel.className = "field-label field-label-lg";
  colorLabel.textContent = "Color de ejemplo";
  const DEFAULT_HEX = "#4a7728";
  const swatches = document.createElement("div");
  swatches.className = "swatches";
  swatches.setAttribute("aria-label", `Colores para ${item.name}`);
  for (const [name, hex] of Object.entries(colorHex)) {
    const swatch = document.createElement("button");
    swatch.type = "button";
    swatch.className = "swatch";
    swatch.title = name;
    swatch.dataset.value = name;
    swatch.setAttribute("aria-label", `${name} (ejemplo de color)`);
    swatch.style.background = hex;
    swatch.classList.toggle("selected", edit.color.trim().toLowerCase() === name);
    swatch.addEventListener("click", () => chooseColor(name));
    swatches.append(swatch);
  }

  const modeHint = document.createElement("p");
  modeHint.className = "hint color-mode-hint";
  modeHint.hidden = true;

  const modelColorHint = document.createElement("p");
  modelColorHint.className = "hint model-color-hint";
  modelColorHint.hidden = true;

  const hsvView = document.createElement("div");
  hsvView.className = "color-hsv-view";
  const hsvHead = document.createElement("div");
  hsvHead.className = "hsv-inline-head";
  const hsvToggle = document.createElement("button");
  hsvToggle.type = "button";
  hsvToggle.className = "hsv-toggle";
  hsvToggle.setAttribute("aria-expanded", "false");   // empieza plegado
  hsvToggle.setAttribute("aria-controls", `hsv-body-${index}`);
  const hsvTriangle = document.createElement("span");
  hsvTriangle.className = "hsv-triangle";
  hsvTriangle.setAttribute("aria-hidden", "true");
  const hsvTitle = document.createElement("span");
  hsvTitle.className = "hsv-inline-title";
  hsvTitle.textContent = "Color HSV";
  const hsvMiniChip = document.createElement("span");   // color actual, visible solo con el panel plegado
  hsvMiniChip.className = "hsv-mini-chip";
  hsvMiniChip.setAttribute("aria-hidden", "true");
  hsvToggle.append(hsvTriangle, hsvTitle, hsvMiniChip);
  hsvHead.append(hsvToggle);
  const hsvBody = document.createElement("div");
  hsvBody.className = "hsv-inline-body";
  hsvBody.id = `hsv-body-${index}`;
  hsvBody.hidden = true;
  hsvToggle.addEventListener("click", () => {
    const open = hsvToggle.getAttribute("aria-expanded") !== "true";
    hsvToggle.setAttribute("aria-expanded", String(open));
    hsvBody.hidden = !open;
  });

  const hueWrap = document.createElement("div");
  const hueRow = document.createElement("div");
  hueRow.className = "hsv-row";
  const hueLabel = document.createElement("span");
  hueLabel.className = "field-label";
  hueLabel.textContent = "Tono (clic o arrastra)";
  const hueValue = document.createElement("span");
  hueValue.className = "hsv-value";
  hueRow.append(hueLabel, hueValue);
  const hueStrip = document.createElement("div");
  hueStrip.className = "hsv-strip";
  hueStrip.tabIndex = 0;
  hueStrip.setAttribute("role", "slider");
  hueStrip.setAttribute("aria-label", "Tono");
  hueStrip.setAttribute("aria-valuemin", "0");
  hueStrip.setAttribute("aria-valuemax", "359");
  const hueMarker = document.createElement("span");
  hueMarker.className = "hsv-marker";
  hueStrip.append(hueMarker);
  hueWrap.append(hueRow, hueStrip);

  const satWrap = document.createElement("div");
  const satRow = document.createElement("div");
  satRow.className = "hsv-row";
  const satLabel = document.createElement("span");
  satLabel.className = "field-label";
  satLabel.textContent = "Saturación";
  const satValue = document.createElement("span");
  satValue.className = "hsv-value";
  satRow.append(satLabel, satValue);
  const satRange = document.createElement("input");
  satRange.className = "hsv-range";
  satRange.type = "range";
  satRange.min = "0";
  satRange.max = "255";
  satRange.step = "1";
  satRange.addEventListener("input", () => { edit.hsv.sat = Number(satRange.value); edit.hsvActive = true; updateHsvUi(); });
  satWrap.append(satRow, satRange);

  const valWrap = document.createElement("div");
  const valRow = document.createElement("div");
  valRow.className = "hsv-row";
  const valLabel = document.createElement("span");
  valLabel.className = "field-label";
  valLabel.textContent = "Brillo";
  const valValue = document.createElement("span");
  valValue.className = "hsv-value";
  valRow.append(valLabel, valValue);
  const valRange = document.createElement("input");
  valRange.className = "hsv-range";
  valRange.type = "range";
  valRange.min = "0";
  valRange.max = "255";
  valRange.step = "1";
  valRange.addEventListener("input", () => { edit.hsv.val = Number(valRange.value); edit.hsvActive = true; updateHsvUi(); });
  valWrap.append(valRow, valRange);

  const chipRow = document.createElement("div");
  chipRow.className = "hsv-inline-color";
  const chip = document.createElement("span");
  chip.className = "hsv-chip";
  const chipText = document.createElement("span");
  chipText.className = "hsv-chip-text";
  chipRow.append(chip, chipText);

  const hexRow = document.createElement("div");
  hexRow.className = "hsv-row hex-row";
  const hexLabel = document.createElement("span");
  hexLabel.className = "field-label field-label-lg";
  hexLabel.textContent = "Código hex";
  const hexInput = document.createElement("input");
  hexInput.className = "input hsv-hex-input";
  hexInput.type = "text";
  hexInput.maxLength = 7;
  hexInput.placeholder = "#00ff88";
  hexInput.setAttribute("aria-label", "Código hexadecimal del color");
  hexInput.addEventListener("input", () => {
    const digits = hexInput.value.trim().replace(/^#/, "");
    if (/^[0-9a-f]{6}$/i.test(digits)) applyHexColor("#" + digits);
  });
  hexInput.addEventListener("change", () => {
    if (!applyHexColor(hexInput.value) && edit.hsv) {
      const [red, green, blue] = hsvDegreesToRgb(edit.hsv.hue, edit.hsv.sat, edit.hsv.val);
      hexInput.value = rgbToHex(red, green, blue);
    }
  });
  hexRow.append(hexLabel, hexInput);

  const hsvReset = document.createElement("button");
  hsvReset.type = "button";
  hsvReset.className = "button small color-hsv-reset";
  hsvReset.textContent = "Volver al color original";
  hsvReset.addEventListener("click", () => {
    const original = originalGarmentColor(index);
    edit.color = "";
    edit.hsvActive = false;
    edit.hsv = original ? rgbToHsv(original.red, original.green, original.blue) : hsvFromHex(DEFAULT_HEX);
    updateColorUi();
  });
  const hsvHint = document.createElement("p");
  hsvHint.className = "hint";
  hsvBody.append(hueWrap, satWrap, valWrap, chipRow, hsvHint);
  hsvView.append(hsvHead, hsvBody);

  colorField.append(colorLabel, swatches, hexRow, modeHint, modelColorHint, hsvView, hsvReset);
  pane.append(colorField);

  function updateHsvUi() {
    if (!edit.hsv) return;
    const hue = Math.round(edit.hsv.hue);
    hueValue.textContent = `${hue}°`;
    satValue.textContent = edit.hsv.sat;
    valValue.textContent = edit.hsv.val;
    satRange.value = edit.hsv.sat;
    valRange.value = edit.hsv.val;
    hueMarker.style.left = `${edit.hsv.hue / 360 * 100}%`;
    hueStrip.setAttribute("aria-valuenow", String(hue));
    const [red, green, blue] = hsvDegreesToRgb(edit.hsv.hue, edit.hsv.sat, edit.hsv.val);
    const hex = rgbToHex(red, green, blue);
    if (document.activeElement !== hexInput) hexInput.value = hex;
    chip.style.background = `rgb(${red}, ${green}, ${blue})`;
    hsvMiniChip.style.background = `rgb(${red}, ${green}, ${blue})`;
    chipText.textContent = hex;
    refreshGarmentColorLayer(index, edit);
    if (modelColorHint) {
      const color = (edit.color || "").trim().toLowerCase();
      modelColorHint.hidden = !((color === "black" || color === "white") && !edit.hsvActive);
    }
  }
  function applyHexColor(value) {
    const hex = normalizeHexColor(value);
    if (!hex) return false;
    edit.color = hex;
    edit.hsv = hsvFromHex(hex);
    edit.hsvActive = true;
    updateColorUi();
    return true;
  }
  function updateColorUi() {
    const lower = (edit.color || "").trim().toLowerCase();
    const customHex = normalizeHexColor(lower);
    swatches.querySelectorAll(".swatch").forEach((button) =>
      button.classList.toggle("selected", button.dataset.value === lower));
    if (modelColorHint) {
      if ((lower === "black" || lower === "white") && !edit.hsvActive) {
        const colorName = lower === "black" ? "negro" : "blanco";
        modelColorHint.textContent = `El color ${colorName} se generará con el modelo y se verá al pulsar «Crear mi diseño».`;
        modelColorHint.hidden = false;
      } else {
        modelColorHint.hidden = true;
      }
    }
    if (customHex) {
      modeHint.textContent = `Color personalizado ${customHex}. Ajusta el tono HSV si lo necesitas.`;
      modeHint.hidden = false;
    } else {
      modeHint.textContent = "";
      modeHint.hidden = true;
    }
    if (edit.hsv) updateHsvUi();
  }
  function chooseColor(value) {
    const lower = String(value || "").trim().toLowerCase();
    const hex = normalizeHexColor(lower) || colorHex[lower];
    if (!hex) return;
    edit.color = lower;
    if (lower === "black" || lower === "white") {
      edit.hsvActive = false;
    } else {
      edit.hsv = hsvFromHex(hex);
      edit.hsvActive = true;
    }
    updateColorUi();
  }
  function setHueFromPointer(event) {
    const box = hueStrip.getBoundingClientRect();
    if (!box.width || !edit.hsv) return;
    edit.hsv.hue = Math.min(359, Math.max(0, (event.clientX - box.left) / box.width * 360));
    if (edit.hsv.sat < 40) edit.hsv.sat = 200;
    if (edit.hsv.val < 60) edit.hsv.val = 220;
    edit.hsvActive = true;
    updateHsvUi();
  }
  hueStrip.addEventListener("pointerdown", (event) => {
    hueStrip.setPointerCapture(event.pointerId);
    setHueFromPointer(event);
  });
  hueStrip.addEventListener("pointermove", (event) => {
    if (hueStrip.hasPointerCapture(event.pointerId)) setHueFromPointer(event);
  });
  hueStrip.addEventListener("keydown", (event) => {
    if (!edit.hsv) return;
    const step = event.shiftKey ? 10 : 1;
    if (event.key === "ArrowRight") { edit.hsv.hue = Math.min(359, edit.hsv.hue + step); edit.hsvActive = true; updateHsvUi(); event.preventDefault(); }
    if (event.key === "ArrowLeft") { edit.hsv.hue = Math.max(0, edit.hsv.hue - step); edit.hsvActive = true; updateHsvUi(); event.preventDefault(); }
  });

  const original = originalGarmentColor(index);
  const initialLower = (edit.color || "").trim().toLowerCase();
  const initialHex = normalizeHexColor(initialLower) || colorHex[initialLower];
  if (initialHex) {
    if (!edit.hsv) edit.hsv = hsvFromHex(initialHex);
  } else if (original) {
    edit.hsv = rgbToHsv(original.red, original.green, original.blue);
    edit.color = "";
  } else if (!edit.hsv) {
    edit.hsv = hsvFromHex(DEFAULT_HEX);
    edit.color = DEFAULT_HEX;
  }
  updateColorUi();

  const materialField = document.createElement("div");
  materialField.className = "field";
  const materialLabel = document.createElement("label");
  materialLabel.className = "field-label field-label-lg";
  materialLabel.textContent = "Material o textura";
  const materialLine = document.createElement("div");
  materialLine.className = "material-line";
  const material = document.createElement("input");
  material.className = "input garment-material";
  material.value = edit.texture;
  material.maxLength = 80;
  material.placeholder = "Ej. denim, silk o none";
  materialLine.append(material);
  materialField.append(materialLabel, materialLine);
  const materialOptions = document.createElement("div");
  materialOptions.className = "material-options";
  for (const value of ["none", ...config.textures.filter((texture) => texture !== "none")]) {
    const option = document.createElement("button");
    option.type = "button";
    option.className = "material-choice";
    option.textContent = value === "none" ? "none" : value;
    option.dataset.value = value;
    option.classList.toggle("selected", edit.texture.toLowerCase() === value);
    option.addEventListener("click", () => {
      edit.texture = value;
      material.value = value;
      materialOptions.querySelectorAll(".material-choice").forEach((button) => button.classList.toggle("selected", button === option));
    });
    materialOptions.append(option);
  }
  material.addEventListener("input", () => {
    edit.texture = material.value;
    materialOptions.querySelectorAll(".material-choice").forEach((button) =>
      button.classList.toggle("selected", button.dataset.value === material.value.trim().toLowerCase()));
  });
  materialField.append(materialOptions);  // los tags van justo debajo de la barra de escritura
  pane.append(materialField);

  return pane;
}
function updateGarmentTabs() {
  const segmentation = state.segmentation;
  const tabList = $("#garmentTabs");
  tabList.replaceChildren();
  if (!segmentation) return;
  const selectedIndices = [...segmentation.selected];
  if (!selectedIndices.includes(state.activeGarmentTab)) {
    state.activeGarmentTab = selectedIndices[0] ?? null;
  }
  $("#garmentEditorHint").textContent = selectedIndices.length
    ? ""
    : "Selecciona una o más prendas para mostrar aquí sus opciones de color y textura.";
  $("#garmentStyleEditor").classList.toggle("no-selection", !selectedIndices.length);
  segmentation.items.forEach((item, index) => {
    if (!segmentation.selected.has(index)) return;
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = "garment-tab";
    tab.classList.toggle("selected", state.activeGarmentTab === index);
    tab.id = `garment-tab-${index}`;
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", String(state.activeGarmentTab === index));
    tab.setAttribute("aria-controls", `garment-pane-${index}`);
    const label = document.createElement("span");
    label.className = "garment-tab-label";
    label.textContent = item.name;
    tab.append(label);
    tab.addEventListener("click", () => setActiveGarmentTab(index));
    tabList.append(tab);
  });
  $("#garmentEditorPanes").querySelectorAll(".garment-editor-pane").forEach((pane, index) =>
    pane.classList.toggle("active", segmentation.selected.has(index)
      && index === state.activeGarmentTab));
}
function setActiveGarmentTab(index) {
  if (!state.segmentation?.selected.has(index)) return;
  state.activeGarmentTab = index;
  updateGarmentTabs();
}
function buildGarmentEditors(config) {
  const segmentation = state.segmentation;
  if (!segmentation) return;
  const panes = $("#garmentEditorPanes");
  panes.replaceChildren();
  state.garmentEdits.clear();
  segmentation.items.forEach((item, index) => {
    state.garmentEdits.set(index, {
      color: "",
      texture: $("#material").value.trim() || "none",
      extra: "",
      hsv: null,
      hsvActive: false,
      points: [],
    });
    panes.append(createGarmentEditor(index, item, config));
  });
  $("#defaultStyle").hidden = true;
  $("#garmentStyleEditor").classList.add("visible");
  updateStylePanel();
  setActiveGarmentTab(0);
}
function toggleGarment(index) {
  if (!state.segmentation) return;
  if (state.segmentation.selected.has(index)) state.segmentation.selected.delete(index);
  else state.segmentation.selected.add(index);
  if (!state.segmentation.selected.has(index) && state.activePlacement?.index === index) {
    state.activePlacement = null;
  }
  drawSegmentationOverlay();
  updateGarmentOptions();
  updateStylePanel();
  const count = state.segmentation.selected.size;
  $("#segmentationStatus").textContent = count
    ? `${count} ${count === 1 ? "prenda seleccionada" : "prendas seleccionadas"}. Solo las prendas seleccionadas se pueden editar; pulsa otra para incluirla o quitarla.`
    : "Ninguna prenda seleccionada. Pulsa una prenda de la imagen o de la lista para poder editarla.";
}

function maskBounds(maskPixels, width, height) {
  let left = width, top = height, right = -1, bottom = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (maskPixels.data[(y * width + x) * 4 + 3] === 0) continue;
      left = Math.min(left, x);
      top = Math.min(top, y);
      right = Math.max(right, x);
      bottom = Math.max(bottom, y);
    }
  }
  return {left, top, width:right - left + 1, height:bottom - top + 1};
}
function garmentAt(clientX, clientY) {
  const segmentation = state.segmentation;
  if (!segmentation) return -1;
  const point = canvasPointAt(clientX, clientY);
  if (!point) return -1;
  const canvas = $("#segmentationOverlay");
  const {x, y} = point;
  for (let index = segmentation.items.length - 1; index >= 0; index -= 1) {
    const pixels = segmentation.items[index].maskPixels;
    if (pixels.data[(y * canvas.width + x) * 4 + 3] > 0) return index;
  }
  return -1;
}
async function detectGarments() {
  clearError();
  if (state.detectingGarments) return;
  if (!state.source) { showError("Elige una imagen antes de detectar las prendas."); return; }
  setImageSelectionLocked(true);
  clearSegmentation("Detectando prendas y afinando sus bordes. Puede tardar un poco…");
  $("#segmentationStatus").textContent += "\nNo se puede cambiar la imagen hasta que termine.";
  const button = $("#detectGarments");
  button.disabled = true;
  button.classList.add("detecting");
  button.setAttribute("aria-busy", "true");
  $(".detect-label").textContent = "Detectando prendas…";
  try {
    const sourceId = state.source;
    const data = await request("/api/segment", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({source_id: sourceId})
    });
    if (sourceId !== state.source) return;
    const items = await Promise.all(data.items.map(async (item) => {
      const maskImage = new Image();
      maskImage.src = item.mask_url;
      await maskImage.decode();
      const maskCanvas = document.createElement("canvas");
      maskCanvas.width = data.width;
      maskCanvas.height = data.height;
      const maskContext = maskCanvas.getContext("2d", {willReadFrequently:true});
      maskContext.drawImage(maskImage, 0, 0, data.width, data.height);
      const maskPixels = maskContext.getImageData(0, 0, data.width, data.height);
      return {...item, maskImage, maskPixels, bounds:maskBounds(maskPixels, data.width, data.height)};
    }));
    const canvas = $("#segmentationOverlay");
    canvas.width = data.width;
    canvas.height = data.height;
    state.colorLayers.clear();
    state.segmentation = {id:data.id, sourceId, items, selected:new Set(), hovered:-1};
    if (items.length === 1) state.segmentation.selected.add(0);
    $("#sourcePreview").classList.add("has-segmentation");
    buildGarmentEditors(window.atelierConfig || {textures:["none"]});
    updateGarmentOptions();
    drawSegmentationOverlay();
    const status = $("#segmentationStatus");
    status.textContent = items.length === 1
      ? "Prenda detectada y seleccionada.  Sólo las prendas seleccionadas podrán ser editadas."
      : `${items.length} prenda(s) detectada(s). Pasa el cursor por encima para identificarlas y haz clic para seleccionarlas; puedes marcar varias.`;
    status.classList.add("visible");
  } catch (error) {
    clearSegmentation(`No se pudieron detectar las prendas: ${error.message}`);
  } finally {
    button.disabled = false;
    button.classList.remove("detecting");
    button.removeAttribute("aria-busy");
    $(".detect-label").textContent = "Detectar prendas";
    setImageSelectionLocked(false);
  }
}
function renderGallery(items) {
  const gallery = $("#imageGallery");
  gallery.innerHTML = "";
  items.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button"; button.className = "thumb"; button.title = item.label; button.dataset.id = item.id;
    const image = document.createElement("img"); image.loading = "lazy"; image.alt = item.label; image.src = item.preview_url;
    button.append(image);
    button.disabled = state.detectingGarments;
    button.addEventListener("click", () => setSource(item.id, item.preview_url));
    gallery.append(button);
  });
  if (!state.source && items.length) setSource(items[0].id, items[0].preview_url);
  else document.querySelectorAll(".thumb").forEach((el) => el.classList.toggle("selected", el.dataset.id === state.source));
  updateGalleryNav();
}
function updateGalleryNav() {
  const prev = $("#prevImages");
  const next = $("#nextImages");
  if (!prev || !next) return;
  prev.disabled = state.detectingGarments || state.galleryIndex <= 0;
  next.disabled = state.detectingGarments || state.galleryIndex >= state.galleryHistory.length - 1;
}
function navigateGallery(step) {
  if (state.detectingGarments) return;
  const target = state.galleryIndex + step;
  if (target < 0 || target >= state.galleryHistory.length) return;
  state.galleryIndex = target;
  renderGallery(state.galleryHistory[target]);
}
async function loadImages() {
  if (state.detectingGarments) return;
  const gallery = $("#imageGallery");
  gallery.innerHTML = Array.from({length:10}, () => '<div class="thumb"></div>').join("");
  try {
    const {items} = await request("/api/images");
    state.galleryHistory = state.galleryHistory.slice(0, state.galleryIndex + 1);
    state.galleryHistory.push(items);
    state.galleryIndex = state.galleryHistory.length - 1;
    renderGallery(items);
  } catch (error) { gallery.innerHTML = '<div class="empty-text">No se pudo cargar el catálogo.</div>'; showError(error.message); updateGalleryNav(); }
}
async function loadHistory() {
  const list = $("#historyList");
  try {
    const {items} = await request("/api/history");
    if (state.historyItem) {
      state.historyItem = items.find((item) => item.filename === state.historyItem.filename) || state.historyItem;
      renderHistorySegmentation();
    }
    list.innerHTML = "";
    if (!items.length) {
      list.innerHTML = '<div class="empty-text">Aún no hay diseños guardados.<br>Cuando generes uno, aparecerá aquí.</div>';
      return;
    }
    items.forEach((item, index) => {
      const openButton = document.createElement("button");
      openButton.type = "button";
      openButton.className = `history-open-button history-item history-tone-${index % 4}`;
      openButton.title = `Ver ${item.label}`;
      const name = document.createElement("span");
      name.className = "history-item-name";
      name.textContent = item.label;
      const arrow = document.createElement("span");
      arrow.className = "history-open";
      arrow.textContent = "↗";
      openButton.append(name, arrow);
      openButton.addEventListener("click", () => displayHistoryResult(item));
      list.append(openButton);
    });
  } catch (error) {
    list.innerHTML = "";
    const message = document.createElement("div");
    message.className = "empty-text";
    message.textContent = `No se pudo cargar el historial: ${error.message}`;
    list.append(message);
  }
}
async function deleteHistoryItem(item, button) {
  if (!window.confirm(`¿Quieres eliminar el diseño «${item.label}»? Esta acción no se puede deshacer.`)) return;
  button.disabled = true;
  try {
    await request(`/api/history/${encodeURIComponent(item.filename)}`, {method:"DELETE"});
    state.historyFilename = null;
    $("#resultsCard").close();
    $("#deleteHistoryButton").hidden = true;
    $("#resultFooter").hidden = true;
    await loadHistory();
  } catch (error) {
    button.disabled = false;
    $("#resultError").textContent = `No se pudo eliminar el diseño: ${error.message}`;
    $("#resultError").classList.add("visible");
  }
}
function populateOptions(config) {
  window.atelierConfig = config;
  $("#colors").innerHTML = config.colors.map((color) => `<option value="${color}">`).join("");
  $("#materialOptions").replaceChildren();
  ["none", ...config.textures.filter((texture) => texture !== "none")].forEach((texture) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "material-choice";
    button.textContent = texture === "none" ? "none" : texture;
    button.dataset.value = texture;
    button.classList.toggle("selected", texture === $("#material").value.trim().toLowerCase());
    button.addEventListener("click", () => {
      $("#material").value = texture;
      updateMaterialChoices();
    });
    $("#materialOptions").append(button);
  });
  $("#swatches").innerHTML = config.colors.filter((color) => colorHex[color]).map((color) => {
    const button = document.createElement("button");
    button.type = "button"; button.className = "swatch"; button.title = color; button.dataset.value = color;
    button.setAttribute("aria-label", color);
    button.style.background = colorHex[color];
    return button.outerHTML;
  }).join("");
  const noColor = document.createElement("button");
  noColor.type = "button";
  noColor.className = "swatch swatch-none";
  noColor.title = "Ninguno";
  noColor.dataset.value = "none";
  noColor.setAttribute("aria-label", "Ninguno: conservar el color original");
  noColor.textContent = "none";
  $("#swatches").append(noColor);
  $("#swatches").querySelectorAll(".swatch").forEach((button) => {
    button.addEventListener("click", () => { $("#color").value = button.dataset.value; updateSwatches(); });
  });
}
function updateMaterialChoices() {
  const value = $("#material").value.trim().toLowerCase();
  $("#materialOptions").querySelectorAll(".material-choice").forEach((button) => {
    button.classList.toggle("selected", button.dataset.value === value);
  });
}
function updateSwatches() {
  $("#swatches").querySelectorAll(".swatch").forEach((button) =>
    button.classList.toggle("selected", button.dataset.value === $("#color").value.trim().toLowerCase()));
}
// ---------- recoloreado HSV ----------
// Misma idea que recolor_hsv de Python: se cambia solo el tono y se reescalan saturación y brillo
// para conservar pliegues y sombras. La vista previa se calcula aquí, en el navegador, sobre una copia
// reducida; el resultado final lo genera el servidor con la imagen a resolución completa.
const HSV_PREVIEW_MAX = 640;   // lado máximo de la vista previa en píxeles
const HSV_MIN_VALUE = 40;      // brillo mínimo para que las sombras no lleguen a negro puro
const MASK_OVERLAY_ALPHA = 175; // opacidad con la que el servidor dibuja las máscaras de prenda
const hsv = { items: [], applySet: new Set(), activeIndex: null, hue: 120, sat: 200, val: 220, width: 0, height: 0,
              base: null, layers: new Map(), frame: 0, applying: false };

function hsvToRgb(hue, saturation, value) {   // hue 0-179 (como OpenCV); saturation y value 0-255
  const s = saturation / 255, v = value / 255, h = hue * 2 / 60;
  const c = v * s, x = c * (1 - Math.abs(h % 2 - 1)), m = v - c;
  const [r, g, b] = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]][Math.floor(h) % 6];
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}
function hsvLutFor(hue, sat) {
  const hueOpenCv = Math.floor(hue / 2) % 180;
  const lut = new Uint8ClampedArray(256 * 3);
  for (let value = 0; value < 256; value++) lut.set(hsvToRgb(hueOpenCv, sat, value), value * 3);
  return lut;
}
function hsvLayer(index) {
  if (hsv.layers.has(index)) return hsv.layers.get(index);
  const canvas = document.createElement("canvas");
  canvas.width = hsv.width;
  canvas.height = hsv.height;
  const context = canvas.getContext("2d", {willReadFrequently: true});
  context.drawImage(state.segmentation.items[index].maskImage, 0, 0, hsv.width, hsv.height);
  const overlay = context.getImageData(0, 0, hsv.width, hsv.height).data;
  const base = hsv.base.data;
  const alpha = new Float32Array(hsv.width * hsv.height);
  const inside = [];
  let maxValue = 1;
  for (let p = 0; p < alpha.length; p++) {
    alpha[p] = Math.min(overlay[p * 4 + 3] / MASK_OVERLAY_ALPHA, 1);
    if (alpha[p] > 0.5) {
      inside.push(p);
      const value = Math.max(base[p * 4], base[p * 4 + 1], base[p * 4 + 2]);
      if (value > maxValue) maxValue = value;
    }
  }
  const layer = {alpha, pixels: Int32Array.from(inside), maxValue};
  hsv.layers.set(index, layer);
  return layer;
}
function getHsvEntry(index) {
  if (!hsv.items) return null;
  if (index === null || index === undefined) return null;
  return hsv.items.find((e) => e.index === index) || null;
}
function renderHsv() {
  hsv.frame = 0;
  if (!hsv.base) {
    return;
  }
  const ctx = $("#hsvCanvas").getContext("2d");
  const hasModified = hsv.items && hsv.items.some((e) => e && e.userModified && e.hue !== null && e.hue !== undefined && e.sat !== null && e.sat !== undefined && e.val !== null && e.val !== undefined);
  if (!hasModified) {
    ctx.putImageData(hsv.base, 0, 0);
    return;
  }
  const out = new ImageData(new Uint8ClampedArray(hsv.base.data), hsv.width, hsv.height);
  const base = hsv.base.data;
  for (const entry of hsv.items) {
    if (!entry.userModified) {
      continue;
    }
    if (entry.hue === null || entry.hue === undefined || entry.sat === null || entry.sat === undefined || entry.val === null || entry.val === undefined) {
      continue;
    }
    const index = entry.index;
    const hue = typeof entry.hue === "number" ? entry.hue : hsv.hue;
    const sat = typeof entry.sat === "number" ? entry.sat : hsv.sat;
    const val = typeof entry.val === "number" ? entry.val : hsv.val;
    const layer = hsvLayer(index);
    const scale = val / (layer.maxValue || 1);
    const lut = hsvLutFor(hue, sat);
    for (let i = 0; i < layer.pixels.length; i++) {
      const p = layer.pixels[i];
      const o = p * 4;
      let value = Math.floor(Math.max(base[o], base[o + 1], base[o + 2]) * scale);
      if (value < HSV_MIN_VALUE) value = HSV_MIN_VALUE;
      const a = layer.alpha[p], k = value * 3;
      out.data[o] = lut[k] * a + base[o] * (1 - a);
      out.data[o + 1] = lut[k + 1] * a + base[o + 1] * (1 - a);
      out.data[o + 2] = lut[k + 2] * a + base[o + 2] * (1 - a);
    }
  }
  ctx.putImageData(out, 0, 0);
}
function scheduleHsv() {
  if (!hsv.frame) hsv.frame = requestAnimationFrame(renderHsv);
}
function updateHsvColorPalette() {
  const palette = $("#hsvColorPalette");
  if (!palette) return;
  palette.replaceChildren();
}
function updateHsvControls() {
  $("#hsvHueValue").textContent = `${Math.round(hsv.hue)}°`;
  $("#hsvSatValue").textContent = hsv.sat;
  $("#hsvValValue").textContent = hsv.val;
  $("#hsvSat").value = hsv.sat;
  $("#hsvVal").value = hsv.val;
  $("#hsvMarker").style.left = `${hsv.hue / 360 * 100}%`;
  $("#hsvStrip").setAttribute("aria-valuenow", Math.round(hsv.hue));
  if (typeof updateHsvColorPalette === "function") updateHsvColorPalette();
}
function setHsvHue(hue) {
  hsv.hue = Math.min(359, Math.max(0, hue));
  if (hsv.sat < 40) hsv.sat = 200;
  if (hsv.val < 60) hsv.val = 220;
  if (hsv.applySet && hsv.applySet.size > 0) {
    for (const idx of hsv.applySet) {
      const entry = getHsvEntry(idx);
      if (entry) {
        entry.hue = hsv.hue;
        entry.sat = hsv.sat;
        entry.val = hsv.val;
        entry.userModified = true;
      }
    }
  } else {
    const entry = getHsvEntry(hsv.activeIndex);
    if (entry) {
      entry.hue = hsv.hue;
      entry.sat = hsv.sat;
      entry.val = hsv.val;
      entry.userModified = true;
    }
  }
  updateHsvControls();
  scheduleHsv();
}
function renderHsvGarments() {
  const list = $("#hsvGarments");
  if (!list) return;
  list.replaceChildren();
  if (!state.segmentation) return;
  if (!hsv.items) hsv.items = [];
  if (!hsv.applySet) hsv.applySet = new Set();
  state.segmentation.items.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "garment-choice";
    const entry = getHsvEntry(index);
    const isApplied = !!entry && hsv.applySet.has(index);
    button.classList.toggle("selected", isApplied);
    button.setAttribute("aria-pressed", String(isApplied));
    button.textContent = garmentLabel(item);
    button.addEventListener("click", () => {
      if (!hsv.items) hsv.items = [];
      if (!hsv.applySet) hsv.applySet = new Set();
      const entryCur = getHsvEntry(index);
      if (!entryCur) {
        hsv.items.push({index, hue: null, sat: null, val: null, userModified: false});
        hsv.applySet.add(index);
        hsv.activeIndex = index;
        hsv.hue = 120; hsv.sat = 200; hsv.val = 220;
        updateHsvControls();
      } else {
        if (hsv.applySet.has(index)) {
          hsv.applySet.delete(index);
          if (hsv.activeIndex === index) {
            const cand = hsv.items.find((e) => hsv.applySet.has(e.index));
            if (cand) {
              hsv.activeIndex = cand.index;
              const n = getHsvEntry(hsv.activeIndex);
              if (n) {
                if (n.hue === null || n.hue === undefined) { hsv.hue = 120; hsv.sat = 200; hsv.val = 220; }
                else { hsv.hue = n.hue; hsv.sat = n.sat; hsv.val = n.val; }
                updateHsvControls();
              }
            } else {
              hsv.activeIndex = null;
              updateHsvControls();
            }
          }
        } else {
          hsv.applySet.add(index);
          hsv.activeIndex = index;
          const n = getHsvEntry(hsv.activeIndex);
          if (n) {
            if (n.hue === null || n.hue === undefined) { hsv.hue = 120; hsv.sat = 200; hsv.val = 220; }
            else { hsv.hue = n.hue; hsv.sat = n.sat; hsv.val = n.val; }
            updateHsvControls();
          }
        }
      }
      renderHsvGarments();
      if (typeof updateHsvColorPalette === "function") updateHsvColorPalette();
      renderHsv();
      scheduleHsv();
    });
    list.append(button);
  });
  if ($("#hsvApply"))       $("#hsvApply").disabled = hsv.applying || !hsv.applySet || hsv.applySet.size === 0;
  if (typeof updateHsvColorPalette === "function") updateHsvColorPalette();
  renderHsv();
}
async function openHsvPicker(initial = []) {
  clearError();
  if (!state.source) { showError("Elige una imagen antes de recolorear con HSV."); return; }
  if (state.detectingGarments) return;
  if (!state.segmentation) {
    await detectGarments();
    if (!state.segmentation) return;
  }
  const {items, selected} = state.segmentation;
  const start = initial.length ? initial : selected.size ? [...selected] : items.map((_, index) => index);
  const maskWidth = items[0].maskImage.naturalWidth, maskHeight = items[0].maskImage.naturalHeight;
  const scale = Math.min(1, HSV_PREVIEW_MAX / Math.max(maskWidth, maskHeight));
  hsv.width = Math.max(1, Math.round(maskWidth * scale));
  hsv.height = Math.max(1, Math.round(maskHeight * scale));
  const canvas = $("#hsvCanvas");
  canvas.width = hsv.width;
  canvas.height = hsv.height;
  const context = canvas.getContext("2d", {willReadFrequently: true});
  context.drawImage($("#sourceImage"), 0, 0, hsv.width, hsv.height);
  hsv.base = context.getImageData(0, 0, hsv.width, hsv.height);
  hsv.layers.clear();
  // No preasignar color: empezar con valores por defecto pero sin mostrar cambio hasta que se ajuste
  if (hsv.frame) { try { cancelAnimationFrame(hsv.frame); } catch(e) {} hsv.frame = 0; }
  hsv.items = start.map((idx) => ({index: idx, hue: null, sat: null, val: null, userModified: false}));
  if (!hsv.applySet) hsv.applySet = new Set();
  hsv.applySet = new Set(start);
  hsv.activeIndex = hsv.items.length ? hsv.items[0].index : null;
  hsv.hue = 120;
  hsv.sat = 200;
  hsv.val = 220;
  hsv.applying = false;
  $("#hsvError").classList.remove("visible");
  $("#hsvApply").textContent = "Aplicar";
  renderHsvGarments();
  updateHsvControls();
  if (typeof updateHsvColorPalette === "function") updateHsvColorPalette();
  const ctx = $("#hsvCanvas").getContext("2d");
  ctx.putImageData(hsv.base, 0, 0);
  $("#hsvDialog").showModal();
}
async     function applyHsv() {
  if (hsv.applying) return;
  if (!hsv.items || hsv.items.length === 0) return;
  if (!hsv.applySet || hsv.applySet.size === 0) {
    if ($("#hsvError")) { $("#hsvError").textContent = "Selecciona al menos una máscara para aplicar."; $("#hsvError").classList.add("visible"); }
    return;
  }
  if (hsv.applySet && hsv.applySet.size > 0) {
    for (const idx of hsv.applySet) {
      const entry = getHsvEntry(idx);
      if (entry) {
        entry.hue = hsv.hue;
        entry.sat = hsv.sat;
        entry.val = hsv.val;
        entry.userModified = true;
      }
    }
  } else {
    const current = getHsvEntry(hsv.activeIndex);
    if (current) {
      current.hue = hsv.hue;
      current.sat = hsv.sat;
      current.val = hsv.val;
      current.userModified = true;
    }
  }
  hsv.applying = true;
  $("#hsvApply").disabled = true;
  $("#hsvCancel").disabled = true;
  $("#hsvApply").textContent = "Aplicando…";
  $("#hsvError").classList.remove("visible");
  try {
    const toApply = hsv.items.filter((i) => hsv.applySet.has(i.index) && i.hue !== null && i.hue !== undefined && i.sat !== null && i.sat !== undefined && i.val !== null && i.val !== undefined && i.userModified);
    if (toApply.length === 0) {
      throw new Error("Mueve tono/saturación/brillo para cambiar alguna máscara antes de aplicar.");
    }
    const done = await request("/api/recolor-hsv", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        source_id: state.source,
        segmentation_id: state.segmentation.id,
        garments: toApply.map((i) => ({index: i.index, hue: Math.round(i.hue), saturation: i.sat, value: i.val})),
      }),
    });
    const job = await request(`/api/jobs/${done.id}`);
    $("#hsvDialog").close();
    loadHistory();
    displayResult(job, done.id);
  } catch (error) {
    let msg = "Error al aplicar";
    if (typeof error === "string") msg = error;
    else if (error && typeof error === "object") {
      msg = error.detail || error.message || (Array.isArray(error) ? error.map(String).join(" | ") : JSON.stringify(error));
    } else msg = String(error);
    if ($("#hsvError")) { $("#hsvError").textContent = msg; $("#hsvError").classList.add("visible"); }
  } finally {
    hsv.applying = false;
    $("#hsvCancel").disabled = false;
    $("#hsvApply").textContent = "Aplicar";
    renderHsvGarments();
  }
}
// ---------- selector HSV incrustado (sustituye a los círculos de color) ----------
function hsvDegreesToRgb(hue, saturation, value) {
  return hsvToRgb(Math.floor(hue / 2) % 180, saturation, value);
}
function normalizeHexColor(value) {
  let text = String(value || "").trim();
  if (!text) return null;
  if (!text.startsWith("#")) text = "#" + text;
  const digits = text.slice(1);
  if (/^[0-9a-f]{3}$/i.test(digits)) {
    return "#" + digits.split("").map((character) => character + character).join("").toLowerCase();
  }
  if (/^[0-9a-f]{6}$/i.test(digits)) return "#" + digits.toLowerCase();
  return null;
}
function rgbToHex(red, green, blue) {
  return "#" + [red, green, blue]
    .map((channel) => Math.max(0, Math.min(255, Math.round(channel))).toString(16).padStart(2, "0"))
    .join("");
}
function rgbToHsv(red, green, blue) {
  const r = red / 255, g = green / 255, b = blue / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const delta = max - min;
  let hue = 0;
  if (delta) {
    if (max === r) hue = ((g - b) / delta) % 6;
    else if (max === g) hue = (b - r) / delta + 2;
    else hue = (r - g) / delta + 4;
    hue *= 60;
    if (hue < 0) hue += 360;
  }
  return {
    hue: Math.round(hue) % 360,
    sat: Math.round((max ? delta / max : 0) * 255),
    val: Math.round(max * 255),
  };
}
function hsvFromHex(hex) {
  let value = String(hex || "").replace("#", "");
  if (value.length === 3) value = value.split("").map((character) => character + character).join("");
  const number = parseInt(value, 16);
  if (!Number.isFinite(number) || value.length !== 6) return {hue: 12, sat: 200, val: 220};
  return rgbToHsv((number >> 16) & 255, (number >> 8) & 255, number & 255);
}
function drawGarmentHsvPreview(canvas, index, edit) {
  if (!canvas || !edit.hsv || !state.segmentation) return;
  const item = state.segmentation.items[index];
  const source = $("#sourceImage");
  if (!item || !source || !source.naturalWidth) return;
  const maskWidth = item.maskImage.naturalWidth || item.maskImage.width;
  const maskHeight = item.maskImage.naturalHeight || item.maskImage.height;
  if (!maskWidth || !maskHeight) return;
  const scale = Math.min(1, HSV_PREVIEW_MAX / Math.max(maskWidth, maskHeight));
  const width = Math.max(1, Math.round(maskWidth * scale));
  const height = Math.max(1, Math.round(maskHeight * scale));
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", {willReadFrequently: true});
  context.drawImage(source, 0, 0, width, height);
  const base = context.getImageData(0, 0, width, height);
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskContext = maskCanvas.getContext("2d", {willReadFrequently: true});
  maskContext.drawImage(item.maskImage, 0, 0, width, height);
  const overlay = maskContext.getImageData(0, 0, width, height).data;
  const alpha = new Float32Array(width * height);
  const inside = [];
  let maxValue = 1;
  for (let p = 0; p < alpha.length; p += 1) {
    alpha[p] = Math.min(overlay[p * 4 + 3] / MASK_OVERLAY_ALPHA, 1);
    if (alpha[p] > 0.5) {
      inside.push(p);
      const value = Math.max(base.data[p * 4], base.data[p * 4 + 1], base.data[p * 4 + 2]);
      if (value > maxValue) maxValue = value;
    }
  }
  const out = new ImageData(new Uint8ClampedArray(base.data), width, height);
  for (let p = 0; p < alpha.length; p += 1) {
    out.data[p * 4 + 3] = Math.round(alpha[p] * 255);
  }
  const lut = hsvLutFor(edit.hsv.hue, edit.hsv.sat);
  const valueScale = edit.hsv.val / (maxValue || 1);
  for (const p of inside) {
    const offset = p * 4;
    let value = Math.floor(Math.max(base.data[offset], base.data[offset + 1], base.data[offset + 2]) * valueScale);
    if (value < HSV_MIN_VALUE) value = HSV_MIN_VALUE;
    const a = alpha[p], k = value * 3;
    out.data[offset] = lut[k] * a + base.data[offset] * (1 - a);
    out.data[offset + 1] = lut[k + 1] * a + base.data[offset + 1] * (1 - a);
    out.data[offset + 2] = lut[k + 2] * a + base.data[offset + 2] * (1 - a);
  }
  context.putImageData(out, 0, 0);
}
function buildGarmentColorLayer(index, edit) {
  if (!edit.hsv || !edit.hsvActive || !state.segmentation) return null;
  const item = state.segmentation.items[index];
  if (!item) return null;
  const maskWidth = item.maskImage.naturalWidth || item.maskImage.width;
  const maskHeight = item.maskImage.naturalHeight || item.maskImage.height;
  if (!maskWidth || !maskHeight) return null;
  const scale = Math.min(1, HSV_PREVIEW_MAX / Math.max(maskWidth, maskHeight));
  const width = Math.max(1, Math.round(maskWidth * scale));
  const height = Math.max(1, Math.round(maskHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  drawGarmentHsvPreview(canvas, index, edit);
  return canvas;
}
function refreshGarmentColorLayer(index, edit) {
  const layer = buildGarmentColorLayer(index, edit);
  if (layer) state.colorLayers.set(index, layer);
  else state.colorLayers.delete(index);
  drawSegmentationOverlay();
}
function originalGarmentColor(index) {
  if (!state.segmentation) return null;
  const item = state.segmentation.items[index];
  const source = $("#sourceImage");
  if (!item || !source || !source.naturalWidth) return null;
  const maskWidth = item.maskImage.naturalWidth || item.maskImage.width;
  const maskHeight = item.maskImage.naturalHeight || item.maskImage.height;
  if (!maskWidth || !maskHeight) return null;
  const scale = Math.min(1, 160 / Math.max(maskWidth, maskHeight));
  const width = Math.max(1, Math.round(maskWidth * scale));
  const height = Math.max(1, Math.round(maskHeight * scale));
  const baseCanvas = document.createElement("canvas");
  baseCanvas.width = width;
  baseCanvas.height = height;
  const baseContext = baseCanvas.getContext("2d", {willReadFrequently: true});
  baseContext.drawImage(source, 0, 0, width, height);
  const base = baseContext.getImageData(0, 0, width, height).data;
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskContext = maskCanvas.getContext("2d", {willReadFrequently: true});
  maskContext.drawImage(item.maskImage, 0, 0, width, height);
  const overlay = maskContext.getImageData(0, 0, width, height).data;
  let red = 0, green = 0, blue = 0, count = 0;
  for (let p = 0; p < width * height; p += 1) {
    if (overlay[p * 4 + 3] > 127) {
      red += base[p * 4];
      green += base[p * 4 + 1];
      blue += base[p * 4 + 2];
      count += 1;
    }
  }
  if (!count) return null;
  return {red: red / count, green: green / count, blue: blue / count};
}
async function uploadImage(file) {
  clearError();
  if (state.detectingGarments) return;
  if (!file) return;
  if (file.size > 20 * 1024 * 1024) { showError("La imagen supera el límite de 20 MB."); return; }
  try {
    const response = await fetch(`/api/upload?filename=${encodeURIComponent(file.name)}`, {
      method: "POST", headers: {"Content-Type": file.type || "application/octet-stream"}, body: file
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "No se pudo cargar la imagen.");
    setSource(data.id, data.preview_url);
  } catch (error) { showError(error.message); }
}
function displayResult(job, jobId) {
  const cacheBust = `?v=${Date.now()}`;
  state.historyItem = null;
  state.historyFilename = job.history_filename || null;
  state.lastResultGarment = job.garment || "";
  renderHistorySegmentation();
  $("#deleteHistoryButton").hidden = true;
  $("#resultFooter").hidden = true;
  $("#resultHeading").textContent = "Tu resultado";
  document.querySelector(".result-showcase").classList.remove("history-showcase");
  document.querySelectorAll(".result-alternates .result-image").forEach((figure) => figure.classList.remove("hidden"));
  $("#resultOriginal").src = `/api/jobs/${jobId}/image/original${cacheBust}`;
  $("#resultMask").src = `/api/jobs/${jobId}/image/mask${cacheBust}`;
  $("#resultEdited").src = `/api/jobs/${jobId}/image/edited${cacheBust}`;
  $("#resultFinal").src = `/api/jobs/${jobId}/image/final${cacheBust}`;
  $("#downloadButton").href = `/api/jobs/${jobId}/image/final?download=1`;
  $("#resultFilename").value = downloadName(job.label || "atelier");
  state.savedName = $("#resultFilename").value;
  $("#downloadButton").download = `${$("#resultFilename").value}.png`;
  setDownloadFilename();
  $("#resultMeta").textContent = `${job.garment} · ${job.label || "diseño listo"}`;
  $("#resultError").classList.remove("visible");
  if (!$("#resultsCard").open) $("#resultsCard").showModal();
}
function downloadName(value) {
  return value.replace(/\.png$/i, "").replace(/[<>:"/\\|?*\x00-\x1f]/g, "_").trim().replace(/[. ]+$/g, "").slice(0, 100) || "atelier";
}
function displayHistoryResult(item) {
  const cacheBust = `?v=${Date.now()}`;
  const imageUrl = `/api/history/${encodeURIComponent(item.filename)}`;
  state.historyItem = item;
  state.historyFilename = item.filename;
  $("#deleteHistoryButton").hidden = false;
  $("#deleteHistoryButton").disabled = false;
  $("#resultFooter").hidden = false;
  $("#resultHeading").textContent = "Diseño anterior";
  document.querySelector(".result-showcase").classList.add("history-showcase");
  document.querySelectorAll(".result-alternates .result-image").forEach((figure) => figure.classList.add("hidden"));
  renderHistorySegmentation();
  $("#resultFinal").src = `${imageUrl}${cacheBust}`;
  $("#resultFilename").value = item.label;
  state.savedName = item.label;
  $("#downloadButton").href = `${imageUrl}?download=1`;
  $("#downloadButton").download = `${$("#resultFilename").value}.png`;
  setDownloadFilename();
  $("#resultMeta").textContent = item.label;
  $("#resultError").classList.remove("visible");
  if (!$("#resultsCard").open) $("#resultsCard").showModal();
}
function renderHistorySegmentation() {
  const section = $("#historySegmentation");
  const gallery = $("#historySegmentationImages");
  const empty = $("#historySegmentationEmpty");
  gallery.replaceChildren();
  empty.hidden = true;
  section.hidden = !state.devMode || !state.historyItem;
  if (section.hidden) return;
  const assets = state.historyItem.assets;
  if (!assets) {
    empty.textContent = "Esta generación es anterior al guardado de máscaras y no tiene datos de segmentación disponibles.";
    empty.hidden = false;
    return;
  }
  const cacheBust = `?v=${Date.now()}`;
  const images = [
    {url: assets.mask_url, label: "Máscara aplicada"},
    {url: assets.segmentation_url, label: "Segmentación completa"},
    ...assets.garment_masks.map((garment) => ({url: garment.url, label: garment.name})),
  ];
  images.forEach(({url, label}) => {
    const figure = document.createElement("figure");
    figure.className = "result-image";
    const image = document.createElement("img");
    image.alt = label;
    image.src = `${url}${cacheBust}`;
    const caption = document.createElement("figcaption");
    caption.textContent = label;
    figure.append(image, caption);
    gallery.append(figure);
  });
}
function setDownloadFilename() {
  const name = `${downloadName($("#resultFilename").value)}.png`;
  $("#downloadButton").download = name;
  const url = new URL($("#downloadButton").href, location.href);
  url.searchParams.set("download", "1");
  url.searchParams.set("name", name);
  $("#downloadButton").href = url.pathname + url.search;
}
async function saveHistoryName() {
  if (!state.historyFilename) return;
  const filename = state.historyFilename;
  $("#resultError").classList.remove("visible");
  try {
    const result = await request(
      `/api/history/${encodeURIComponent(state.historyFilename)}/rename`,
      {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({name: $("#resultFilename").value}),
      },
    );
    state.historyFilename = result.filename;
    if (state.historyItem) {
      state.historyItem.filename = result.filename;
      if (result.label) state.historyItem.label = result.label;
      $("#resultFinal").src = `/api/history/${encodeURIComponent(result.filename)}?v=${Date.now()}`;
      $("#downloadButton").href = `/api/history/${encodeURIComponent(result.filename)}?download=1`;
    }
    if (result.label) {
      state.savedName = result.label;
      $("#resultFilename").value = result.label;
      setDownloadFilename();
      $("#resultMeta").textContent = state.historyItem
        ? result.label
        : `${state.lastResultGarment} · ${result.label}`;
    }
    await loadHistory();
  } catch (error) {
    // El diálogo ya está cerrado: se restaura el nombre guardado y se reabre
    // una vez para mostrar el error (al volver a cerrar ya no hay cambios).
    $("#resultFilename").value = state.savedName;
    setDownloadFilename();
    $("#resultError").textContent = `No se pudo guardar el nombre: ${error.message}`;
    $("#resultError").classList.add("visible");
    if (state.historyFilename === filename && !$("#resultsCard").open) $("#resultsCard").showModal();
  }
}
$("#resultFilename").addEventListener("input", () => {
  setDownloadFilename();
});
function updateQueueStatus(job = null) {
  const button = $("#generateButton");
  const split = $("#generateSplit");
  const queueAnother = $("#queueAnotherButton");
  const generatingLabel = $("#generatingLabel");
  const cancel = $("#cancelGeneration");
  const status = $("#queueStatus");
  const statusText = $("#queueStatusText");
  const spinner = $("#queueSpinner");
  const count = state.pendingJobs.length;
  const currentJob = job || state.pendingJobs[0];
  const generating = state.submittingJob || count > 0;
  button.hidden = generating;
  split.hidden = !generating;
  button.disabled = state.submittingJob;
  button.textContent = "✦ Crear mi diseño";
  queueAnother.disabled = state.submittingJob;
  generatingLabel.textContent = state.submittingJob ? "Añadiendo a la cola…" : "Generando…";
  cancel.hidden = count === 0;
  cancel.disabled = state.cancelling;
  cancel.textContent = state.cancelling ? "Cancelando…" : "Cancelar la generación en curso";
  spinner.hidden = currentJob?.status !== "running";
  if (state.submittingJob) {
    statusText.textContent = "Enviando tu solicitud…";
    status.hidden = false;
  } else if (!count) {
    statusText.textContent = "";
    status.hidden = true;
  } else if (currentJob?.status === "queued") {
    statusText.textContent = `Tu cambio está en la cola, puesto ${currentJob.queue_position || count}${count > 1 ? ` · ${count} trabajos pendientes` : ""}.`;
    status.hidden = false;
  } else {
    const pending = count - 1;
    statusText.textContent = pending > 0
      ? `${pending} ${pending === 1 ? "trabajo pendiente" : "trabajos pendientes"}.`
      : "";
    status.hidden = pending === 0;
  }
  updateStylePanel();
}
function waitForResultsClose() {
  const dialog = $("#resultsCard");
  if (!dialog.open) return Promise.resolve();
  return new Promise((resolve) => dialog.addEventListener("close", resolve, {once: true}));
}
function hasWorkToLose() {
  const currentResultOpen = $("#resultsCard").open && !state.historyItem;
  return Boolean(state.submittingJob || state.activeJob || state.pendingJobs.length || currentResultOpen);
}
function showReloadWarning() {
  const dialog = $("#reloadWarning");
  if (!dialog.open && !state.allowReload && hasWorkToLose()) dialog.showModal();
}
function requestReloadConfirmation(event) {
  if (state.allowReload || !hasWorkToLose()) return;
  event.preventDefault();
  showReloadWarning();
}
window.addEventListener("beforeunload", (event) => {
  if (state.allowReload || !hasWorkToLose()) return;
  event.preventDefault();
  event.returnValue = "";
});
document.addEventListener("keydown", (event) => {
  const key = (event.key || "").toLowerCase();
  if (event.key === "F5" || ((event.ctrlKey || event.metaKey) && key === "r")) {
    requestReloadConfirmation(event);
  }
});
$("#reloadWarning").addEventListener("cancel", (event) => event.preventDefault());
$("#stayOnPage").addEventListener("click", () => $("#reloadWarning").close());
$("#leavePage").addEventListener("click", () => {
  state.allowReload = true;
  $("#reloadWarning").close();
  location.reload();
});
async function pollQueue() {
  if (state.pollingJobs) return;
  state.pollingJobs = true;
  try {
    while (state.pendingJobs.length) {
      const current = state.pendingJobs[0];
      state.activeJob = current.id;
      let job;
      while (state.pendingJobs[0]?.id === current.id) {
        try {
          job = await request(`/api/jobs/${current.id}`);
        } catch (error) {
          showError(`No se pudo consultar el estado del trabajo: ${error.message}`);
          await new Promise((resolve) => setTimeout(resolve, 2000));
          continue;
        }
        current.status = job.status;
        current.queue_position = job.queue_position;
        updateQueueStatus(job);
        if (job.status === "completed" || job.status === "failed") break;
        await new Promise((resolve) => setTimeout(resolve, 1300));
      }
      if (state.pendingJobs[0]?.id !== current.id) continue;
      state.pendingJobs.shift();
      state.activeJob = null;
      updateQueueStatus();
      if (job.status === "completed") {
        loadHistory();
        displayResult(job, current.id);
        if (state.pendingJobs.length || state.submittingJob) await waitForResultsClose();
      } else if (job.status !== "cancelled") {
        showError(job.error || "No se pudo generar el diseño.");
      }
    }
  } finally {
    state.activeJob = null;
    state.pollingJobs = false;
    updateQueueStatus();
  }
}
// Negro y blanco → InstructPix2Pix. Cualquier otro color de la paleta, y los colores
// personalizados (sliders HSV o código HEX) → recolor_hsv.
const PIX2PIX_COLORS = new Set(["black", "white"]);
function colorRoute(edit) {
  const name = (edit.color || "").trim().toLowerCase();
  if (PIX2PIX_COLORS.has(name) && !edit.hsvActive) {
    return {color: name, hsv: null};
  }
  if (edit.hsv && edit.hsvActive) return {color: "", hsv: edit.hsv};
  const hexValue = colorHex[name] || normalizeHexColor(name);   // p. ej. el color por defecto si no se pudo leer el original
  if (hexValue) return {color: "", hsv: hsvFromHex(hexValue)};
  return {color: "", hsv: null};
}
async function generate() {
  clearError();
  if (!state.source) { showError("Elige una imagen antes de crear el diseño."); return; }
  const colorValue = $("#color").value.trim();
  const color = colorValue.toLowerCase() === "none" ? "" : colorValue;
  const material = $("#material").value.trim();
  const extra = "";
  if (state.segmentation && state.segmentation.selected.size === 0) {
    showError("Selecciona al menos una de las prendas detectadas.");
    return;
  }
  const garmentEdits = state.segmentation
    ? [...state.segmentation.selected].map((index) => {
        const edit = state.garmentEdits.get(index);
        const route = colorRoute(edit);
        return {
          index,
          color: route.color,
          texture: edit.texture.trim() || "none",
          extra: "",
          hsv: route.hsv ? {
            hue: Math.round(route.hsv.hue) % 360,
            saturation: Math.round(route.hsv.sat),
            value: Math.round(route.hsv.val),
          } : null,
          placements: edit.points,
        };
      })
    : null;
  if (garmentEdits && garmentEdits.some((edit) =>
    !edit.color && !edit.hsv && edit.texture.toLowerCase() === "none" && !edit.placements.length)) {
    showError("Cada prenda incluida necesita al menos un cambio de color o de material.");
    return;
  }
  if (!state.segmentation && !color && (!material || material.toLowerCase() === "none") && !extra) {
    showError("Indica un color o un material antes de generar.");
    return;
  }
  if (state.submittingJob) return;
  state.submittingJob = true;
  updateQueueStatus();
  try {
    const job = await request("/api/generate", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        source_id: state.source,
        color,
        texture: material || "none",
        extra,
        segmentation_id: state.segmentation?.id || null,
        garment_indices: state.segmentation ? [...state.segmentation.selected] : null,
        garment_edits: garmentEdits
      })
    });
    state.pendingJobs.push({id: job.id, status: "queued", queue_position: job.queue_position});
    updateQueueStatus();
    pollQueue();
  } catch (error) {
    showError(error.message);
  } finally {
    state.submittingJob = false;
    updateQueueStatus();
  }
}
async function cancelGeneration() {
  if (state.cancelling || !state.pendingJobs.length) return;
  const target = state.pendingJobs.shift();
  state.activeJob = state.pendingJobs[0]?.id || null;
  state.cancelling = true;
  updateQueueStatus();
  try {
    await request(`/api/jobs/${encodeURIComponent(target.id)}/cancel`, {method:"POST"});
  } catch (error) {
    showError(`No se pudo cancelar la generación en curso: ${error.message}`);
  } finally {
    state.cancelling = false;
    updateQueueStatus();
  }
}
$("#refreshImages").addEventListener("click", loadImages);
$("#prevImages").addEventListener("click", () => navigateGallery(-1));
$("#nextImages").addEventListener("click", () => navigateGallery(1));
$("#refreshHistory").addEventListener("click", loadHistory);
$("#uploadButton").addEventListener("click", () => $("#fileInput").click());
$("#fileInput").addEventListener("change", (event) => uploadImage(event.target.files[0]));
$("#detectGarments").addEventListener("click", detectGarments);
$("#settingsButton").addEventListener("click", () => {
  const menu = $("#settingsMenu");
  menu.hidden = !menu.hidden;
  $("#settingsButton").setAttribute("aria-expanded", String(!menu.hidden));
});
$("#devModeToggle").addEventListener("change", (event) => setDevMode(event.target.checked));
document.querySelectorAll('input[name="theme"]').forEach((input) =>
  input.addEventListener("change", (event) => {
    if (event.target.checked) setTheme(event.target.value);
  }));
$("#devModeIndicator").addEventListener("click", () => setDevMode(false));
// el menú se cierra al hacer clic fuera de él o al pulsar Escape
document.addEventListener("click", (event) => { if (!event.target.closest(".settings")) closeSettings(); });
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeSettings(); });
$("#sourcePreview").addEventListener("pointermove", (event) => {
  if (!state.segmentation) return;
  const index = garmentAt(event.clientX, event.clientY);
  if (index === state.segmentation.hovered) return;
  state.segmentation.hovered = index;
  drawSegmentationOverlay();
  $("#segmentationStatus").textContent = state.activePlacement
    ? `Pulsa dentro de la máscara de ${state.segmentation.items[state.activePlacement.index].name} para añadir ${state.activePlacement.type === "pocket" ? "el bolsillo" : "el botón"}.`
    : index >= 0
    ? (state.devMode
        ? `${state.segmentation.items[index].name} · ${Math.round(state.segmentation.items[index].score * 100)}% de confianza`
        : state.segmentation.items[index].name)
    : "Pasa el cursor por encima para identificar una prenda; haz clic para seleccionarla.";
});
$("#sourcePreview").addEventListener("pointerleave", () => {
  if (!state.segmentation) return;
  state.segmentation.hovered = -1;
  drawSegmentationOverlay();
});
$("#sourcePreview").addEventListener("click", (event) => {
  if (!state.segmentation) return;
  if (state.activePlacement) {
    const index = garmentAt(event.clientX, event.clientY);
    if (index !== state.activePlacement.index) {
      $("#segmentationStatus").textContent = `Pulsa dentro de la prenda seleccionada (${state.segmentation.items[state.activePlacement.index].name}).`;
      return;
    }
    const position = canvasPointAt(event.clientX, event.clientY);
    if (!position) return;
    const item = state.segmentation.items[index];
    const edit = state.garmentEdits.get(index);
    if (edit.points.length >= 12) {
      $("#segmentationStatus").textContent = "Has alcanzado el máximo de 12 puntos para esta prenda.";
      return;
    }
    edit.points.push({
      type: state.activePlacement.type,
      x: Math.round((position.x - item.bounds.left) / Math.max(item.bounds.width - 1, 1) * 100),
      y: Math.round((position.y - item.bounds.top) / Math.max(item.bounds.height - 1, 1) * 100),
    });
    renderPlacementPoints(index);
    drawSegmentationOverlay();
    $("#segmentationStatus").textContent = "Ubicación añadida. Puedes seguir pulsando para colocar más o cancelar.";
    return;
  }
  const index = garmentAt(event.clientX, event.clientY);
  if (index >= 0) toggleGarment(index);
});
$("#material").addEventListener("input", updateMaterialChoices);
$("#color").addEventListener("input", updateSwatches);
$("#generateButton").addEventListener("click", generate);
$("#queueAnotherButton").addEventListener("click", generate);
$("#cancelGeneration").addEventListener("click", cancelGeneration);
$("#closeResults").addEventListener("click", () => $("#resultsCard").close());
$("#deleteHistoryButton").addEventListener("click", () => {
  if (state.historyItem) deleteHistoryItem(state.historyItem, $("#deleteHistoryButton"));
});
$("#resultsCard").addEventListener("click", (event) => {
  if (event.target === $("#resultsCard")) $("#resultsCard").close();
});
$("#resultsCard").addEventListener("close", () => {
  if (!state.historyFilename) return;
  if (downloadName($("#resultFilename").value) === state.savedName) return;
  saveHistoryName();
});
["resultOriginal", "resultMask", "resultEdited", "resultFinal"].forEach((id) => {
  $(`#${id}`).addEventListener("error", () => {
    $("#resultError").textContent = "No se pudo cargar una de las imágenes. Comprueba que el servidor siga abierto y vuelve a generar el diseño.";
    $("#resultError").classList.add("visible");
  });
});
(async function init() {
  setTheme(readTheme());
  setDevMode(readDevMode());
  try { populateOptions(await request("/api/config")); }
  catch (error) { showError(error.message); }
  await Promise.all([loadImages(), loadHistory()]);
})();
