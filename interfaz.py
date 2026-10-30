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
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

try:
    from winotify import Notification  # solo funciona en Windows
except ImportError:
    Notification = None  # sin winotify la interfaz va igual, solo que sin notificación

from src import config as C


logger = logging.getLogger("interfaz")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
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


HTML = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#f7f7f4">
  <title>Atelier — color para vestir</title>
  <style>
    :root {
      color-scheme: light;
      --paper: #f7f7f4; --white: #fff; --ink: #20211e; --muted: #777972;
      --line: #e9e9e3; --accent: #4a7728; --accent-dark: #24541c;
      --accent-mid: #89c34b; --accent-soft: #becc66;
      --green: #4a7728; --green-dark: #24541c; --lime: #89c34b; --lime-soft: #becc66;
      --plum: #732670; --plum-mid: #862d67; --plum-dark: #5c1f59; --lavender: #c8afe4;
      --plum-surface: #f2ebf8;
      --shadow: 0 12px 36px #29271e0b;
      --surface-raised: #fff; --surface-soft: #fbfbf8; --surface-muted: #f4f3ef;
      --surface-hover: #f4f7ec; --surface-accent: #ecf2df; --surface-preview: #f1f4ea;
      --text-strong: #454741; --text-default: #565851; --text-subtle: #62645d;
      --text-muted: #94968f; --text-label: #a1a39d; --line-strong: #e5e5df;
      --line-soft: #dedfd7;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    :root[data-theme="dark"] {
      color-scheme: dark;
      --paper: #171815; --white: #242521; --ink: #eef2e6; --muted: #a6a79f;
      --line: #3a3b36; --accent: #89c34b; --accent-dark: #becc66;
      --accent-mid: #becc66; --accent-soft: #4a7728;
      --green: #89c34b; --green-dark: #becc66; --lime: #becc66; --lime-soft: #89c34b;
      --plum: #c8afe4; --plum-mid: #c8afe4; --plum-dark: #c8afe4; --lavender: #c8afe4;
      --plum-surface: #35213a;
      --shadow: 0 12px 36px #0005;
      --surface-raised: #242521; --surface-soft: #20211e; --surface-muted: #2c2d29;
      --surface-hover: #343c2c; --surface-accent: #26331d; --surface-preview: #262d20;
      --text-strong: #d0d1ca; --text-default: #c7c8c1; --text-subtle: #bfc0b8;
      --text-muted: #a6a79f; --text-label: #92938c; --line-strong: #41423c;
      --line-soft: #484a43;
    }
    :root[data-theme="dark"] .brand-mark,
    :root[data-theme="dark"] .thumb.selected::after,
    :root[data-theme="dark"] .generate,
    :root[data-theme="dark"] .generate:hover:not(:disabled),
    :root[data-theme="dark"] .hsv-apply,
    :root[data-theme="dark"] .hsv-apply:hover,
    :root[data-theme="dark"] .result-download,
    :root[data-theme="dark"] .result-download:hover { color: #171815; }
    * { box-sizing: border-box; }
    html { -webkit-text-size-adjust: 100%; }
    body { margin: 0; color: var(--ink); background: var(--paper); }
    button, input, select { font: inherit; }
    button { cursor: pointer; }
    .shell { min-height: 100vh; min-height: 100dvh; display: grid;
      grid-template-columns: clamp(220px, 17vw, 248px) minmax(0, 1fr); }
    aside { position: sticky; top: 0; height: 100vh; height: 100dvh; padding: 27px 20px; background: var(--surface-raised);
      border-right: 1px solid var(--line); display: flex; flex-direction: column; }
    .brand { display: flex; align-items: center; gap: 14px; padding: 0 8px 27px; }
    .brand-mark { display: grid; place-items: center; width: 56px; height: 56px; border-radius: 18px;
      color: white; background: var(--plum); font-family: Georgia, serif; font-size: 33px; }
    .brand-name { font-family: Georgia, "Times New Roman", serif; font-size: 32px; letter-spacing: -1px; }
    .brand-sub { margin-top: 1px; color: var(--lime); font-size: 12px; letter-spacing: 2px; text-transform: uppercase; }
    .history-nav { display: flex; flex: 1; flex-direction: column; min-height: 0; margin-top: 28px; }
    .history-nav-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 0 0 9px; }
    .history-nav-title { padding-left: 10px; color: var(--plum-dark); font-size: 10px; font-weight: 700; letter-spacing: 1.1px; text-transform: uppercase; }
    .history-refresh { display: grid; place-items: center; width: 27px; height: 27px; border: 1px solid var(--line);
      border-radius: 8px; background: var(--surface-raised); color: var(--text-subtle); font-size: 12px; }
    .history-refresh:hover { border-color: var(--accent-mid); color: var(--accent-dark); }
    .history-list { display: grid; flex: 1; align-content: start; gap: 7px; min-height: 0; overflow-y: auto; padding: 1px 4px 2px 0; }
    .history-item { width: 100%; }
    .history-open-button { display: flex; align-items: center; justify-content: space-between; gap: 8px; width: 100%;
      min-width: 0; padding: 9px 10px; border: 1px solid var(--line); border-radius: 9px;
      background: var(--surface-raised); color: var(--text-strong); text-align: left; }
    .history-open-button:hover { border-color: var(--accent-mid); background: var(--surface-hover); color: var(--accent-dark); }
    .history-item-name { display: block; min-width: 0; overflow: hidden; color: var(--plum); font-size: 11px; font-weight: 600;
      text-overflow: ellipsis; white-space: nowrap; }
    .history-open { flex: 0 0 auto; color: var(--accent); font-size: 14px; }
    :root:not([data-theme="dark"]) .history-open { color: var(--green-dark); }
    main { min-width: 0; max-width: 1700px; width: 100%; margin: 0 auto; padding: 20px clamp(20px, 4vw, 58px) 60px clamp(12px, 1.5vw, 24px); }
    .dev-mode-indicator { display: none; padding: 8px 11px; border: 1px solid var(--lavender);
      border-radius: 99px; background: var(--surface-accent); color: var(--accent-dark); font-size: 10px; font-weight: 650; cursor: pointer; }
    .dev-mode .dev-mode-indicator { display: flex; }
    .dev-mode-indicator:hover { border-color: var(--accent-mid); background: var(--surface-accent); }
    .settings { position: fixed; top: 18px; right: 12px; z-index: 40; display: flex; align-items: center; gap: 8px; }
    .settings-button { display: grid; place-items: center; width: 32px; height: 32px; padding: 0; border: 1px solid var(--line);
      border-radius: 50%; background: var(--surface-raised); color: var(--text-subtle); cursor: pointer; transition: border-color .2s, color .2s; }
    .settings-button svg { transition: transform .3s; }
    .settings-button:hover, .settings-button[aria-expanded="true"] { border-color: var(--accent-mid); color: var(--accent-dark); }
    .settings-button[aria-expanded="true"] svg { transform: rotate(60deg); }
    .settings-menu { position: absolute; top: calc(100% + 8px); right: 0; z-index: 30; width: min(270px, calc(100vw - 24px));
      padding: 14px; border: 1px solid var(--line); border-radius: 14px; background: var(--surface-raised); color: var(--ink);
      box-shadow: 0 16px 40px #29271e22; }
    .settings-title { margin-bottom: 10px; font-size: 12px; font-weight: 700; }
    .settings-themes { display: flex; gap: 8px; margin-bottom: 14px; }
    .settings-theme { display: flex; flex: 1; align-items: center; gap: 7px; padding: 9px;
      border: 1px solid var(--line); border-radius: 9px; cursor: pointer; font-size: 11px; }
    .settings-theme input { margin: 0; accent-color: var(--accent); }
    .settings-option { display: flex; align-items: flex-start; gap: 10px; cursor: pointer; }
    .settings-option input { margin-top: 2px; accent-color: var(--accent); }
    .settings-option b { display: block; font-size: 12px; }
    .settings-option small { display: block; margin-top: 3px; color: var(--muted); font-size: 10px; line-height: 1.5; }
    body:not(.dev-mode) .dev-only { display: none; }
    body:not(.dev-mode) .result-alternates { grid-template-rows: minmax(0, 1fr); }
    .hero { display: flex; align-items: flex-end; justify-content: space-between; gap: 25px; margin-bottom: 12px; padding-right: 48px; }
    .eyebrow { margin-bottom: 10px; color: var(--green); font-size: 10px; font-weight: 750; letter-spacing: 1.8px; text-transform: uppercase; }
    h1 { margin: 0; font-family: Georgia, "Times New Roman", serif; font-size: clamp(34px, 3.5vw, 44px); font-weight: 500; letter-spacing: -1.8px; }
    .layout { display: grid; grid-template-columns: minmax(0, 1.18fr) minmax(360px, 1fr);
      gap: clamp(14px, 1.3vw, 19px); align-items: stretch; }
    .stack { display: grid; gap: 18px; }
    .card { min-width: 0; padding: 19px; border: 1px solid var(--line); border-radius: 17px; background: var(--white); box-shadow: var(--shadow); }
    .card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 15px; margin-bottom: 13px; }
    .card-head h2 { margin: 0; font-size: 14px; font-weight: 700; letter-spacing: -.15px; }
    .card-head p { margin: 5px 0 0; color: var(--lime); font-size: 11px; }
    .step { display: grid; place-items: center; flex: 0 0 25px; width: 25px; height: 25px; border-radius: 50%;
      color: var(--accent-dark); background: var(--surface-accent); font-size: 11px; font-weight: 700; }
    .preview { position: relative; display: grid; place-items: center; height: 170px; overflow: hidden; border: 1px dashed var(--line-soft);
      border-radius: 13px; background: linear-gradient(145deg, #fafaf7, #f1f1eb); }
    .preview img { display: none; position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
    .preview canvas { display: none; position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; cursor: crosshair; }
    .preview.has-image img { display: block; }
    .preview.has-segmentation canvas { display: block; }
    .preview-empty { display: grid; justify-items: center; gap: 8px; color: var(--text-muted); font-size: 12px; }
    .preview-empty .upload-symbol { display: grid; place-items: center; width: 42px; height: 42px; border-radius: 14px;
      background: var(--surface-raised); color: var(--accent); font-size: 20px; box-shadow: 0 4px 15px #3331; }
    .preview.has-image .preview-empty { display: none; }
    .source-toolbar { display: flex; flex-direction: column; align-items: stretch; gap: 8px; margin: 14px 0 13px; }
    .source-actions { display: flex; flex-wrap: wrap; gap: 7px; }
    .segmentation-status { display: none; margin: 12px 0 9px; color: var(--muted); font-size: 12px; line-height: 1.5; }
    .segmentation-status.visible { display: block; }
    .garment-options { display: flex; flex-wrap: wrap; gap: 7px; margin: 0 0 12px; }
    .garment-choice { padding: 7px 9px; border: 1px solid var(--line); border-radius: 9px; background: var(--surface-raised);
      color: var(--text-default); font-size: 10px; }
    .garment-choice:hover { border-color: var(--plum-mid); }
    .garment-choice.selected { border-color: var(--plum); background: var(--plum-surface); color: var(--plum-dark); font-weight: 650; }
    .button { display: inline-flex; align-items: center; justify-content: center; gap: 8px; min-height: 36px; padding: 8px 12px;
      border: 1px solid var(--line); border-radius: 9px; background: var(--surface-raised); color: var(--text-strong); font-size: 11px; font-weight: 650; }
    .button:hover { border-color: var(--accent-mid); background: var(--surface-hover); }
    .button.small { min-height: 31px; padding: 6px 10px; font-size: 10px; }
    .detect-button { flex-direction: column; }
    .detect-button-content { display: flex; align-items: center; justify-content: center; gap: 7px; }
    .detect-spinner { display: none; color: var(--accent); font-size: 15px; line-height: 1; }
    .detect-button.detecting .detect-spinner { display: inline-block; }
    .detect-button.detecting .detect-spinner { animation: detect-spin .9s linear infinite; }
    .detect-progress { display: none; width: 100%; height: 3px; overflow: hidden; border-radius: 99px; background: var(--line); }
    .detect-button.detecting .detect-progress { display: block; }
    .detect-progress::after { display: block; width: 38%; height: 100%; border-radius: inherit; background: var(--lime);
      content: ""; animation: detect-progress 1.2s ease-in-out infinite; }
    .detect-button:disabled { cursor: wait; }
    @keyframes detect-spin { to { transform: rotate(360deg); } }
    @keyframes detect-progress { from { transform: translateX(-110%); } to { transform: translateX(280%); } }
    @media (prefers-reduced-motion: reduce) {
      .detect-button.detecting .detect-spinner, .detect-progress::after { animation-duration: 3s; }
    }
    .picker { display: grid; grid-template-columns: 84px minmax(0, 1fr); gap: 14px; align-items: stretch; }
    .picker-main { display: flex; min-width: 0; flex-direction: column; }
    .picker-main .preview { flex: 1; height: auto; min-height: 300px; }
    .picker-main .source-actions { flex: 1 1 100%; gap: 8px; }
    .picker-main .source-actions .button { flex: 1 1 auto; min-height: 44px; padding: 10px 14px; font-size: 12px; }
    .gallery-switch { display: grid; grid-template-columns: 52px minmax(0, 1fr) 52px; gap: 8px; }
    .picker-main .gallery-switch .button { min-height: 44px; padding: 10px 14px; font-size: 12px; }
    .picker-main .gallery-switch .gallery-nav { padding: 0; font-size: 22px; font-weight: 700; line-height: 1;
      color: var(--green); background: var(--surface-accent); border-color: var(--accent-mid); }
    .picker-main .gallery-switch .gallery-nav:hover:not(:disabled) { color: var(--plum-dark); background: var(--plum-surface); border-color: var(--plum); }
    .picker-main .gallery-switch .gallery-nav:disabled { opacity: .5; color: var(--text-muted); cursor: default; }
    .gallery { display: grid; grid-template-columns: minmax(0, 1fr); gap: 6px; align-content: start; }
    .thumb { position: relative; height: 56px; overflow: hidden; padding: 0; border: 2px solid transparent; border-radius: 8px;
      background: var(--surface-preview); }
    .thumb img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .thumb:hover { transform: translateY(-1px); }
    .thumb.selected { border-color: var(--plum); box-shadow: 0 0 0 2px #73267020; }
    .thumb.selected::after { position: absolute; top: 5px; right: 5px; content: "✓"; display: grid; place-items: center;
      width: 18px; height: 18px; border-radius: 50%; background: var(--plum); color: white; font-size: 11px; }
    .result-showcase.history-showcase { display: block; flex: 0 0 auto; }
    .history-showcase .result-image.featured { width: 100%; height: min(48dvh, 460px); min-height: 260px; max-height: none; }
    .history-showcase .result-image.featured img { min-height: 0; }
    .field-label { display: block; margin: 0 0 8px; color: var(--text-strong); font-size: 11px; font-weight: 700; }
    .field-label-lg { color: var(--green); font-size: 13px; }
    .field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 13px; }
    .field { min-width: 0; margin-bottom: 12px; }
    .input { width: 100%; min-height: 40px; padding: 9px 11px; border: 1px solid var(--line-strong); border-radius: 9px;
      outline: none; background: var(--surface-raised); color: var(--ink); font-size: 12px; }
    .input:focus { border-color: var(--accent-mid); box-shadow: 0 0 0 3px #c8afe433; }
    .swatches { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .swatch { appearance: none; width: 23px; height: 23px; padding: 0; border: 2px solid #fff; border-radius: 50%; box-shadow: 0 0 0 1px #deded8; }
    .swatch[title="white"] { background: #fff !important; border-color: #777972; box-shadow: none; }
    .swatch-none { flex: 0 0 auto; width: max-content; min-width: max-content; padding: 0 9px;
      border-radius: 99px; background: var(--surface-raised); color: var(--text-subtle); font-size: 9px; white-space: nowrap; }
    .swatch.selected { box-shadow: 0 0 0 2px var(--plum); }
    .material-line { display: flex; align-items: center; gap: 8px; }
    .material-line .input { flex: 1; }
    .material-options { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 0; }
    .material-choice { padding: 5px 8px; border: 1px solid var(--line); border-radius: 99px; background: var(--surface-raised);
      color: var(--text-subtle); font-size: 9px; }
    .material-choice:hover { border-color: var(--plum-mid); color: var(--plum-dark); }
    .material-choice.selected { border-color: var(--plum); background: var(--plum-surface); color: var(--plum-dark); font-weight: 650; }
    .garment-style-editor { display: none; margin-bottom: 14px; }
    .garment-style-editor.visible { display: block; }
    :root { --empty-hint-size: 17px; }  /* tamaño común de las dos frases de aviso de la tarjeta de estilo */
    /* aviso "Selecciona una o más prendas...": grande y centrado en el hueco de la tarjeta cuando no hay ninguna seleccionada */
    .garment-style-editor.visible.no-selection { display: flex; flex-direction: column; justify-content: center; min-height: 220px; }
    #garmentStyleEditor.no-selection #garmentEditorHint { max-width: 340px; margin: 0 auto; color: var(--text-default);
      font-size: var(--empty-hint-size); font-weight: 400; line-height: 1.4; text-align: center; }
    .style-empty { display: none; }
    .card.blank .style-empty { display: grid; place-items: center; min-height: 140px; padding: 16px; text-align: center;
      color: var(--muted); font-size: var(--empty-hint-size); line-height: 1.4; }
    .style-empty p { max-width: 340px; margin: 0; }
    @media (min-width: 1021px) { .stack.right .card.blank .style-empty { flex: 1; } }
    .garment-tabs { display: flex; gap: 6px; margin: 0 0 12px; overflow-x: auto; padding: 2px 1px 5px; }
    .garment-tab { flex: 0 0 auto; max-width: 180px; padding: 7px 10px; border: 1px solid var(--line);
      border-radius: 9px; background: var(--surface-raised); color: var(--text-default); font-size: 10px; }
    .garment-tab.selected { border-color: var(--plum); background: var(--plum-surface); color: var(--plum-dark); font-weight: 650; }
    .garment-tab-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .garment-editor-pane { display: none; }
    .garment-editor-pane.active { display: block; }
    .garment-editor-pane .field + .field,
    #defaultStyle .field + .field { margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--line); }
    .garment-editor-pane .texture-grid { margin-top: 8px; }
    .placement-tools { display: flex; flex-wrap: wrap; gap: 7px; }
    .placement-choice[aria-pressed="true"] { border-color: var(--plum); background: var(--plum-surface); color: var(--plum-dark); }
    .placement-status { margin: 8px 0 0; color: var(--muted); font-size: 10px; line-height: 1.5; }
    .placement-list { display: grid; gap: 5px; margin: 8px 0 0; padding: 0; list-style: none; }
    .placement-item { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 6px 8px;
      border-radius: 8px; background: var(--surface-soft); color: var(--text-default); font-size: 10px; }
    .placement-remove { flex: 0 0 auto; padding: 1px 5px; border: 0; background: transparent; color: var(--accent-dark); font-size: 14px; }
    .preview.placing canvas { cursor: crosshair; }
    .texture-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px; min-height: 67px; }
    .texture-tile { min-width: 0; padding: 0 0 5px; overflow: hidden; border: 2px solid transparent; border-radius: 9px;
      background: var(--surface-muted); color: var(--text-default); font-size: 9px; }
    .texture-tile img { display: block; width: 100%; height: 36px; object-fit: cover; margin-bottom: 4px; }
    .texture-tile span { display: block; overflow: hidden; padding: 0 3px; text-overflow: ellipsis; white-space: nowrap; }
    .texture-tile.selected { border-color: var(--plum); background: var(--plum-surface); color: var(--plum-dark); }
    .hint { margin: 8px 0 0; color: var(--text-muted); font-size: 10px; line-height: 1.5; }
    .right .card-head h2 { font-size: 13px; }
    .right .card-head p { font-size: 10px; }
    .right .step { font-size: 10px; }
    .right .field-label { margin-bottom: 6px; font-size: 10px; }
    .right .field-label-lg { font-size: 12px; }
    .right .input { min-height: 36px; padding: 8px 10px; font-size: 11px; }
    .right .field { margin-bottom: 10px; }
    .right .hint { margin-top: 6px; font-size: 9px; line-height: 1.45; }
    .field > .field-label + .hint { margin: 0 0 8px; }
    .right #garmentEditorHint { font-size: 10px; }
    textarea.input { resize: vertical; min-height: 74px; line-height: 1.5; }
    .generate { width: 100%; min-height: 46px; margin-top: 1px; border: 0; border-radius: 11px; background: var(--accent);
      color: white; font-size: 12px; font-weight: 700; transition: background .2s, transform .2s; }
    .generate:hover:not(:disabled) { transform: translateY(-1px); background: var(--accent-dark); }
    .generate:disabled { cursor: wait; opacity: .65; }
    .generate-actions { margin-top: 1px; }
    .generate-actions .generate { margin-top: 0; }
    .generate-split { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, .9fr); gap: 8px; }
    .generate-split[hidden] { display: none; }
    .generate-secondary { border: 1px solid var(--line-strong); background: var(--surface-raised); color: var(--ink); }
    .generate-secondary:hover:not(:disabled) { border-color: var(--accent-mid); background: var(--surface-hover); color: var(--accent-dark); }
    .generate-busy { display: inline-flex; align-items: center; justify-content: center; gap: 7px; cursor: progress; }
    .generate-busy:disabled { cursor: progress; opacity: .85; }
    .generate-busy .queue-spinner { color: white; font-size: 16px; }
    .generate-cancel { width: 100%; min-height: 36px; margin-top: 8px; border: 1px solid var(--lime-soft); border-radius: 9px;
      background: var(--surface-accent); color: var(--accent-dark); font-size: 11px; font-weight: 600; }
    .generate-cancel:hover:not(:disabled) { border-color: var(--accent-mid); background: var(--surface-hover); }
    .generate-cancel:disabled { opacity: .6; cursor: wait; }
    .generate-cancel[hidden] { display: none; }
    .queue-status { display: flex; align-items: center; justify-content: center; gap: 6px;
      margin: 9px 0 0; color: var(--muted); font-size: 10px; line-height: 1.5; text-align: center; }
    .queue-status[hidden] { display: none; }
    .queue-spinner { display: inline-block; color: var(--lime); font-size: 15px; line-height: 1; animation: queue-spin .9s linear infinite; }
    .queue-spinner[hidden] { display: none; }
    @keyframes queue-spin { to { transform: rotate(360deg); } }
    @media (min-width: 1021px) and (max-height: 900px) {
      main { padding-top: 10px; }
      .hero { margin-bottom: 8px; }
      h1 { font-size: 40px; }
      .card { padding: 15px; }
      .thumb { height: 48px; }
      .field { margin-bottom: 8px; }
      .swatches { gap: 6px; margin-top: 6px; }
      .swatch { width: 20px; height: 20px; }
      textarea.input { min-height: 58px; }
      .hint { margin-top: 5px; }
    }
    .error { display: none; margin-top: 12px; padding: 11px; border-radius: 9px; background: var(--surface-accent); color: var(--accent-dark); font-size: 11px; line-height: 1.5; }
    .error.visible { display: block; }
    .result-dialog { width: min(1000px, calc(100vw - 32px)); max-width: none; max-height: calc(100dvh - 32px); padding: 0;
      overflow: auto; border: 1px solid var(--line); border-radius: 20px; background: var(--white); color: var(--ink);
      box-shadow: 0 28px 100px #12120f40; }
    .result-dialog[open] { display: flex; flex-direction: column; }
    .result-dialog::backdrop { background: #20211ecc; backdrop-filter: blur(5px); }
    .hsv-open { margin-top: 10px; }
    .hsv-dot { width: 14px; height: 14px; border-radius: 50%; flex: none;
      background: conic-gradient(#f00, #ff0, #0f0, #0ff, #00f, #f0f, #f00); }
    .hsv-dialog { width: min(920px, calc(100vw - 32px)); }
    .hsv-body { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(260px, 1fr); gap: 22px; padding: 22px 23px 24px; }
    .hsv-preview { display: grid; place-items: center; min-height: 300px; padding: 10px; border-radius: 14px; background: var(--surface-preview); }
    .hsv-preview canvas { display: block; width: auto; height: auto; max-width: 100%; max-height: min(66dvh, 640px); border-radius: 8px; }
    .hsv-controls { display: flex; flex-direction: column; gap: 20px; min-width: 0; }
    .hsv-garments { display: flex; flex-wrap: wrap; gap: 6px; }
    .hsv-row { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
    .hsv-row .field-label { margin: 0; }
    .hsv-value { color: var(--accent-dark); font-size: 12px; font-weight: 700; font-variant-numeric: tabular-nums; }
    .hsv-hex-input { width: 132px; padding: 7px 9px; text-align: right; text-transform: lowercase; font-variant-numeric: tabular-nums; }
    .hex-row { margin: 16px 0 0; }
    .hsv-strip { position: relative; height: 36px; border: 1px solid var(--line); border-radius: 10px; cursor: ew-resize; touch-action: none;
      background: linear-gradient(to right, #f00 0%, #ff0 16.667%, #0f0 33.333%, #0ff 50%, #00f 66.667%, #f0f 83.333%, #f00 100%); }
    .hsv-strip:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
    .hsv-marker { position: absolute; top: -5px; bottom: -5px; width: 6px; margin-left: -3px; border-radius: 3px;
      background: #fff; box-shadow: 0 0 0 2px #000b; pointer-events: none; }
    .hsv-range { width: 100%; accent-color: var(--accent); }
    .hsv-apply { border-color: var(--accent); background: var(--accent); color: #fff; }
    .hsv-apply:hover { background: var(--accent-dark); color: #fff; }
    .hsv-apply:disabled { opacity: .55; cursor: not-allowed; }
    @media (max-width: 760px) { .hsv-body { grid-template-columns: minmax(0, 1fr); } }
    @media (min-width: 1021px) { .stack.right .hsv-open { min-height: 34px; font-size: 11px; } }
    .color-hsv-view { display: flex; flex-direction: column; gap: 13px; margin-top: 10px; padding: 13px;
      border: 1px solid var(--lavender); border-radius: 12px; background: var(--plum-surface); }
    .color-hsv-view[hidden], .color-swatches-view[hidden] { display: none; }
    .hsv-inline-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .hsv-inline-title { color: var(--plum-mid); font-size: 12px; font-weight: 700; }
    .hsv-inline-color { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }
    .hsv-chip { width: 28px; height: 28px; flex: none; border: 2px solid #fff; border-radius: 50%; box-shadow: 0 0 0 1px var(--line-strong); }
    .hsv-chip-text { color: var(--text-subtle); font-size: 11px; font-variant-numeric: tabular-nums; }
    .hsv-sep { color: var(--text-subtle); font-size: 11px; }
    .hsv-inline-color span:last-child { color: var(--text-subtle); font-size: 11px; font-variant-numeric: tabular-nums; }
    .color-mode-hint { margin: 8px 0 0; }
    .hsv-toggle { display: flex; align-items: center; gap: 9px; width: 100%; padding: 0; border: 0; background: none;
      color: inherit; font: inherit; text-align: left; cursor: pointer; }
    .hsv-toggle:focus-visible { outline: 2px solid var(--accent); outline-offset: 4px; border-radius: 6px; }
    .hsv-triangle { flex: none; width: 0; height: 0; border-style: solid; border-width: 5px 0 5px 8px;
      border-color: transparent transparent transparent var(--accent-dark); transition: transform .15s ease; }
    .hsv-toggle[aria-expanded="true"] .hsv-triangle { transform: rotate(90deg); }
    .hsv-mini-chip { flex: none; width: 14px; height: 14px; margin-left: auto; border: 1px solid var(--line-strong); border-radius: 50%; }
    .hsv-toggle[aria-expanded="true"] .hsv-mini-chip { display: none; }
    .hsv-inline-body { display: flex; flex-direction: column; gap: 13px; }
    .hsv-inline-body[hidden] { display: none; }
    @media (prefers-reduced-motion: reduce) { .hsv-triangle { transition: none; } }
    .results-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 15px;
      padding: 19px 23px; border-bottom: 1px solid var(--line); }
    .results-title { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .results-title h2 { margin: 0; font-family: Georgia, "Times New Roman", serif; font-size: 22px; font-weight: 500; }
    .result-meta { overflow: hidden; margin-top: 3px; color: var(--lime); font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
    .result-filename { width: min(210px, 22vw); min-height: 34px; padding: 7px 9px; border: 1px solid var(--line-strong);
      border-radius: 8px; outline: none; background: var(--surface-raised); color: var(--ink); font-size: 11px; }
    .result-filename:focus { border-color: var(--accent-mid); box-shadow: 0 0 0 3px #c8afe433; }
    .result-actions { display: flex; flex: 0 1 auto; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
    .result-download { border-color: var(--accent); background: var(--accent); color: white; }
    .result-download:hover { border-color: var(--accent-dark); background: var(--accent-dark); color: white; }
    .result-footer { display: flex; justify-content: flex-end; padding: 0 22px 18px; }
    .result-footer[hidden], .result-delete[hidden] { display: none; }  /* sin esto display:flex/inline-flex anula el atributo hidden */
    .result-delete { border-color: var(--lime-soft); background: var(--surface-accent); color: var(--accent-dark); }
    .result-delete:hover { border-color: var(--accent-mid); background: var(--surface-hover); }
    .result-showcase { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(170px, .7fr); gap: 14px;
      min-height: 0; padding: 18px 22px 22px; }
    .result-image { min-width: 0; margin: 0; overflow: hidden; border: 1px solid var(--line); border-radius: 12px; background: var(--surface-muted); }
    .result-image img { display: block; width: 100%; height: 100%; min-height: 0; object-fit: contain; }
    .result-image figcaption { padding: 9px 11px; color: var(--lime); font-size: 10px; }
    .result-image.featured { display: flex; flex-direction: column; min-height: 300px; max-height: calc(100dvh - 155px); }
    .result-image.featured img { flex: 1; min-height: 240px; }
    .result-alternates { display: grid; grid-template-rows: repeat(3, minmax(0, 1fr)); gap: 10px; min-height: 0; }
    .result-alternates .result-image { display: flex; flex-direction: column; min-height: 0; }
    .result-alternates .result-image img { flex: 1; min-height: 60px; }
    .history-segmentation { flex: 0 0 auto; padding: 0 22px 18px; }
    .history-segmentation h3 { margin: 0 0 10px; color: var(--text-strong); font-size: 12px; }
    .history-segmentation-images { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
    .history-segmentation-images .result-image { display: flex; min-height: 145px; flex-direction: column; }
    .history-segmentation-images .result-image img { flex: 1; min-height: 110px; }
    .history-segmentation-empty { margin: 0; color: var(--text-subtle); font-size: 11px; }
    .result-error { display: none; margin: 0 22px 18px; padding: 11px; border-radius: 9px; background: var(--surface-accent); color: var(--accent-dark); font-size: 11px; }
    .result-error.visible { display: block; }
    .confirm-dialog { width: min(430px, calc(100vw - 32px)); }
    .confirm-body { padding: 22px 23px 0; }
    .confirm-body h2 { margin: 0; font-family: Georgia, "Times New Roman", serif; font-size: 21px; font-weight: 500; }
    .confirm-body p { margin: 10px 0 0; color: var(--muted); font-size: 12px; line-height: 1.6; }
    .confirm-actions { display: flex; justify-content: flex-end; gap: 8px; padding: 20px 23px 22px; }
    .confirm-leave { border-color: var(--lime-soft); background: var(--surface-accent); color: var(--accent-dark); }
    .confirm-leave:hover { border-color: var(--accent-mid); background: var(--surface-hover); }
    @media (min-width: 1021px) {
      /* la tarjeta ocupa todo el alto de la fila y el contenido se reparte para llenarlo */
      .stack.right .card { display: flex; flex-direction: column; }
      .stack.right .card-head { margin-bottom: 15px; }
      .stack.right .card-head h2 { font-size: 15px; }
      .stack.right .card-head p { margin-top: 5px; font-size: 11px; }
      .stack.right #defaultStyle { display: flex; flex: 0 0 auto; flex-direction: column; }
      .stack.right #defaultStyle[hidden] { display: none; }  /* sin esto el display:flex anularía el atributo hidden */
      .stack.right .garment-style-editor.visible { display: flex; flex-direction: column; gap: 14px; }
      .stack.right .garment-editor-pane.active { display: flex; flex-direction: column; }
      .stack.right .garment-editor-pane .field,
      .stack.right #defaultStyle .field { margin-bottom: 0; }
      .stack.right .field { margin-bottom: 12px; }
      .stack.right .field-label { margin-bottom: 7px; font-size: 12px; }
      .stack.right .field-label-lg { font-size: 14px; }
      .stack.right .input { min-height: 42px; padding: 10px 12px; border-radius: 10px; font-size: 13px; }
      .stack.right textarea.input { min-height: 78px; }
      .stack.right .hint { margin-top: 7px; font-size: 10.5px; line-height: 1.5; }
      .stack.right .field > .field-label + .hint { margin: 0 0 9px; }
      .stack.right #garmentEditorHint { font-size: 10.5px; }
      .stack.right .swatches { gap: 9px; margin-top: 11px; }
      .stack.right .swatch { width: 26px; height: 26px; }
      .stack.right .swatch-none { padding: 0 11px; font-size: 10px; }
      .stack.right .hex-row { margin-top: 20px; }
      .stack.right .material-options { gap: 7px; margin-top: 9px; }
      .stack.right .material-choice { padding: 6px 11px; font-size: 11px; }
      .stack.right .garment-tab { padding: 8px 12px; font-size: 11px; }
      .stack.right .generate-actions { margin-top: 30px; }
      .stack.right .generate { min-height: 52px; margin-top: 0; border-radius: 12px; font-size: 14px; }
      .stack.right .generate-cancel { min-height: 40px; font-size: 12px; }
      .stack.right .queue-status { font-size: 11px; }
      .stack.right .error { font-size: 12px; }
    }
    @media (min-width: 1021px) and (max-width: 1240px) {
      .shell { grid-template-columns: 220px minmax(0, 1fr); }
      main { padding-right: 38px; }
      .layout { grid-template-columns: minmax(0, 1.25fr) minmax(340px, 1fr); gap: 14px; }
      .card { padding: 16px; }
    }
    .empty-text { display: grid; place-items: center; min-height: 74px; color: var(--text-muted); font-size: 10px; text-align: center; }
    footer { margin-top: 27px; color: var(--text-label); font-size: 10px; text-align: center; }
    .hidden { display: none !important; }
    @media (max-width: 1020px) {
      .shell { grid-template-columns: 205px minmax(0, 1fr); }
      .layout { grid-template-columns: 1fr; gap: 14px; }
      .stack.right { grid-row: 2; }
    }
    @media (max-width: 860px) {
      /* en tablet la barra lateral pasa a ser una cabecera con el historial en horizontal */
      .shell { display: block; }
      aside { position: static; height: auto; padding: 12px 16px; border-right: 0; border-bottom: 1px solid var(--line); }
      .brand { padding: 0; }
      .brand-mark { width: 42px; height: 42px; border-radius: 13px; font-size: 25px; }
      .brand-name { font-size: 25px; }
      .brand-sub { display: none; }
      .history-nav { flex: none; margin-top: 12px; min-height: 0; }
      .history-nav-head { margin-bottom: 6px; }
      .history-list { display: flex; flex-direction: row; gap: 6px; overflow-x: auto; overflow-y: hidden; padding: 1px 0 5px; }
      .history-item { flex: 0 0 auto; width: auto; max-width: 190px; }
      .history-open-button { padding: 7px 9px; }
      main { padding: 16px clamp(14px, 3vw, 26px) 38px; }
      .hero { padding-right: 44px; }
    }
    @media (max-width: 680px) {
      .shell { display: block; }
      aside { position: static; height: auto; padding: 12px 16px; border-right: 0; border-bottom: 1px solid var(--line); }
      .brand { padding: 0; } .history-nav { display: none; }
      main { padding: 18px 14px 36px; }
      .settings-button { width: 30px; height: 30px; }
      .hero { align-items: flex-start; margin-bottom: 10px; padding-right: 42px; }
      h1 { font-size: clamp(30px, 5vw, 36px); }
      .card { padding: 15px; }
      /* en móvil no cabe en dos columnas: la imagen elegida arriba y los ejemplos debajo en horizontal */
      .picker { grid-template-columns: minmax(0, 1fr); }
      .picker-main { order: -1; }
      .picker-main .preview { flex: none; min-height: 0; height: 150px; }
      .gallery { grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; }
      .thumb { height: 80px; }
      .result-dialog { width: calc(100vw - 20px); max-height: calc(100dvh - 20px); border-radius: 15px; }
      .results-head { align-items: flex-start; padding: 14px; }
      .results-title { align-items: flex-start; gap: 8px; }
      .results-title h2 { font-size: 18px; }
      .result-actions { flex-direction: column; }
      .result-filename { width: min(145px, 31vw); }
      .result-showcase { grid-template-columns: minmax(0, 1fr); padding: 12px; }
      .result-image.featured { min-height: 220px; max-height: 48dvh; }
      .result-image.featured img { min-height: 180px; }
      .history-showcase .result-image.featured { height: min(38dvh, 320px); min-height: 190px; }
      .result-alternates { grid-template-columns: minmax(0, 1fr); grid-template-rows: 95px; }
      .dev-mode .result-alternates { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .result-alternates .result-image img { min-height: 0; }
    }
    @media (max-width: 480px) {
      main { padding: 14px 11px 32px; }
      .settings { top: 10px; right: 10px; }
      .settings-menu { width: calc(100vw - 20px); }
      .hero { padding-right: 40px; }
      .card { padding: 13px; border-radius: 14px; }
      .picker-main .preview { height: 130px; }
      .gallery { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .thumb { height: 74px; }
      .generate-split { grid-template-columns: minmax(0, 1fr); }
      .result-dialog { width: calc(100vw - 14px); max-height: calc(100dvh - 14px); }
      .results-head { flex-direction: column; align-items: stretch; gap: 10px; padding: 13px; }
      .result-actions { flex-direction: row; flex-wrap: wrap; width: 100%; }
      .result-filename { flex: 1 1 130px; width: auto; }
      .results-title h2 { font-size: 17px; }
      .result-showcase { padding: 10px; gap: 10px; }
      .result-image.featured { min-height: 190px; max-height: 42dvh; }
      .result-image.featured img { min-height: 150px; }
      .history-showcase .result-image.featured { height: min(34dvh, 280px); min-height: 170px; }
      .history-segmentation { padding: 0 13px 14px; }
      .result-footer { padding: 0 13px 14px; }
      .result-error { margin: 0 13px 14px; }
      .confirm-body { padding: 19px 18px 0; }
      .confirm-actions { padding: 16px 18px 18px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">
        <div class="brand-mark">a</div>
        <div><div class="brand-name">atelier</div><div class="brand-sub">color studio</div></div>
      </div>
      <section class="history-nav" aria-label="Prendas editadas">
        <div class="history-nav-head">
          <span class="history-nav-title">Prendas editadas</span>
          <button class="history-refresh" id="refreshHistory" type="button" aria-label="Actualizar historial">↻</button>
        </div>
        <div class="history-list" id="historyList" aria-label="Historial de diseños">
          <div class="empty-text">Cargando historial…</div>
        </div>
      </section>
    </aside>
    <main id="editor">
      <div class="settings">
            <button class="dev-mode-indicator" id="devModeIndicator" type="button"
                    title="Desactivar modo desarrollador" aria-label="Desactivar modo desarrollador">Modo desarrollador</button>
            <button class="settings-button" id="settingsButton" type="button" title="Configuración" aria-label="Configuración"
                    aria-haspopup="true" aria-expanded="false" aria-controls="settingsMenu">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                   stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
              </svg>
            </button>
            <div class="settings-menu" id="settingsMenu" hidden>
              <div class="settings-title">Configuración</div>
              <div class="settings-title">Tema</div>
              <div class="settings-themes" role="group" aria-label="Tema de la interfaz">
                <label class="settings-theme">
                  <input type="radio" name="theme" id="themeLight" value="light">
                  <span>Claro</span>
                </label>
                <label class="settings-theme">
                  <input type="radio" name="theme" id="themeDark" value="dark">
                  <span>Oscuro</span>
                </label>
              </div>
              <label class="settings-option">
                <input type="checkbox" id="devModeToggle">
                <span>
                  <b>Modo desarrollador</b>
                  <small>Muestra la confianza de detección de las prendas y los pasos intermedios del resultado (máscara y edición del modelo).</small>
                </span>
              </label>
            </div>
      </div>
      <header class="hero">
        <div>
          <h1>Un nuevo tono. La misma esencia.</h1>
        </div>
      </header>
      <div class="layout">
        <div class="stack">
          <section class="card">
            <div class="card-head">
              <div><h2>Elige tu imagen</h2></div>
            </div>
            <div class="picker">
              <div class="gallery" id="imageGallery" aria-label="Imágenes del catálogo"></div>
              <div class="picker-main">
                <div class="preview" id="sourcePreview">
                  <div class="preview-empty"><span class="upload-symbol">↥</span><span>Selecciona una imagen para empezar</span></div>
                  <img id="sourceImage" alt="Imagen seleccionada">
                  <canvas id="segmentationOverlay" aria-label="Máscaras de prendas detectadas"></canvas>
                </div>
                <div class="segmentation-status" id="segmentationStatus" aria-live="polite"></div>
                <div class="garment-options" id="garmentOptions" role="group" aria-label="Prendas detectadas"></div>
                <div class="source-toolbar">
                  <div class="source-actions">
                    <button class="button small garment-choice selected detect-button" id="detectGarments" type="button">
                      <span class="detect-button-content">
                        <span class="detect-spinner" aria-hidden="true">↻</span>
                        <span class="detect-label">Detectar prendas</span>
                      </span>
                      <span class="detect-progress" role="progressbar" aria-label="Detección de prendas en curso"
                            aria-valuetext="Detección en curso"></span>
                    </button>
                    <button class="button small" id="uploadButton" type="button">＋ Subir foto</button>
                  </div>
                  <div class="gallery-switch" role="group" aria-label="Navegar por las imágenes de ejemplo">
                    <button class="button small gallery-nav" id="prevImages" type="button" aria-label="Imágenes anteriores" title="Imágenes anteriores" disabled>←</button>
                    <button class="button small" id="refreshImages" type="button">↻ Otras imágenes</button>
                    <button class="button small gallery-nav" id="nextImages" type="button" aria-label="Imágenes siguientes" title="Imágenes siguientes" disabled>→</button>
                  </div>
                  <input class="hidden" id="fileInput" type="file" accept="image/jpeg,image/png,image/webp,image/bmp">
                </div>
              </div>
            </div>
          </section>
        </div>
        <div class="stack right">
          <section class="card blank" id="styleCard">
            <div class="card-head">
              <div><h2> Modifica las prendas seleccionadas</h2></div>
            </div>
            <div class="style-empty" id="styleEmpty">
              <p>Elige una imagen y pulsa <b>Detectar prendas</b> para configurar el color y la textura de cada una.</p>
            </div>
            <div id="defaultStyle" hidden>
            <div class="field">
              <label class="field-label" for="color">Color</label>
              <p class="hint">Elige un color o selecciona «Ninguno» para conservar el original.</p>
              <input class="input" id="color" list="colors" value="black" maxlength="80" placeholder="Escribe un color en inglés">
              <datalist id="colors"></datalist>
              <div class="swatches" id="swatches" aria-label="Colores sugeridos"></div>
            </div>
            <div class="field">
              <label class="field-label field-label-lg" for="material">Material o textura</label>
              <p class="hint">Elige o escribe un material. Atelier intentará reproducir su grano, costuras y patrón sobre la prenda, manteniendo el color elegido.</p>
              <div class="material-line">
                <input class="input" id="material" value="none" maxlength="80" autocomplete="off" placeholder="Ej. denim, silk, leather">
              </div>
              <div class="material-options" id="materialOptions" role="group" aria-label="Materiales sugeridos"></div>
            </div>
            </div>
            <div class="garment-style-editor" id="garmentStyleEditor" aria-live="polite">
              <div class="garment-tabs" id="garmentTabs" role="tablist" aria-label="Configurar cada prenda"></div>
              <div id="garmentEditorPanes"></div>
              <p class="hint" id="garmentEditorHint"></p>
            </div>
            <div class="generate-actions" hidden>
              <button class="generate" id="generateButton" type="button" aria-live="polite">✦ &nbsp; Crear mi diseño</button>
              <div class="generate-split" id="generateSplit" hidden>
                <button class="generate generate-secondary" id="queueAnotherButton" type="button">＋ &nbsp; Añadir otra a la cola</button>
                <button class="generate generate-busy" id="generatingButton" type="button" disabled aria-live="polite">
                  <span class="queue-spinner" aria-hidden="true">↻</span>
                  <span id="generatingLabel">Generando…</span>
                </button>
              </div>
            </div>
            <button class="generate-cancel" id="cancelGeneration" type="button" hidden>Cancelar la generación en curso</button>
            <p class="queue-status" id="queueStatus" role="status" aria-live="polite" hidden>
              <span class="queue-spinner" id="queueSpinner" aria-hidden="true" hidden>↻</span>
              <span id="queueStatusText"></span>
            </p>
            <div class="error" id="errorMessage" role="alert"></div>
          </section>
          <dialog class="result-dialog" id="resultsCard" aria-labelledby="resultHeading">
            <div class="results-head">
              <div class="results-title"><div><h2 id="resultHeading">Tu resultado</h2><div class="result-meta" id="resultMeta"></div></div></div>
              <div class="result-actions">
                <input class="result-filename" id="resultFilename" aria-label="Nombre del archivo PNG" maxlength="100" value="atelier" title="Nombre del archivo PNG">
                <a class="button small result-download" id="downloadButton" download>↓ Descargar</a>
                <button class="button small" id="closeResults" type="button">Cerrar</button>
              </div>
            </div>
            <div class="result-showcase">
              <figure class="result-image featured"><img id="resultFinal" alt="Resultado final"><figcaption>Resultado final</figcaption></figure>
              <div class="result-alternates">
                <figure class="result-image"><img id="resultOriginal" alt="Imagen original"><figcaption>Original</figcaption></figure>
                <figure class="result-image dev-only"><img id="resultMask" alt="Máscara segmentada"><figcaption>Máscara</figcaption></figure>
                <figure class="result-image dev-only"><img id="resultEdited" alt="Edición del modelo"><figcaption>Edición del modelo</figcaption></figure>
              </div>
            </div>
            <section class="history-segmentation dev-only" id="historySegmentation" hidden>
              <h3>Máscaras de segmentación</h3>
              <div class="history-segmentation-images" id="historySegmentationImages"></div>
              <p class="history-segmentation-empty" id="historySegmentationEmpty" hidden></p>
            </section>
            <div class="result-footer" id="resultFooter" hidden>
              <button class="button small result-delete" id="deleteHistoryButton" type="button" hidden>Eliminar diseño</button>
            </div>
            <div class="result-error" id="resultError" role="alert"></div>
          </dialog>
          <dialog class="result-dialog confirm-dialog" id="reloadWarning" aria-labelledby="reloadWarningTitle">
            <div class="confirm-body">
              <h2 id="reloadWarningTitle">¿Salir de la generación?</h2>
              <p>Si recargas la página ahora, perderás la generación actual. Elige si quieres volver o salir.</p>
            </div>
            <div class="confirm-actions">
              <button class="button small" id="stayOnPage" type="button">Volver</button>
              <button class="button small confirm-leave" id="leavePage" type="button">Salir</button>
            </div>
          </dialog>
        </div>
      </div>
      <footer>Atelier · Recolorado de prendas con inteligencia artificial</footer>
    </main>
  </div>
  <script>
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
      hsvReset.className = "button small";
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
      hsvBody.append(hueWrap, satWrap, valWrap, chipRow, hsvReset, hsvHint);
      hsvView.append(hsvHead, hsvBody);

      colorField.append(colorLabel, swatches, hexRow, modeHint, hsvView);
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
        edit.hsv = hsvFromHex(hex);
        edit.hsvActive = true;
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
      $("#segmentationStatus").textContent += " No se puede cambiar la imagen hasta que termine.";
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
        items.forEach((item) => {
          const openButton = document.createElement("button");
          openButton.type = "button";
          openButton.className = "history-open-button history-item";
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
    // personalizados (sliders HSV o código HEX), incluido un negro/blanco retocado a mano → recolor_hsv.
    const PIX2PIX_COLORS = new Set(["black", "white"]);
    function sameHsv(a, b) {
      return Math.round(a.hue) % 360 === b.hue && Math.round(a.sat) === b.sat && Math.round(a.val) === b.val;
    }
    function colorRoute(edit) {
      const name = (edit.color || "").trim().toLowerCase();
      if (PIX2PIX_COLORS.has(name) && (!edit.hsv || !edit.hsvActive || sameHsv(edit.hsv, hsvFromHex(colorHex[name])))) {
        return {color: name, hsv: null};
      }
      if (edit.hsv && edit.hsvActive) return {color: "", hsv: edit.hsv};
      const hexValue = colorHex[name] || normalizeHexColor(name);   // p. ej. el color por defecto si no se pudo leer el original
      if (hexValue) return {color: "", hsv: edit.hsv || hsvFromHex(hexValue)};
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
  </script>
</body>
</html>"""


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


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML


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