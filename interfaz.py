"""Interfaz web local para el recoloreado de prendas con InstructPix2Pix.

Uso desde la raíz del proyecto:
    python interfaz.py
"""

import logging
import random
import re
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
        app.state.textures = None
        yield
        if worker is not None:
            await run_in_threadpool(worker.shutdown, wait=True)
            worker = None


app = FastAPI(title="Atelier — recolore prendas", lifespan=lifespan)


class GenerationRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=100)
    color: str = Field(default="", max_length=80)
    texture: str = Field(default=C.PIX2PIX_NO_TEXTURE, max_length=80)
    extra: str = Field(default="", max_length=500)
    texture_id: str | None = Field(default=None, max_length=100)
    segmentation_id: str | None = Field(default=None, max_length=100)
    garment_indices: list[int] | None = None


class SegmentationRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=100)


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
      --line: #e9e9e3; --accent: #c75d3a; --accent-dark: #a8472b;
      --green: #657b63; --shadow: 0 12px 36px #29271e0b;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); background: var(--paper); }
    button, input, select { font: inherit; }
    button { cursor: pointer; }
    .shell { min-height: 100vh; display: grid; grid-template-columns: 248px minmax(0, 1fr); }
    aside { position: sticky; top: 0; height: 100vh; padding: 27px 20px; background: #fff;
      border-right: 1px solid var(--line); display: flex; flex-direction: column; }
    .brand { display: flex; align-items: center; gap: 11px; padding: 0 8px 27px; }
    .brand-mark { display: grid; place-items: center; width: 38px; height: 38px; border-radius: 13px;
      color: white; background: var(--accent); font-family: Georgia, serif; font-size: 22px; }
    .brand-name { font-family: Georgia, "Times New Roman", serif; font-size: 22px; letter-spacing: -.6px; }
    .brand-sub { margin-top: 1px; color: var(--muted); font-size: 10px; letter-spacing: 1.7px; text-transform: uppercase; }
    .nav-label { padding: 0 10px; color: #a1a39d; font-size: 10px; font-weight: 700; letter-spacing: 1.3px; text-transform: uppercase; }
    .nav-item { display: flex; align-items: center; gap: 12px; margin-top: 9px; padding: 12px 13px;
      border-radius: 10px; color: #565851; font-size: 13px; text-decoration: none; }
    .nav-item.active { color: var(--accent-dark); background: #fbf0eb; font-weight: 650; }
    .nav-icon { width: 18px; text-align: center; font-size: 16px; }
    .history-nav { display: flex; flex-direction: column; min-height: 0; margin-top: 22px; }
    .history-nav-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 0 0 9px; }
    .history-nav-title { padding-left: 10px; color: #a1a39d; font-size: 10px; font-weight: 700; letter-spacing: 1.1px; text-transform: uppercase; }
    .history-refresh { display: grid; place-items: center; width: 27px; height: 27px; border: 1px solid var(--line);
      border-radius: 8px; background: white; color: #666860; font-size: 12px; }
    .history-refresh:hover { border-color: #d5b6aa; color: var(--accent-dark); }
    .history-list { display: grid; gap: 7px; max-height: min(46vh, 430px); overflow-y: auto; padding: 1px 4px 2px 0; }
    .history-item { display: flex; align-items: center; justify-content: space-between; gap: 8px; width: 100%;
      min-height: 39px; padding: 9px 10px; border: 1px solid var(--line); border-radius: 9px;
      background: #fff; color: #454741; text-align: left; }
    .history-item:hover { border-color: #d5b6aa; background: #fdf9f6; color: var(--accent-dark); }
    .history-item-name { display: block; min-width: 0; overflow: hidden; font-size: 11px; font-weight: 600;
      text-overflow: ellipsis; white-space: nowrap; }
    .history-open { flex: 0 0 auto; color: var(--accent); font-size: 14px; }
    .side-note { margin-top: auto; padding: 16px; border: 1px solid var(--line); border-radius: 14px; background: #fbfbf8; }
    .side-note strong { display: block; margin-bottom: 6px; font-size: 12px; }
    .side-note p { margin: 0; color: var(--muted); font-size: 11px; line-height: 1.6; }
    main { min-width: 0; max-width: 1500px; width: 100%; margin: 0 auto; padding: 20px clamp(20px, 4vw, 58px) 60px; }
    .topline { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; }
    .breadcrumb { color: var(--muted); font-size: 12px; }
    .device-pill { display: flex; align-items: center; gap: 8px; padding: 8px 11px; border: 1px solid var(--line);
      border-radius: 99px; background: #fff; color: #5e6159; font-size: 11px; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: #82a179; }
    .hero { display: flex; align-items: flex-end; justify-content: space-between; gap: 25px; margin-bottom: 18px; }
    .eyebrow { margin-bottom: 10px; color: var(--accent); font-size: 10px; font-weight: 750; letter-spacing: 1.8px; text-transform: uppercase; }
    h1 { margin: 0; font-family: Georgia, "Times New Roman", serif; font-size: clamp(34px, 3.5vw, 44px); font-weight: 500; letter-spacing: -1.8px; }
    .hero p { margin: 11px 0 0; color: var(--muted); font-size: 13px; }
    .step-chip { flex: 0 0 auto; margin-bottom: 3px; color: #81837c; font-size: 11px; }
    .step-chip b { color: var(--ink); }
    .layout { display: grid; grid-template-columns: minmax(0, 1.42fr) minmax(300px, .86fr); gap: 19px; align-items: start; }
    .stack { display: grid; gap: 18px; }
    .card { min-width: 0; padding: 19px; border: 1px solid var(--line); border-radius: 17px; background: var(--white); box-shadow: var(--shadow); }
    .card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 15px; margin-bottom: 13px; }
    .card-head h2 { margin: 0; font-size: 14px; font-weight: 700; letter-spacing: -.15px; }
    .card-head p { margin: 5px 0 0; color: var(--muted); font-size: 11px; }
    .step { display: grid; place-items: center; flex: 0 0 25px; width: 25px; height: 25px; border-radius: 50%;
      color: var(--accent-dark); background: #fbf0eb; font-size: 11px; font-weight: 700; }
    .preview { position: relative; display: grid; place-items: center; height: 170px; overflow: hidden; border: 1px dashed #dedfd7;
      border-radius: 13px; background: linear-gradient(145deg, #fafaf7, #f1f1eb); }
    .preview img { display: none; position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
    .preview canvas { display: none; position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; cursor: crosshair; }
    .preview.has-image img { display: block; }
    .preview.has-segmentation canvas { display: block; }
    .preview-empty { display: grid; justify-items: center; gap: 8px; color: #8b8d86; font-size: 12px; }
    .preview-empty .upload-symbol { display: grid; place-items: center; width: 42px; height: 42px; border-radius: 14px;
      background: white; color: var(--accent); font-size: 20px; box-shadow: 0 4px 15px #3331; }
    .preview.has-image .preview-empty { display: none; }
    .source-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin: 14px 0 13px; }
    .source-actions { display: flex; flex-wrap: wrap; gap: 7px; }
    .segmentation-status { display: none; margin: -5px 0 10px; color: var(--muted); font-size: 10px; line-height: 1.5; }
    .segmentation-status.visible { display: block; }
    .garment-options { display: flex; flex-wrap: wrap; gap: 7px; margin: -2px 0 12px; }
    .garment-choice { padding: 7px 9px; border: 1px solid var(--line); border-radius: 9px; background: white;
      color: #565851; font-size: 10px; }
    .garment-choice:hover { border-color: #d5b6aa; }
    .garment-choice.selected { border-color: var(--accent); background: #fbf0eb; color: var(--accent-dark); font-weight: 650; }
    .source-name { overflow: hidden; color: var(--muted); font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
    .button { display: inline-flex; align-items: center; justify-content: center; gap: 8px; min-height: 36px; padding: 8px 12px;
      border: 1px solid var(--line); border-radius: 9px; background: white; color: #43453f; font-size: 11px; font-weight: 650; }
    .button:hover { border-color: #d5b6aa; background: #fdf9f6; }
    .button.small { min-height: 31px; padding: 6px 10px; font-size: 10px; }
    .gallery { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; }
    .thumb { position: relative; height: 112px; overflow: hidden; padding: 0; border: 2px solid transparent; border-radius: 8px;
      background: #f2f2ed; }
    .thumb img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .thumb:hover { transform: translateY(-1px); }
    .thumb.selected { border-color: var(--accent); box-shadow: 0 0 0 2px #c75d3a20; }
    .thumb.selected::after { position: absolute; top: 5px; right: 5px; content: "✓"; display: grid; place-items: center;
      width: 18px; height: 18px; border-radius: 50%; background: var(--accent); color: white; font-size: 11px; }
    .gallery-note { margin: 12px 0 0; color: var(--muted); font-size: 10px; }
    .history-showcase { grid-template-columns: minmax(0, 1fr); }
    .history-showcase .result-image.featured { width: 100%; }
    .field-label { display: block; margin: 0 0 8px; color: #454741; font-size: 11px; font-weight: 700; }
    .field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 13px; }
    .field { min-width: 0; margin-bottom: 12px; }
    .input { width: 100%; min-height: 40px; padding: 9px 11px; border: 1px solid #e5e5df; border-radius: 9px;
      outline: none; background: #fff; color: var(--ink); font-size: 12px; }
    .input:focus { border-color: #d99b85; box-shadow: 0 0 0 3px #c75d3a14; }
    .swatches { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .swatch { width: 23px; height: 23px; padding: 0; border: 2px solid #fff; border-radius: 50%; box-shadow: 0 0 0 1px #deded8; }
    .swatch-none { width: auto; min-width: 38px; padding: 0 7px; border-radius: 99px; background: #fff; color: #62645d; font-size: 9px; }
    .swatch.selected { box-shadow: 0 0 0 2px var(--accent); }
    .material-line { display: flex; align-items: center; gap: 8px; }
    .material-line .input { flex: 1; }
    .material-options { display: flex; flex-wrap: wrap; gap: 6px; margin: -2px 0 10px; }
    .material-choice { padding: 5px 8px; border: 1px solid var(--line); border-radius: 99px; background: #fff;
      color: #62645d; font-size: 9px; }
    .material-choice:hover { border-color: #d5b6aa; color: var(--accent-dark); }
    .material-choice.selected { border-color: var(--accent); background: #fbf0eb; color: var(--accent-dark); font-weight: 650; }
    .texture-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px; min-height: 67px; }
    .texture-tile { min-width: 0; padding: 0 0 5px; overflow: hidden; border: 2px solid transparent; border-radius: 9px;
      background: #f4f3ef; color: #565851; font-size: 9px; }
    .texture-tile img { display: block; width: 100%; height: 36px; object-fit: cover; margin-bottom: 4px; }
    .texture-tile span { display: block; overflow: hidden; padding: 0 3px; text-overflow: ellipsis; white-space: nowrap; }
    .texture-tile.selected { border-color: var(--accent); }
    .hint { margin: 8px 0 0; color: #94968f; font-size: 10px; line-height: 1.5; }
    textarea.input { resize: vertical; min-height: 74px; line-height: 1.5; }
    .generate { width: 100%; min-height: 46px; margin-top: 1px; border: 0; border-radius: 11px; background: var(--accent);
      color: white; font-size: 12px; font-weight: 700; transition: background .2s, transform .2s; }
    .generate:hover:not(:disabled) { transform: translateY(-1px); background: var(--accent-dark); }
    .generate:disabled { cursor: wait; opacity: .65; }
    @media (min-width: 1021px) and (max-height: 900px) {
      main { padding-top: 10px; }
      .topline { margin-bottom: 12px; }
      .hero { margin-bottom: 12px; }
      h1 { font-size: 40px; }
      .card { padding: 15px; }
      .preview { height: 160px; }
      .thumb { height: 102px; }
      .gallery-note { margin-top: 8px; }
      .field { margin-bottom: 8px; }
      .swatches { gap: 6px; margin-top: 6px; }
      .swatch { width: 20px; height: 20px; }
      .texture-tile img { height: 36px; }
      .texture-tile { padding-bottom: 3px; font-size: 8px; }
      textarea.input { min-height: 58px; }
      .hint { margin-top: 5px; }
    }
    .progress-card { display: none; margin-top: 14px; padding: 13px; border: 1px solid #e7e5dc; border-radius: 10px; background: #fbfbf7; }
    .progress-card.visible { display: block; }
    .progress-top { display: flex; justify-content: space-between; gap: 12px; font-size: 10px; }
    .progress-message { overflow: hidden; color: #53554e; text-overflow: ellipsis; white-space: nowrap; }
    .progress-state { color: var(--green); font-weight: 700; }
    .progress-track { height: 4px; margin-top: 10px; overflow: hidden; border-radius: 9px; background: #eaeae4; }
    .progress-bar { width: 30%; height: 100%; border-radius: inherit; background: var(--accent); animation: progress 1.8s ease-in-out infinite; }
    @keyframes progress { from { transform: translateX(-110%); } to { transform: translateX(430%); } }
    .error { display: none; margin-top: 12px; padding: 11px; border-radius: 9px; background: #fff0ed; color: #a33d29; font-size: 11px; line-height: 1.5; }
    .error.visible { display: block; }
    .result-dialog { width: min(1000px, calc(100vw - 32px)); max-width: none; max-height: calc(100dvh - 32px); padding: 0;
      overflow: auto; border: 1px solid var(--line); border-radius: 20px; background: var(--white); color: var(--ink);
      box-shadow: 0 28px 100px #12120f40; }
    .result-dialog[open] { display: flex; flex-direction: column; }
    .result-dialog::backdrop { background: #20211ecc; backdrop-filter: blur(5px); }
    .results-head { display: flex; align-items: center; justify-content: space-between; gap: 15px; padding: 19px 23px;
      border-bottom: 1px solid var(--line); }
    .results-title { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .results-title h2 { margin: 0; font-family: Georgia, "Times New Roman", serif; font-size: 22px; font-weight: 500; }
    .result-meta { overflow: hidden; margin-top: 3px; color: var(--muted); font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
    .result-filename { width: min(210px, 22vw); min-height: 34px; padding: 7px 9px; border: 1px solid #e5e5df;
      border-radius: 8px; outline: none; color: var(--ink); font-size: 11px; }
    .result-filename:focus { border-color: #d99b85; box-shadow: 0 0 0 3px #c75d3a14; }
    .result-actions { display: flex; flex: 0 0 auto; gap: 8px; }
    .result-download { border-color: var(--accent); background: var(--accent); color: white; }
    .result-download:hover { border-color: var(--accent-dark); background: var(--accent-dark); color: white; }
    .result-showcase { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(170px, .7fr); gap: 14px;
      min-height: 0; padding: 18px 22px 22px; }
    .result-image { min-width: 0; margin: 0; overflow: hidden; border: 1px solid var(--line); border-radius: 12px; background: #f6f6f2; }
    .result-image img { display: block; width: 100%; height: 100%; min-height: 0; object-fit: contain; }
    .result-image figcaption { padding: 9px 11px; color: #62645d; font-size: 10px; }
    .result-image.featured { display: flex; flex-direction: column; min-height: 300px; max-height: calc(100dvh - 155px); }
    .result-image.featured img { flex: 1; min-height: 240px; }
    .result-alternates { display: grid; grid-template-rows: repeat(3, minmax(0, 1fr)); gap: 10px; min-height: 0; }
    .result-alternates .result-image { display: flex; flex-direction: column; min-height: 0; }
    .result-alternates .result-image img { flex: 1; min-height: 60px; }
    .result-error { display: none; margin: 0 22px 18px; padding: 11px; border-radius: 9px; background: #fff0ed; color: #a33d29; font-size: 11px; }
    .result-error.visible { display: block; }
    .empty-text { display: grid; place-items: center; min-height: 74px; color: #92948e; font-size: 10px; text-align: center; }
    footer { margin-top: 27px; color: #999b94; font-size: 10px; text-align: center; }
    .hidden { display: none !important; }
    @media (max-width: 1020px) {
      .shell { grid-template-columns: 205px minmax(0, 1fr); }
      .layout { grid-template-columns: 1fr; }
      .stack.right { grid-row: 2; }
    }
    @media (max-width: 680px) {
      .shell { display: block; }
      aside { position: static; height: auto; padding: 13px 18px; border-right: 0; border-bottom: 1px solid var(--line); }
      .brand { padding: 0; } .nav-label, .nav-item, .history-nav, .side-note { display: none; }
      main { padding: 23px 15px 40px; }
      .topline { margin-bottom: 22px; } .device-pill { font-size: 9px; }
      .hero { align-items: flex-start; } .step-chip { padding-top: 6px; white-space: nowrap; }
      .card { padding: 17px; } .gallery { grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; }
      .preview { height: 150px; }
      .result-dialog { width: calc(100vw - 20px); max-height: calc(100dvh - 20px); border-radius: 15px; }
      .results-head { align-items: flex-start; padding: 14px; }
      .results-title { align-items: flex-start; gap: 8px; }
      .results-title h2 { font-size: 18px; }
      .result-actions { flex-direction: column; }
      .result-filename { width: min(145px, 31vw); }
      .result-showcase { grid-template-columns: minmax(0, 1fr); padding: 12px; }
      .result-image.featured { min-height: 220px; max-height: 48dvh; }
      .result-image.featured img { min-height: 180px; }
      .result-alternates { grid-template-columns: repeat(3, minmax(0, 1fr)); grid-template-rows: 95px; }
      .result-alternates .result-image img { min-height: 0; }
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
      <div class="nav-label">Espacio creativo</div>
      <a class="nav-item active" href="#editor"><span class="nav-icon">◈</span>Editor de prendas</a>
      <section class="history-nav" aria-label="Prendas editadas">
        <div class="history-nav-head">
          <span class="history-nav-title">Prendas editadas</span>
          <button class="history-refresh" id="refreshHistory" type="button" aria-label="Actualizar historial">↻</button>
        </div>
        <div class="history-list" id="historyList" aria-label="Historial de diseños">
          <div class="empty-text">Cargando historial…</div>
        </div>
      </section>
      <div class="side-note">
        <strong>Color con intención</strong>
        <p>El modelo conserva la prenda en su sitio y aplica el cambio dentro de sus bordes.</p>
      </div>
    </aside>
    <main id="editor">
      <div class="topline">
        <div class="breadcrumb">Estudio <span aria-hidden="true">/</span> <b>Recolorado creativo</b></div>
        <div class="device-pill"><span class="dot"></span> Generación local activada</div>
      </div>
      <header class="hero">
        <div>
          <h1>Un nuevo tono. La misma esencia.</h1>
          <p>Elige una imagen y explora una nueva versión de tu prenda.</p>
        </div>
        <div class="step-chip"><b>01</b> — Preparar diseño</div>
      </header>
      <div class="layout">
        <div class="stack">
          <section class="card">
            <div class="card-head">
              <div><h2>Elige tu imagen</h2><p>Selecciona una prenda del catálogo o sube tu propia foto.</p></div>
              <span class="step">1</span>
            </div>
            <div class="preview" id="sourcePreview">
              <div class="preview-empty"><span class="upload-symbol">↥</span><span>Selecciona una imagen para empezar</span></div>
              <img id="sourceImage" alt="Imagen seleccionada">
              <canvas id="segmentationOverlay" aria-label="Máscaras de prendas detectadas"></canvas>
            </div>
            <div class="source-toolbar">
              <span class="source-name" id="sourceName">Aún no has elegido una imagen</span>
              <div class="source-actions">
                <button class="button small" id="uploadButton" type="button">＋ Subir foto</button>
                <button class="button small" id="refreshImages" type="button">↻ Otras imágenes</button>
                <button class="button small" id="detectGarments" type="button">◌ Detectar prendas</button>
              </div>
              <input class="hidden" id="fileInput" type="file" accept="image/jpeg,image/png,image/webp,image/bmp">
            </div>
            <div class="segmentation-status" id="segmentationStatus" aria-live="polite"></div>
            <div class="garment-options" id="garmentOptions" role="group" aria-label="Prendas detectadas"></div>
            <div class="gallery" id="imageGallery" aria-label="Imágenes del catálogo"></div>
            <p class="gallery-note">Elige una de las imágenes aleatorias del conjunto o carga un JPG, PNG, WEBP o BMP (máx. 20 MB).</p>
          </section>
        </div>
        <div class="stack right">
          <section class="card">
            <div class="card-head">
              <div><h2>Define el nuevo estilo</h2><p>Combina un color con el material que prefieras.</p></div>
              <span class="step">3</span>
            </div>
            <div class="field">
              <label class="field-label" for="color">Color</label>
              <input class="input" id="color" list="colors" value="black" maxlength="80" placeholder="Escribe un color en inglés">
              <datalist id="colors"></datalist>
              <div class="swatches" id="swatches" aria-label="Colores sugeridos"></div>
              <p class="hint">Elige un color o selecciona none para conservar el original.</p>
            </div>
            <div class="field">
              <label class="field-label" for="material">Material o textura</label>
              <div class="material-line">
                <input class="input" id="material" value="leather" maxlength="80" autocomplete="off" placeholder="Ej. denim, silk, leather">
                <button class="button small" id="refreshTextures" type="button">Ver muestras</button>
              </div>
              <p class="hint">Elige un material o una muestra visual. Atelier intentará reproducir su grano, costuras y patrón sobre la prenda, manteniendo el color elegido.</p>
            </div>
            <div class="material-options" id="materialOptions" role="group" aria-label="Materiales sugeridos"></div>
            <div class="texture-grid" id="textureGallery"></div>
            <div class="source-toolbar" style="margin:10px 0 17px">
              <span class="source-name" id="textureNote">Cargando muestras…</span>
              <div style="display:flex;gap:7px">
                <button class="button small" id="moreTextures" type="button">↻ Otras muestras</button>
                <button class="button small" id="noTexture" type="button">Sin muestra</button>
              </div>
            </div>
            <div class="field">
              <label class="field-label" for="extra">Instrucción adicional <span style="color:#999;font-weight:400">(opcional)</span></label>
              <textarea class="input" id="extra" maxlength="500" placeholder="Ej. make the buttons gold"></textarea>
              <p class="hint">Escribe la instrucción en inglés para obtener mejores resultados; se combinará con los cambios de color y material.</p>
            </div>
            <button class="generate" id="generateButton" type="button">✦ &nbsp; Crear mi diseño</button>
            <div class="progress-card" id="progressCard" aria-live="polite">
              <div class="progress-top"><span class="progress-message" id="progressMessage">Preparando…</span><span class="progress-state" id="progressState">EN CURSO</span></div>
              <div class="progress-track"><div class="progress-bar"></div></div>
            </div>
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
                <figure class="result-image"><img id="resultMask" alt="Máscara segmentada"><figcaption>Máscara</figcaption></figure>
                <figure class="result-image"><img id="resultEdited" alt="Edición del modelo"><figcaption>Edición del modelo</figcaption></figure>
              </div>
            </div>
            <div class="result-error" id="resultError" role="alert"></div>
          </dialog>
        </div>
      </div>
      <footer>Atelier · Recolorado de prendas con inteligencia artificial</footer>
    </main>
  </div>
  <script>
    const $ = (selector) => document.querySelector(selector);
    const state = { source: null, textureId: null, activeJob: null, textureTimer: null, segmentation: null };
    const colorHex = {
      black:"#252525", white:"#f7f5ef", red:"#c94843", blue:"#466ca5", "navy blue":"#293650",
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
    function setSource(source, label, previewUrl) {
      if (state.source !== source) clearSegmentation();
      state.source = source;
      $("#sourceName").textContent = label;
      $("#sourceImage").src = previewUrl;
      $("#sourcePreview").classList.add("has-image");
      document.querySelectorAll(".thumb").forEach((el) => el.classList.toggle("selected", el.dataset.id === source));
      clearError();
    }
    function clearSegmentation(message = "") {
      state.segmentation = null;
      const canvas = $("#segmentationOverlay");
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
      $("#sourcePreview").classList.remove("has-segmentation");
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
      segmentation.items.forEach((item, index) => {
        context.globalAlpha = index === segmentation.hovered ? 0.8 : segmentation.selected.has(index) ? 0.6 : 0.25;
        context.drawImage(item.maskImage, 0, 0, canvas.width, canvas.height);
      });
      context.globalAlpha = 1;
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
        button.textContent = `${item.name} · ${Math.round(item.score * 100)}%`;
        button.addEventListener("click", () => toggleGarment(index));
        list.append(button);
      });
    }
    function toggleGarment(index) {
      if (!state.segmentation) return;
      if (state.segmentation.selected.has(index)) state.segmentation.selected.delete(index);
      else state.segmentation.selected.add(index);
      drawSegmentationOverlay();
      updateGarmentOptions();
      $("#segmentationStatus").textContent = state.segmentation.selected.size
        ? `${state.segmentation.selected.size} prenda(s) seleccionada(s). El color y material elegidos se aplicarán a todas.`
        : "Pasa el cursor sobre una prenda de la imagen o elige una abajo.";
    }
    function garmentAt(clientX, clientY) {
      const segmentation = state.segmentation;
      if (!segmentation) return -1;
      const canvas = $("#segmentationOverlay");
      const rect = canvas.getBoundingClientRect();
      const scale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
      const x = Math.floor((clientX - rect.left - (rect.width - canvas.width * scale) / 2) / scale);
      const y = Math.floor((clientY - rect.top - (rect.height - canvas.height * scale) / 2) / scale);
      if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return -1;
      for (let index = segmentation.items.length - 1; index >= 0; index -= 1) {
        const pixels = segmentation.items[index].maskPixels;
        if (pixels.data[(y * canvas.width + x) * 4 + 3] > 0) return index;
      }
      return -1;
    }
    async function detectGarments() {
      clearError();
      if (!state.source) { showError("Elige una imagen antes de detectar las prendas."); return; }
      clearSegmentation("Detectando prendas y afinando sus bordes. Puede tardar un poco…");
      const button = $("#detectGarments");
      button.disabled = true;
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
          return {...item, maskImage, maskPixels:maskContext.getImageData(0, 0, data.width, data.height)};
        }));
        const canvas = $("#segmentationOverlay");
        canvas.width = data.width;
        canvas.height = data.height;
        state.segmentation = {id:data.id, sourceId, items, selected:new Set(), hovered:-1};
        $("#sourcePreview").classList.add("has-segmentation");
        updateGarmentOptions();
        drawSegmentationOverlay();
        const status = $("#segmentationStatus");
        status.textContent = `${items.length} prenda(s) detectada(s). Pasa el cursor por encima para identificarlas y haz clic para seleccionarlas; puedes marcar varias.`;
        status.classList.add("visible");
      } catch (error) {
        clearSegmentation(`No se pudieron detectar las prendas: ${error.message}`);
      } finally {
        button.disabled = false;
      }
    }
    async function loadImages() {
      const gallery = $("#imageGallery");
      gallery.innerHTML = Array.from({length:10}, () => '<div class="thumb"></div>').join("");
      try {
        const {items} = await request("/api/images");
        gallery.innerHTML = "";
        items.forEach((item) => {
          const button = document.createElement("button");
          button.type = "button"; button.className = "thumb"; button.title = item.label; button.dataset.id = item.id;
          const image = document.createElement("img"); image.loading = "lazy"; image.alt = item.label; image.src = item.preview_url;
          button.append(image);
          button.addEventListener("click", () => setSource(item.id, item.label, item.preview_url));
          gallery.append(button);
        });
        if (!state.source && items.length) setSource(items[0].id, items[0].label, items[0].preview_url);
        else document.querySelectorAll(".thumb").forEach((el) => el.classList.toggle("selected", el.dataset.id === state.source));
      } catch (error) { gallery.innerHTML = '<div class="empty-text">No se pudo cargar el catálogo.</div>'; showError(error.message); }
    }
    async function loadTextures() {
      const gallery = $("#textureGallery");
      if ($("#material").value.trim().toLowerCase() === "none") {
        state.textureId = null;
        gallery.innerHTML = '<div class="empty-text">Sin textura ni muestra de referencia.</div>';
        $("#textureNote").textContent = "Textura desactivada";
        return;
      }
      gallery.innerHTML = '<div class="empty-text">Buscando muestras…</div>';
      try {
        const query = new URLSearchParams({material: $("#material").value});
        const data = await request(`/api/textures?${query}`);
        $("#textureNote").textContent = data.note;
        gallery.innerHTML = "";
        if (!data.items.length) gallery.innerHTML = '<div class="empty-text">No hay muestras. Puedes usar el material sin referencia visual.</div>';
        data.items.forEach((item) => {
          const button = document.createElement("button");
          button.type = "button"; button.className = "texture-tile";
          button.classList.toggle("selected", state.textureId === item.id); button.dataset.id = item.id;
          const image = document.createElement("img"); image.loading = "lazy"; image.alt = item.id; image.src = item.preview_url;
          const label = document.createElement("span"); label.textContent = item.id;
          button.append(image, label);
          button.addEventListener("click", () => {
            state.textureId = state.textureId === item.id ? null : item.id;
            document.querySelectorAll(".texture-tile").forEach((el) => el.classList.toggle("selected", el.dataset.id === state.textureId));
          });
          gallery.append(button);
        });
        if (state.textureId && !data.items.some((item) => item.id === state.textureId)) state.textureId = null;
      } catch (error) { gallery.innerHTML = '<div class="empty-text">No se pudieron cargar las muestras.</div>'; $("#textureNote").textContent = error.message; }
    }
    async function loadHistory() {
      const list = $("#historyList");
      try {
        const {items} = await request("/api/history");
        list.innerHTML = "";
        if (!items.length) {
          list.innerHTML = '<div class="empty-text">Aún no hay diseños guardados.<br>Cuando generes uno, aparecerá aquí.</div>';
          return;
        }
        items.forEach((item) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "history-item";
          button.title = `Ver ${item.label}`;
          const name = document.createElement("span");
          name.className = "history-item-name";
          name.textContent = item.label;
          const arrow = document.createElement("span");
          arrow.className = "history-open";
          arrow.textContent = "↗";
          button.append(name, arrow);
          button.addEventListener("click", () => displayHistoryResult(item));
          list.append(button);
        });
      } catch (error) {
        list.innerHTML = "";
        const message = document.createElement("div");
        message.className = "empty-text";
        message.textContent = `No se pudo cargar el historial: ${error.message}`;
        list.append(message);
      }
    }
    function populateOptions(config) {
      $("#colors").innerHTML = config.colors.map((color) => `<option value="${color}">`).join("");
      $("#materialOptions").replaceChildren();
      ["none", ...config.textures.filter((texture) => texture !== "none")].forEach((texture) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "material-choice";
        button.textContent = texture;
        button.classList.toggle("selected", texture === $("#material").value.trim().toLowerCase());
        button.addEventListener("click", () => {
          $("#material").value = texture;
          state.textureId = null;
          document.querySelectorAll(".texture-tile").forEach((tile) => tile.classList.remove("selected"));
          updateMaterialChoices();
          loadTextures();
        });
        $("#materialOptions").append(button);
      });
      $("#swatches").innerHTML = config.colors.filter((color) => colorHex[color]).map((color) => {
        const button = document.createElement("button");
        button.type = "button"; button.className = "swatch"; button.title = color; button.setAttribute("aria-label", color);
        button.style.background = colorHex[color];
        return button.outerHTML;
      }).join("");
      const noColor = document.createElement("button");
      noColor.type = "button";
      noColor.className = "swatch swatch-none";
      noColor.title = "none";
      noColor.setAttribute("aria-label", "Sin color");
      noColor.textContent = "none";
      $("#swatches").append(noColor);
      $("#swatches").querySelectorAll(".swatch").forEach((button) => {
        button.addEventListener("click", () => { $("#color").value = button.title; updateSwatches(); });
      });
    }
    function updateMaterialChoices() {
      const value = $("#material").value.trim().toLowerCase();
      $("#materialOptions").querySelectorAll(".material-choice").forEach((button) => {
        button.classList.toggle("selected", button.textContent.toLowerCase() === value);
      });
    }
    function updateSwatches() {
      $("#swatches").querySelectorAll(".swatch").forEach((button) => button.classList.toggle("selected", button.title === $("#color").value.trim().toLowerCase()));
    }
    async function uploadImage(file) {
      clearError();
      if (!file) return;
      if (file.size > 20 * 1024 * 1024) { showError("La imagen supera el límite de 20 MB."); return; }
      $("#sourceName").textContent = "Cargando imagen…";
      try {
        const response = await fetch(`/api/upload?filename=${encodeURIComponent(file.name)}`, {
          method: "POST", headers: {"Content-Type": file.type || "application/octet-stream"}, body: file
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "No se pudo cargar la imagen.");
        setSource(data.id, data.label, data.preview_url);
      } catch (error) { $("#sourceName").textContent = "No se pudo cargar la imagen"; showError(error.message); }
    }
    function displayResult(job, jobId) {
      const cacheBust = `?v=${Date.now()}`;
      $("#resultHeading").textContent = "Tu resultado";
      document.querySelector(".result-showcase").classList.remove("history-showcase");
      document.querySelectorAll(".result-alternates .result-image").forEach((figure) => figure.classList.remove("hidden"));
      $("#resultOriginal").src = `/api/jobs/${jobId}/image/original${cacheBust}`;
      $("#resultMask").src = `/api/jobs/${jobId}/image/mask${cacheBust}`;
      $("#resultEdited").src = `/api/jobs/${jobId}/image/edited${cacheBust}`;
      $("#resultFinal").src = `/api/jobs/${jobId}/image/final${cacheBust}`;
      $("#downloadButton").href = `/api/jobs/${jobId}/image/final?download=1`;
      $("#resultFilename").value = downloadName(`${job.label || "atelier"}-${jobId.slice(0, 8)}`);
      $("#downloadButton").download = `${$("#resultFilename").value}.png`;
      setDownloadFilename();
      $("#resultMeta").textContent = `${job.garment} · ${job.label || "diseño listo"}`;
      $("#resultError").classList.remove("visible");
      $("#resultsCard").showModal();
    }
    function downloadName(value) {
      return value.replace(/\.png$/i, "").replace(/[<>:"/\\|?*\x00-\x1f]/g, "_").trim().replace(/[. ]+$/g, "").slice(0, 100) || "atelier";
    }
    function displayHistoryResult(item) {
      const cacheBust = `?v=${Date.now()}`;
      const imageUrl = `/api/history/${encodeURIComponent(item.filename)}`;
      $("#resultHeading").textContent = "Diseño anterior";
      document.querySelector(".result-showcase").classList.add("history-showcase");
      document.querySelectorAll(".result-alternates .result-image").forEach((figure) => figure.classList.add("hidden"));
      $("#resultFinal").src = `${imageUrl}${cacheBust}`;
      $("#resultFilename").value = downloadName(item.filename);
      $("#downloadButton").href = `${imageUrl}?download=1`;
      $("#downloadButton").download = `${$("#resultFilename").value}.png`;
      setDownloadFilename();
      $("#resultMeta").textContent = item.label;
      $("#resultError").classList.remove("visible");
      $("#resultsCard").showModal();
    }
    function setDownloadFilename() {
      const name = `${downloadName($("#resultFilename").value)}.png`;
      $("#resultFilename").value = name.slice(0, -4);
      $("#downloadButton").download = name;
      const url = new URL($("#downloadButton").href, location.href);
      url.searchParams.set("download", "1");
      url.searchParams.set("name", name);
      $("#downloadButton").href = url.pathname + url.search;
    }
    $("#resultFilename").addEventListener("input", () => {
      setDownloadFilename();
    });
    async function pollJob(jobId) {
      while (state.activeJob === jobId) {
        const job = await request(`/api/jobs/${jobId}`);
        $("#progressMessage").textContent = job.message || "Preparando…";
        if (job.status === "completed") {
          $("#progressState").textContent = "LISTO";
          $("#progressCard").classList.remove("visible");
          $("#generateButton").disabled = false;
          state.activeJob = null;
          loadHistory();
          displayResult(job, jobId);
          return;
        }
        if (job.status === "failed") {
          $("#progressCard").classList.remove("visible");
          $("#generateButton").disabled = false;
          state.activeJob = null;
          showError(job.error || "No se pudo generar el diseño.");
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 1300));
      }
    }
    async function generate() {
      clearError();
      if (!state.source) { showError("Elige una imagen antes de crear el diseño."); return; }
      const colorValue = $("#color").value.trim();
      const color = colorValue.toLowerCase() === "none" ? "" : colorValue;
      const material = $("#material").value.trim();
      const extra = $("#extra").value.trim();
      if (state.segmentation && state.segmentation.selected.size === 0) {
        showError("Selecciona al menos una de las prendas detectadas.");
        return;
      }
      if (!color && (!material || material.toLowerCase() === "none") && !extra && !state.textureId) {
        showError("Indica un color, material, muestra de textura o instrucción antes de generar.");
        return;
      }
      $("#generateButton").disabled = true;
      $("#progressCard").classList.add("visible");
      $("#progressMessage").textContent = "Preparando el modelo…";
      $("#progressState").textContent = "EN CURSO";
      try {
        const job = await request("/api/generate", {
          method: "POST", headers: {"Content-Type":"application/json"},
          body: JSON.stringify({
            source_id: state.source,
            color,
            texture: material || "none",
            extra,
            texture_id: material.toLowerCase() === "none" ? null : state.textureId,
            segmentation_id: state.segmentation?.id || null,
            garment_indices: state.segmentation ? [...state.segmentation.selected] : null
          })
        });
        state.activeJob = job.id;
        await pollJob(job.id);
      } catch (error) {
        state.activeJob = null;
        $("#progressCard").classList.remove("visible");
        $("#generateButton").disabled = false;
        showError(error.message);
      }
    }
    $("#refreshImages").addEventListener("click", loadImages);
    $("#refreshHistory").addEventListener("click", loadHistory);
    $("#uploadButton").addEventListener("click", () => $("#fileInput").click());
    $("#fileInput").addEventListener("change", (event) => uploadImage(event.target.files[0]));
    $("#refreshTextures").addEventListener("click", loadTextures);
    $("#moreTextures").addEventListener("click", loadTextures);
    $("#detectGarments").addEventListener("click", detectGarments);
    $("#sourcePreview").addEventListener("pointermove", (event) => {
      if (!state.segmentation) return;
      const index = garmentAt(event.clientX, event.clientY);
      if (index === state.segmentation.hovered) return;
      state.segmentation.hovered = index;
      drawSegmentationOverlay();
      $("#segmentationStatus").textContent = index >= 0
        ? `${state.segmentation.items[index].name} · ${Math.round(state.segmentation.items[index].score * 100)}% de confianza`
        : "Pasa el cursor por encima para identificar una prenda; haz clic para seleccionarla.";
    });
    $("#sourcePreview").addEventListener("pointerleave", () => {
      if (!state.segmentation) return;
      state.segmentation.hovered = -1;
      drawSegmentationOverlay();
    });
    $("#sourcePreview").addEventListener("click", (event) => {
      if (!state.segmentation) return;
      const index = garmentAt(event.clientX, event.clientY);
      if (index >= 0) toggleGarment(index);
    });
    $("#noTexture").addEventListener("click", () => {
      state.textureId = null;
      document.querySelectorAll(".texture-tile").forEach((el) => el.classList.remove("selected"));
    });
    $("#material").addEventListener("input", () => {
      updateMaterialChoices();
      if ($("#material").value.trim().toLowerCase() === "none") {
        state.textureId = null;
        document.querySelectorAll(".texture-tile").forEach((tile) => tile.classList.remove("selected"));
      }
      clearTimeout(state.textureTimer);
      state.textureTimer = setTimeout(loadTextures, 450);
    });
    $("#color").addEventListener("input", updateSwatches);
    $("#generateButton").addEventListener("click", generate);
    $("#closeResults").addEventListener("click", () => $("#resultsCard").close());
    $("#resultsCard").addEventListener("click", (event) => {
      if (event.target === $("#resultsCard")) $("#resultsCard").close();
    });
    ["resultOriginal", "resultMask", "resultEdited", "resultFinal"].forEach((id) => {
      $(`#${id}`).addEventListener("error", () => {
        $("#resultError").textContent = "No se pudo cargar una de las imágenes. Comprueba que el servidor siga abierto y vuelve a generar el diseño.";
        $("#resultError").classList.add("visible");
      });
    });
    (async function init() {
      try { populateOptions(await request("/api/config")); }
      catch (error) { showError(error.message); }
      await Promise.all([loadImages(), loadTextures(), loadHistory()]);
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


def get_textures(app_state):
    if app_state.textures is None:
        from src.recolor import load_texture_index

        app_state.textures = load_texture_index(C.PIX2PIX_TEXTURES_DIR)
    return app_state.textures


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


def history_directory():
    return Path(C.PIX2PIX_OUTPUT_DIR)


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
    return {
        "items": [
            {
                "filename": path.name,
                "label": path.stem.removeprefix(C.PIX2PIX_OUTPUT_PREFIX).replace("_", " "),
                "modified": path.stat().st_mtime,
                "preview_url": f"/api/history/{path.name}",
            }
            for path in files[:100]
        ]
    }


@app.get("/api/history/{filename}")
async def history_image(filename: str, download: bool = False, name: str | None = None):
    output_dir = history_directory().resolve()
    path = (output_dir / filename).resolve()
    if (path.parent != output_dir or not filename.startswith(C.PIX2PIX_OUTPUT_PREFIX)
            or path.suffix.lower() != ".png" or not path.is_file()):
        raise HTTPException(status_code=404, detail="No se encontró ese diseño guardado.")
    download_name = safe_download_filename(name, path.name) if download else None
    return FileResponse(path, media_type="image/png", filename=download_name)


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


@app.get("/api/textures")
async def textures(request: Request, material: str = ""):
    from src.recolor import find_textures

    entries, note = find_textures(material, get_textures(request.app.state))
    chosen = random.sample(entries, min(C.PIX2PIX_NUM_TEXTURE_CHOICES, len(entries)))
    return {
        "note": note,
        "items": [
            {
                "id": entry["id"],
                "category": entry["category"],
                "preview_url": f"/api/textures/{entry['id']}/preview",
            }
            for entry in chosen
        ],
    }


@app.get("/api/textures/{texture_id}/preview")
async def texture_preview(texture_id: str, request: Request):
    texture = next((entry for entry in get_textures(request.app.state)
                    if entry["id"] == texture_id), None)
    if texture is None:
        raise HTTPException(status_code=404, detail="La muestra de textura no está disponible.")
    return FileResponse(texture["path"])


def run_generation(app_state, job_id, source, color, texture, extra, texture_path,
                  segmentation, garment_indices):
    with job_lock:
        jobs[job_id]["status"] = "running"

    def progress(message):
        with job_lock:
            jobs[job_id]["message"] = message

    try:
        import torch
        from src.recolor import generate as generate_recolor

        dataset = get_dataset(app_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = generate_recolor(
            source, color, texture, extra, texture_path, dataset, device, progress,
            segmentation=segmentation, garment_indices=garment_indices,
        )
        if result is None:
            raise RuntimeError("No se detectó ninguna prenda clara. Prueba con otra imagen.")

        output_dir = Path(C.PIX2PIX_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"\W+", "_", result["label"]).strip("_")[:45] or "diseno"
        final_path = output_dir / f"{C.PIX2PIX_OUTPUT_PREFIX}{safe_label}_{job_id[:8]}.png"
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
                "label": result["label"],
                "images": paths,
                "saved_path": str(final_path),
            })
    except Exception as exc:
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
    texture_id = (
        options.texture_id
        if texture_name.lower() != C.PIX2PIX_NO_TEXTURE
        else None
    )
    extra = options.extra.strip()
    if (not color and texture_name.lower() == C.PIX2PIX_NO_TEXTURE
            and not extra and not texture_id):
        raise HTTPException(
            status_code=400,
            detail="Indica un color, material, muestra de textura o instrucción antes de generar.",
        )
    texture_path = None
    if texture_id:
        texture = next((entry for entry in get_textures(request.app.state)
                        if entry["id"] == texture_id), None)
        if texture is None:
            raise HTTPException(status_code=400, detail="La muestra de textura elegida ya no existe.")
        texture_path = texture["path"]

    segmentation = None
    garment_indices = None
    if options.segmentation_id:
        segmentation = request.app.state.segmentations.get(options.segmentation_id)
        if segmentation is None or segmentation["source_id"] != options.source_id:
            raise HTTPException(status_code=400, detail="Vuelve a detectar las prendas de la imagen seleccionada.")
        garment_indices = options.garment_indices
        if not garment_indices:
            raise HTTPException(status_code=400, detail="Selecciona al menos una prenda detectada.")
        if any(index < 0 or index >= len(segmentation["garments"]) for index in garment_indices):
            raise HTTPException(status_code=400, detail="Una de las prendas seleccionadas ya no está disponible.")
    elif options.garment_indices is not None:
        raise HTTPException(status_code=400, detail="Primero detecta las prendas de la imagen.")

    with job_lock:
        if request.app.state.segmenting:
            raise HTTPException(status_code=409, detail="Espera a que termine la detección de prendas.")
        if any(job["status"] in {"queued", "running"} for job in jobs.values()):
            raise HTTPException(status_code=409, detail="Ya hay una generación en curso. Espera a que termine.")
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
        )
    return {"id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    with job_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No se encontró esta generación.")
        return {
            "id": job_id,
            **{key: value for key, value in job.items() if key not in {"images", "saved_path"}},
        }


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
