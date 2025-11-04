# app_overlay.py
import os, io, sys, uuid, shutil, subprocess, json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

app = FastAPI(title="Four-Dot Rectifier")

TMP = Path("/tmp")
TMP.mkdir(parents=True, exist_ok=True)

# ------------- Helpers -------------

def _order_corners_tl_tr_br_bl_np(pts_xy):
    """Order 4 points as TL, TR, BR, BL using sums and diffs."""
    import numpy as np
    pts = np.asarray(pts_xy, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _warp_by_corners(image_bgr, src_pts_xy, width_mm, height_mm, dpi=300.0, margin_mm=10.0, enforce_axes=True):
    """Perspective-rectify to known mm dimensions."""
    import numpy as np, cv2
    px_per_mm = (dpi / 25.4) if dpi else 4.0
    W_rect = int(round(width_mm * px_per_mm))
    H_rect = int(round(height_mm * px_per_mm))
    M = int(round((margin_mm or 0.0) * px_per_mm))
    W = W_rect + 2 * M
    H = H_rect + 2 * M

    src = _order_corners_tl_tr_br_bl_np(src_pts_xy)
    dst = np.array(
        [[M,        M       ],
         [M+W_rect, M       ],
         [M+W_rect, M+H_rect],
         [M,        M+H_rect]], dtype=np.float32
    )
    Hmat = cv2.getPerspectiveTransform(src, dst)
    rect = cv2.warpPerspective(image_bgr, Hmat, (W, H), flags=cv2.INTER_CUBIC)
    if enforce_axes:
        rect = cv2.resize(rect, (W, H), interpolation=cv2.INTER_CUBIC)
    return rect

# --------------------------------- Routes ---------------------------------

@app.get("/")
def health():
    return {"ok": True, "routes": ["/ui (GET)", "/rectify (POST)"]}

# ---------- Main UI with Google Maps-style pan and zoom ----------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Four-Dot Rectifier</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    margin: 0;
    padding: 1.5rem;
    background: #f5f5f5;
  }
  .container {
    max-width: 1400px;
    margin: 0 auto;
    background: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }
  h1 {
    margin: 0 0 1.5rem 0;
    color: #1a1a1a;
    font-size: 1.75rem;
  }
  .upload-section {
    margin-bottom: 1.5rem;
    padding: 1rem;
    background: #f9f9f9;
    border: 2px dashed #ddd;
    border-radius: 4px;
  }
  .upload-section input[type="file"] {
    display: block;
    width: 100%;
    padding: 0.5rem;
  }
  .instructions {
    background: #e3f2fd;
    padding: 1rem;
    margin-bottom: 1.5rem;
    border-radius: 4px;
    border-left: 4px solid #2196f3;
  }
  .instructions p {
    margin: 0.25rem 0;
    color: #1565c0;
    font-size: 0.9rem;
  }
  .canvas-container {
    position: relative;
    border: 2px solid #ddd;
    overflow: hidden;
    height: 70vh;
    background: #fafafa;
    margin-bottom: 1.5rem;
    display: none;
    cursor: grab;
  }
  .canvas-container.active {
    display: block;
  }
  .canvas-container.grabbing {
    cursor: grabbing;
  }
  .canvas-container.crosshair {
    cursor: crosshair;
  }
  
  kbd {
    background: #f0f0f0;
    border: 1px solid #ccc;
    border-radius: 3px;
    padding: 2px 6px;
    font-family: monospace;
    font-size: 0.85em;
  }
  
  /* Wrapper for canvas and markers to align them */
  #canvasWrapper {
    position: relative;
    display: inline-block;
    transform-origin: 0 0;
  }
  
  #canvas {
    display: block;
  }
  
  #markersContainer {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }
  
  /* Marker overlay styles */
  .marker {
    position: absolute;
    width: 60px;
    height: 60px;
    margin-left: -30px;
    margin-top: -30px;
    pointer-events: none;
    z-index: 1000;
    animation: markerPulse 0.5s ease-out;
  }
  
  @keyframes markerPulse {
    0% { transform: scale(0); }
    50% { transform: scale(1.3); }
    100% { transform: scale(1); }
  }
  
  .marker-outer {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 50px;
    height: 50px;
    margin-left: -25px;
    margin-top: -25px;
    border-radius: 50%;
    border: 4px solid white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }
  
  .marker-inner {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 40px;
    height: 40px;
    margin-left: -20px;
    margin-top: -20px;
    border-radius: 50%;
  }
  
  .marker-label {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: white;
    font-weight: bold;
    font-size: 20px;
    text-shadow: 0 1px 3px rgba(0,0,0,0.5);
  }
  
  /* Zoom controls overlay */
  .zoom-overlay {
    position: absolute;
    top: 10px;
    right: 10px;
    background: white;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    display: flex;
    flex-direction: column;
    z-index: 1001;
  }
  
  .zoom-overlay button {
    width: 40px;
    height: 40px;
    border: none;
    background: white;
    cursor: pointer;
    font-size: 20px;
    font-weight: bold;
    color: #666;
    transition: background 0.2s;
  }
  
  .zoom-overlay button:hover {
    background: #f0f0f0;
  }
  
  .zoom-overlay button:first-child {
    border-radius: 4px 4px 0 0;
  }
  
  .zoom-overlay button:last-child {
    border-radius: 0 0 4px 4px;
    border-top: 1px solid #ddd;
  }
  
  .zoom-info {
    position: absolute;
    bottom: 10px;
    left: 10px;
    background: rgba(255, 255, 255, 0.9);
    padding: 0.5rem;
    border-radius: 4px;
    font-size: 0.875rem;
    color: #666;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }
  
  .points-display {
    padding: 0.75rem;
    background: #f5f5f5;
    border-radius: 4px;
    margin-bottom: 1rem;
    font-family: monospace;
    color: #333;
  }
  .controls {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    align-items: flex-end;
    margin-bottom: 1rem;
  }
  .control-group {
    display: flex;
    flex-direction: column;
  }
  .control-group label {
    font-size: 0.875rem;
    color: #666;
    margin-bottom: 0.25rem;
  }
  .control-group input[type="number"] {
    padding: 0.5rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 1rem;
    width: 120px;
  }
  .control-group input[type="checkbox"] {
    margin-right: 0.5rem;
  }
  .checkbox-label {
    display: flex;
    align-items: center;
    font-size: 0.875rem;
    color: #333;
  }
  button {
    padding: 0.625rem 1.5rem;
    border: none;
    border-radius: 4px;
    font-size: 1rem;
    cursor: pointer;
    transition: background 0.2s;
  }
  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .btn-primary {
    background: #2196f3;
    color: white;
  }
  .btn-primary:hover:not(:disabled) {
    background: #1976d2;
  }
  .btn-secondary {
    background: #757575;
    color: white;
  }
  .btn-secondary:hover:not(:disabled) {
    background: #616161;
  }
  .btn-danger {
    background: #f44336;
    color: white;
  }
  .btn-danger:hover:not(:disabled) {
    background: #d32f2f;
  }
  .action-buttons {
    display: flex;
    gap: 0.75rem;
  }
  #result {
    margin-top: 2rem;
  }
  #result h2 {
    margin: 0 0 1rem 0;
    color: #1a1a1a;
  }
  #result img {
    max-width: 100%;
    height: auto;
    border: 1px solid #ddd;
    border-radius: 4px;
  }
  .error {
    padding: 1rem;
    background: #ffebee;
    color: #c62828;
    border-radius: 4px;
    margin-top: 1rem;
  }
</style>
</head>
<body>
<div class="container">
  <h1>Four-Dot Rectifier</h1>
  
  <form id="form">
    <div class="upload-section">
      <input type="file" id="fileInput" name="image" accept="image/*" required>
    </div>

    <div class="instructions">
      <p><strong>Instructions:</strong> Hold <kbd>Shift</kbd> and click to place each of the 4 corner dots.</p>
      <p><strong>Navigation:</strong> Mouse wheel to zoom (smooth & slow) • Click and drag to pan • Double-click to zoom in</p>
    </div>

    <div class="canvas-container" id="canvasContainer">
      <div id="canvasWrapper">
        <canvas id="canvas"></canvas>
        <div id="markersContainer"></div>
      </div>
      <div class="zoom-overlay">
        <button type="button" id="zoomInBtn" title="Zoom in">+</button>
        <button type="button" id="zoomOutBtn" title="Zoom out">−</button>
      </div>
      <div class="zoom-info" id="zoomInfo">Zoom: 100%</div>
    </div>

    <div class="points-display" id="pointsDisplay">
      Points selected: 0 / 4
    </div>

    <div class="controls">
      <div class="control-group">
        <label>Width (mm)</label>
        <input type="number" step="0.1" name="width_mm" value="381.0" required>
      </div>
      
      <div class="control-group">
        <label>Height (mm)</label>
        <input type="number" step="0.1" name="height_mm" value="228.6" required>
      </div>
      
      <div class="control-group">
        <label>DPI</label>
        <input type="number" step="1" name="dpi" value="300">
      </div>
      
      <div class="control-group">
        <label>Margin (mm)</label>
        <input type="number" step="0.1" name="margin_mm" value="10.0">
      </div>
      
      <div class="control-group">
        <label class="checkbox-label">
          <input type="checkbox" name="enforce_axes" checked>
          Enforce axes
        </label>
      </div>
    </div>

    <div class="action-buttons">
      <button type="button" class="btn-secondary" id="undoBtn">Undo Last Point</button>
      <button type="button" class="btn-danger" id="resetBtn">Reset All</button>
      <button type="submit" class="btn-primary" id="submitBtn" disabled>Rectify Image</button>
    </div>
  </form>

  <div id="result"></div>
</div>

<script>
const fileInput = document.getElementById('fileInput');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const canvasContainer = document.getElementById('canvasContainer');
const canvasWrapper = document.getElementById('canvasWrapper');
const markersContainer = document.getElementById('markersContainer');
const pointsDisplay = document.getElementById('pointsDisplay');
const undoBtn = document.getElementById('undoBtn');
const resetBtn = document.getElementById('resetBtn');
const submitBtn = document.getElementById('submitBtn');
const resultDiv = document.getElementById('result');
const zoomInBtn = document.getElementById('zoomInBtn');
const zoomOutBtn = document.getElementById('zoomOutBtn');
const zoomInfo = document.getElementById('zoomInfo');

let img = new Image();
let naturalW = 0, naturalH = 0;
let points = [];
let scale = 1.0;
let panX = 0, panY = 0;
let isDragging = false;
let dragStartX = 0, dragStartY = 0;
let dragMoved = false;
let shiftHeld = false;

const COLORS = ['#e53935', '#1e88e5', '#43a047', '#fb8c00'];
const MIN_SCALE = 0.1;
const MAX_SCALE = 8.0;
const ZOOM_STEP = 0.08; // Slower button zoom
const WHEEL_ZOOM_STEP = 0.025; // Much slower wheel zoom (was 0.05)
const DRAG_THRESHOLD = 10; // Pixels of movement before it's considered a drag (increased from 3)

function updateZoomInfo() {
  zoomInfo.textContent = `Zoom: ${Math.round(scale * 100)}%`;
}

function fitImageToContainer() {
  const containerW = canvasContainer.clientWidth;
  const containerH = canvasContainer.clientHeight;
  const scaleW = containerW / naturalW;
  const scaleH = containerH / naturalH;
  scale = Math.min(scaleW, scaleH, 1.0) * 0.9; // 90% to add padding
  panX = (containerW - naturalW * scale) / 2;
  panY = (containerH - naturalH * scale) / 2;
}

function drawCanvas() {
  if (!naturalW) {
    canvas.width = 800;
    canvas.height = 600;
    ctx.clearRect(0, 0, 800, 600);
    return;
  }

  canvas.width = naturalW;
  canvas.height = naturalH;
  
  canvasWrapper.style.width = naturalW + 'px';
  canvasWrapper.style.height = naturalH + 'px';
  canvasWrapper.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
  
  ctx.clearRect(0, 0, naturalW, naturalH);
  ctx.drawImage(img, 0, 0, naturalW, naturalH);
  
  updateMarkers();
  updateZoomInfo();
}

function updateMarkers() {
  markersContainer.innerHTML = '';
  
  points.forEach((p, i) => {
    const marker = document.createElement('div');
    marker.className = 'marker';
    marker.style.left = p.x + 'px';
    marker.style.top = p.y + 'px';
    
    marker.innerHTML = `
      <div class="marker-outer"></div>
      <div class="marker-inner" style="background-color: ${COLORS[i]}"></div>
      <div class="marker-label">${i + 1}</div>
    `;
    
    markersContainer.appendChild(marker);
  });
  
  updatePointsDisplay();
}

function updatePointsDisplay() {
  const count = points.length;
  if (count === 0) {
    pointsDisplay.textContent = 'Points selected: 0 / 4';
  } else {
    const coords = points.map((p, i) => 
      `Point ${i+1}: (${Math.round(p.x)}, ${Math.round(p.y)})`
    ).join('  •  ');
    pointsDisplay.textContent = `Points selected: ${count} / 4  •  ${coords}`;
  }
  
  submitBtn.disabled = (count !== 4);
}

function screenToCanvas(screenX, screenY) {
  const rect = canvasWrapper.getBoundingClientRect();
  const x = (screenX - rect.left) / scale;
  const y = (screenY - rect.top) / scale;
  return { x, y };
}

function zoomAtPoint(zoomIn, mouseX, mouseY, step = ZOOM_STEP) {
  const oldScale = scale;
  const newScale = zoomIn 
    ? Math.min(scale + step, MAX_SCALE)
    : Math.max(scale - step, MIN_SCALE);
  
  if (oldScale === newScale) return;
  
  // Get the point on the canvas under the mouse BEFORE zoom
  const canvasX = (mouseX - panX) / oldScale;
  const canvasY = (mouseY - panY) / oldScale;
  
  // Apply new scale
  scale = newScale;
  
  // Adjust pan so the same canvas point stays under the mouse AFTER zoom
  panX = mouseX - canvasX * newScale;
  panY = mouseY - canvasY * newScale;
  
  drawCanvas();
}

fileInput.addEventListener('change', () => {
  points = [];
  resultDiv.innerHTML = '';
  const file = fileInput.files[0];
  if (!file) return;
  
  const url = URL.createObjectURL(file);
  img.onload = () => {
    naturalW = img.naturalWidth;
    naturalH = img.naturalHeight;
    canvasContainer.classList.add('active');
    fitImageToContainer();
    drawCanvas();
  };
  img.src = url;
});

// Shift key tracking for crosshair cursor
document.addEventListener('keydown', (e) => {
  if (e.key === 'Shift' && !shiftHeld) {
    shiftHeld = true;
    if (canvasContainer.classList.contains('active')) {
      canvasContainer.classList.add('crosshair');
    }
  }
});

document.addEventListener('keyup', (e) => {
  if (e.key === 'Shift') {
    shiftHeld = false;
    canvasContainer.classList.remove('crosshair');
  }
});

canvasContainer.addEventListener('mousedown', (e) => {
  isDragging = true;
  dragMoved = false;
  dragStartX = e.clientX - panX;
  dragStartY = e.clientY - panY;
  canvasContainer.classList.add('grabbing');
});

canvasContainer.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  
  const deltaX = Math.abs(e.clientX - panX - dragStartX);
  const deltaY = Math.abs(e.clientY - panY - dragStartY);
  
  if (deltaX > DRAG_THRESHOLD || deltaY > DRAG_THRESHOLD) {
    dragMoved = true;
  }
  
  panX = e.clientX - dragStartX;
  panY = e.clientY - dragStartY;
  drawCanvas();
});

canvasContainer.addEventListener('mouseup', (e) => {
  canvasContainer.classList.remove('grabbing');
  
  // Only place marker if: not dragging, shift was held, and we have room for more points
  if (!dragMoved && shiftHeld && naturalW && points.length < 4) {
    const coords = screenToCanvas(e.clientX, e.clientY);
    if (coords.x >= 0 && coords.x <= naturalW && coords.y >= 0 && coords.y <= naturalH) {
      points.push(coords);
      updateMarkers();
    }
  }
  
  isDragging = false;
});

canvasContainer.addEventListener('mouseleave', () => {
  isDragging = false;
  canvasContainer.classList.remove('grabbing');
});

canvasContainer.addEventListener('dblclick', (e) => {
  if (!naturalW) return;
  
  // Get mouse position relative to container
  const rect = canvasContainer.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  
  zoomAtPoint(true, mouseX, mouseY);
});

canvasContainer.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (!naturalW) return;
  
  // Get mouse position relative to container
  const rect = canvasContainer.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  
  const zoomIn = e.deltaY < 0;
  zoomAtPoint(zoomIn, mouseX, mouseY, WHEEL_ZOOM_STEP);
});

zoomInBtn.addEventListener('click', () => {
  if (!naturalW) return;
  // Zoom toward center of visible viewport
  const centerX = canvasContainer.clientWidth / 2;
  const centerY = canvasContainer.clientHeight / 2;
  zoomAtPoint(true, centerX, centerY);
});

zoomOutBtn.addEventListener('click', () => {
  if (!naturalW) return;
  // Zoom from center of visible viewport
  const centerX = canvasContainer.clientWidth / 2;
  const centerY = canvasContainer.clientHeight / 2;
  zoomAtPoint(false, centerX, centerY);
});

undoBtn.addEventListener('click', () => {
  points.pop();
  updateMarkers();
});

resetBtn.addEventListener('click', () => {
  points = [];
  updateMarkers();
});

document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (points.length !== 4) return;
  
  const formData = new FormData(e.target);
  formData.set('enforce_axes', e.target.enforce_axes.checked ? 'true' : 'false');
  formData.set('points', JSON.stringify(points.map(p => ({ x: p.x, y: p.y }))));
  
  submitBtn.disabled = true;
  submitBtn.textContent = 'Processing...';
  resultDiv.innerHTML = '';
  
  try {
    const response = await fetch('/rectify', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText);
    }
    
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    
    resultDiv.innerHTML = `
      <h2>Rectified Image</h2>
      <a href="${url}" download="rectified.png" class="btn-primary" style="display:inline-block; text-decoration:none; margin-bottom:1rem;">
        Download PNG
      </a>
      <br>
      <img src="${url}" alt="Rectified result">
    `;
  } catch (err) {
    resultDiv.innerHTML = `<div class="error"><strong>Error:</strong> ${err.message}</div>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Rectify Image';
  }
});
</script>
</body>
</html>
    """

# ---------- Rectify API ----------
@app.post("/rectify")
async def rectify(
    image: UploadFile = File(...),
    points: str = Form(...),
    width_mm: float = Form(...),
    height_mm: float = Form(...),
    dpi: Optional[float] = Form(300),
    margin_mm: Optional[float] = Form(10.0),
    enforce_axes: bool = Form(True),
):
    try:
        import numpy as np, cv2
        from PIL import Image

        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        im = np.array(pil)[:, :, ::-1]

        pts_list = json.loads(points)
        if not (isinstance(pts_list, list) and len(pts_list) == 4):
            raise HTTPException(status_code=400, detail="Provide exactly 4 points as [{x,y}, ...].")
        src = np.array([[p["x"], p["y"]] for p in pts_list], dtype=np.float32)

        rect = _warp_by_corners(im, src, width_mm, height_mm, dpi=dpi,
                                margin_mm=margin_mm or 0.0, enforce_axes=enforce_axes)
        ok, buf = cv2.imencode(".png", rect)
        if not ok:
            raise HTTPException(status_code=500, detail="PNG encode failed")
        return Response(content=buf.tobytes(), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
