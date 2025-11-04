# app_improved.py
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

# ---------- Main UI with scrollable canvas ----------
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
    margin: 0;
    color: #1565c0;
  }
  .canvas-container {
    position: relative;
    border: 2px solid #ddd;
    overflow: auto;
    max-height: 70vh;
    background: #fafafa;
    margin-bottom: 1.5rem;
    display: none;
  }
  .canvas-container.active {
    display: block;
  }
  #canvas {
    display: block;
    cursor: crosshair;
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
  .zoom-controls {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }
  .zoom-btn {
    padding: 0.5rem 0.75rem;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1.125rem;
    font-weight: bold;
    transition: background 0.2s;
  }
  .zoom-btn:hover {
    background: #f0f0f0;
  }
  .zoom-btn:active {
    background: #e0e0e0;
  }
  .zoom-label {
    font-size: 0.875rem;
    color: #666;
    min-width: 60px;
    text-align: center;
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
      <p><strong>Instructions:</strong> Load an image, then click on each of the 4 corner dots in any order. The image is displayed at 2x size for precision - scroll to navigate.</p>
    </div>

    <div class="canvas-container" id="canvasContainer">
      <canvas id="canvas"></canvas>
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

      <div class="control-group">
        <label>Image Scale</label>
        <div class="zoom-controls">
          <button type="button" class="zoom-btn" id="zoomOut">−</button>
          <span class="zoom-label" id="zoomLabel">200%</span>
          <button type="button" class="zoom-btn" id="zoomIn">+</button>
        </div>
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
const pointsDisplay = document.getElementById('pointsDisplay');
const undoBtn = document.getElementById('undoBtn');
const resetBtn = document.getElementById('resetBtn');
const submitBtn = document.getElementById('submitBtn');
const resultDiv = document.getElementById('result');
const zoomInBtn = document.getElementById('zoomIn');
const zoomOutBtn = document.getElementById('zoomOut');
const zoomLabel = document.getElementById('zoomLabel');

let img = new Image();
let naturalW = 0, naturalH = 0;
let points = [];
let scale = 2.0; // Default 2x scale

const COLORS = ['#e53935', '#1e88e5', '#43a047', '#fb8c00']; // Red, Blue, Green, Orange

function updateZoomLabel() {
  zoomLabel.textContent = `${Math.round(scale * 100)}%`;
}

function drawCanvas() {
  if (!naturalW) {
    canvas.width = 800;
    canvas.height = 600;
    ctx.clearRect(0, 0, 800, 600);
    return;
  }

  const displayW = Math.round(naturalW * scale);
  const displayH = Math.round(naturalH * scale);
  
  canvas.width = displayW;
  canvas.height = displayH;
  
  ctx.clearRect(0, 0, displayW, displayH);
  ctx.drawImage(img, 0, 0, displayW, displayH);

  // Draw points
  points.forEach((p, i) => {
    const x = p.x * scale;
    const y = p.y * scale;
    
    // Outer circle (white border)
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(x, y, 12, 0, 2 * Math.PI);
    ctx.stroke();
    
    // Inner circle (colored)
    ctx.fillStyle = COLORS[i];
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, 2 * Math.PI);
    ctx.fill();
    
    // Label
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 14px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(i + 1), x, y);
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
    drawCanvas();
  };
  img.src = url;
});

canvas.addEventListener('click', (e) => {
  if (!naturalW || points.length >= 4) return;
  
  const rect = canvas.getBoundingClientRect();
  const scrollLeft = canvasContainer.scrollLeft;
  const scrollTop = canvasContainer.scrollTop;
  
  const canvasX = e.clientX - rect.left + scrollLeft;
  const canvasY = e.clientY - rect.top + scrollTop;
  
  // Convert from display coordinates to natural image coordinates
  const x = canvasX / scale;
  const y = canvasY / scale;
  
  points.push({ x, y });
  drawCanvas();
});

undoBtn.addEventListener('click', () => {
  points.pop();
  drawCanvas();
});

resetBtn.addEventListener('click', () => {
  points = [];
  drawCanvas();
});

zoomInBtn.addEventListener('click', () => {
  if (scale < 4.0) {
    scale += 0.5;
    drawCanvas();
    updateZoomLabel();
  }
});

zoomOutBtn.addEventListener('click', () => {
  if (scale > 0.5) {
    scale -= 0.5;
    drawCanvas();
    updateZoomLabel();
  }
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

updateZoomLabel();
</script>
</body>
</html>
    """

# ---------- Rectify API ----------
@app.post("/rectify")
async def rectify(
    image: UploadFile = File(...),
    points: str = Form(...),             # JSON [{x,y}, ...] length=4
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
        im = np.array(pil)[:, :, ::-1]  # to BGR

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
