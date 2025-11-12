# app_maps_style_v5.py
import os, io, sys, uuid, shutil, subprocess, json, math
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

app = FastAPI(title="Four-Dot Rectifier with Alignment")

TMP = Path("/tmp")
TMP.mkdir(parents=True, exist_ok=True)

# ------------- Helpers -------------

def _order_corners_tl_tr_br_bl_np(pts_xy):
    """
    Order 4 points as TL, TR, BR, BL using a robust algorithm.
    Works with both portrait and landscape orientations and any click order.
    """
    import numpy as np
    pts = np.asarray(pts_xy, dtype=np.float32)
    
    # Method: Sort points by Y, split into top/bottom pairs, then sort each by X
    # This works regardless of image orientation
    
    # Sort all points by Y coordinate (top to bottom)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    
    # Split into top two and bottom two
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]
    
    # Sort each pair by X coordinate (left to right)
    top_sorted = top_two[np.argsort(top_two[:, 0])]
    bottom_sorted = bottom_two[np.argsort(bottom_two[:, 0])]
    
    # Assign corners
    tl = top_sorted[0]     # leftmost of top two
    tr = top_sorted[1]     # rightmost of top two
    bl = bottom_sorted[0]  # leftmost of bottom two
    br = bottom_sorted[1]  # rightmost of bottom two
    
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

def _align_image_by_edge(image_bgr, pt1, pt2, direction='horizontal'):
    """
    Rotate image so the line between pt1 and pt2 is horizontal or vertical.
    
    Args:
        image_bgr: OpenCV image (BGR)
        pt1: First point (x, y)
        pt2: Second point (x, y)
        direction: 'horizontal' or 'vertical'
    
    Returns:
        Rotated image
    """
    import numpy as np, cv2
    
    # Calculate angle of the line
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # Note: In image coordinates, Y increases downward, but atan2 gives us
    # the angle in standard math coordinates. When we use this with OpenCV's
    # rotation, we need to negate it to get the correct visual rotation.
    angle_deg = -angle_deg
    
    # Adjust angle based on desired direction
    if direction == 'horizontal':
        # Calculate rotation to make line horizontal (parallel to x-axis)
        # Two options: rotate to 0° or 180°
        rot_to_0 = -angle_deg
        rot_to_180 = 180 - angle_deg
        
        # Normalize angles to [-180, 180] range
        rot_to_0 = ((rot_to_0 + 180) % 360) - 180
        rot_to_180 = ((rot_to_180 + 180) % 360) - 180
        
        # Choose the rotation with smaller absolute value (minimal rotation)
        if abs(rot_to_0) <= abs(rot_to_180):
            rotation_angle = rot_to_0
        else:
            rotation_angle = rot_to_180
    else:  # vertical
        # Calculate rotation to make line vertical (parallel to y-axis)
        # Two options: rotate to 90° or -90°
        rot_to_90 = 90 - angle_deg
        rot_to_minus_90 = -90 - angle_deg
        
        # Normalize angles to [-180, 180] range
        rot_to_90 = ((rot_to_90 + 180) % 360) - 180
        rot_to_minus_90 = ((rot_to_minus_90 + 180) % 360) - 180
        
        # Choose the rotation with smaller absolute value (minimal rotation)
        if abs(rot_to_90) <= abs(rot_to_minus_90):
            rotation_angle = rot_to_90
        else:
            rotation_angle = rot_to_minus_90
    
    # Get image dimensions
    h, w = image_bgr.shape[:2]
    center = (w / 2, h / 2)
    
    # Calculate rotation matrix
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # Calculate new bounding dimensions to avoid cropping
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Adjust translation to keep entire image
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    # Perform rotation
    rotated = cv2.warpAffine(image_bgr, M, (new_w, new_h), 
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    
    return rotated

# --------------------------------- Routes ---------------------------------

@app.get("/")
def health():
    return {"ok": True, "routes": ["/ui (GET)", "/rectify (POST)", "/align (POST)"]}

# ---------- Main UI with two-step workflow ----------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Four-Dot Rectifier with Alignment</title>
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
  h2 {
    margin: 2rem 0 1rem 0;
    color: #1a1a1a;
    font-size: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e0e0e0;
  }
  .step {
    margin-bottom: 2rem;
  }
  .step.hidden {
    display: none;
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
  
  #canvasWrapper, #alignCanvasWrapper {
    position: relative;
    display: inline-block;
    transform-origin: 0 0;
  }
  
  #canvas, #alignCanvas {
    display: block;
  }
  
  #markersContainer, #alignMarkersContainer {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }
  
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
  .radio-group {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
  }
  .radio-group label {
    display: flex;
    align-items: center;
    cursor: pointer;
  }
  .radio-group input[type="radio"] {
    margin-right: 0.5rem;
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
  .btn-success {
    background: #43a047;
    color: white;
  }
  .btn-success:hover:not(:disabled) {
    background: #388e3c;
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
    flex-wrap: wrap;
  }
  .result-section {
    background: #f9f9f9;
    padding: 1.5rem;
    border-radius: 8px;
    margin-top: 2rem;
  }
  .result-section img {
    max-width: 100%;
    height: auto;
    border: 1px solid #ddd;
    border-radius: 4px;
    margin: 1rem 0;
  }
  .result-buttons {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1rem;
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
  <h1>Four-Dot Rectifier with Alignment</h1>
  
  <!-- STEP 1: RECTIFICATION -->
  <div class="step" id="step1">
    <h2>Step 1: Rectify Image</h2>
    
    <form id="rectifyForm">
      <div class="upload-section">
        <input type="file" id="fileInput" name="image" accept="image/*" required>
      </div>

      <div class="instructions">
        <p><strong>Instructions:</strong> Hold <kbd>Shift</kbd> and click to place each of the 4 corner dots (works with portrait or landscape photos, any click order).</p>
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

    <div id="rectifyResult"></div>
  </div>

  <!-- STEP 2: ALIGNMENT (hidden initially) -->
  <div class="step hidden" id="step2">
    <h2>Step 2: Align Object Edge (Optional)</h2>
    
    <div class="instructions">
      <p><strong>Instructions:</strong> Hold <kbd>Shift</kbd> and click 2 points on an object edge you want to align.</p>
      <p><strong>Choose alignment:</strong> Horizontal (left-to-right) or Vertical (top-to-bottom)</p>
    </div>

    <div class="canvas-container" id="alignCanvasContainer">
      <div id="alignCanvasWrapper">
        <canvas id="alignCanvas"></canvas>
        <div id="alignMarkersContainer"></div>
      </div>
      <div class="zoom-overlay">
        <button type="button" id="alignZoomInBtn" title="Zoom in">+</button>
        <button type="button" id="alignZoomOutBtn" title="Zoom out">−</button>
      </div>
      <div class="zoom-info" id="alignZoomInfo">Zoom: 100%</div>
    </div>

    <div class="points-display" id="alignPointsDisplay">
      Points selected: 0 / 2
    </div>

    <div class="radio-group">
      <label>
        <input type="radio" name="alignDirection" value="horizontal" checked>
        Horizontal
      </label>
      <label>
        <input type="radio" name="alignDirection" value="vertical">
        Vertical
      </label>
    </div>

    <div class="action-buttons">
      <button type="button" class="btn-secondary" id="alignUndoBtn">Undo Last Point</button>
      <button type="button" class="btn-danger" id="alignResetBtn">Reset Points</button>
      <button type="button" class="btn-success" id="applyAlignBtn" disabled>Apply Alignment</button>
      <button type="button" class="btn-secondary" id="skipAlignBtn">Skip Alignment</button>
    </div>

    <div id="alignResult"></div>
  </div>
</div>

<script>
// ========== STEP 1: RECTIFICATION ==========
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
const zoomInBtn = document.getElementById('zoomInBtn');
const zoomOutBtn = document.getElementById('zoomOutBtn');
const zoomInfo = document.getElementById('zoomInfo');
const rectifyResult = document.getElementById('rectifyResult');

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
const MAX_SCALE = 8.0;
const ZOOM_STEP = 0.08;
const WHEEL_ZOOM_IN_STEP = 0.025;
const WHEEL_ZOOM_OUT_STEP = 0.05;
const DRAG_THRESHOLD = 10;

let minScale = 0.1; // Will be updated based on image fit

let rectifiedImageBlob = null; // Store for step 2

function updateZoomInfo() {
  zoomInfo.textContent = `Zoom: ${Math.round(scale * 100)}%`;
}

function fitImageToContainer() {
  const containerW = canvasContainer.clientWidth;
  const containerH = canvasContainer.clientHeight;
  const scaleW = containerW / naturalW;
  const scaleH = containerH / naturalH;
  const fitScale = Math.min(scaleW, scaleH, 1.0) * 0.9;
  
  // Set this as the minimum zoom level
  minScale = fitScale;
  
  scale = fitScale;
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
    : Math.max(scale - step, minScale);
  
  if (oldScale === newScale) return;
  
  const canvasX = (mouseX - panX) / oldScale;
  const canvasY = (mouseY - panY) / oldScale;
  
  scale = newScale;
  
  panX = mouseX - canvasX * newScale;
  panY = mouseY - canvasY * newScale;
  
  drawCanvas();
}

fileInput.addEventListener('change', () => {
  points = [];
  rectifyResult.innerHTML = '';
  document.getElementById('step2').classList.add('hidden');
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

document.addEventListener('keydown', (e) => {
  if (e.key === 'Shift' && !shiftHeld) {
    shiftHeld = true;
    if (canvasContainer.classList.contains('active')) {
      canvasContainer.classList.add('crosshair');
    }
    if (alignCanvasContainer.classList.contains('active')) {
      alignCanvasContainer.classList.add('crosshair');
    }
  }
});

document.addEventListener('keyup', (e) => {
  if (e.key === 'Shift') {
    shiftHeld = false;
    canvasContainer.classList.remove('crosshair');
    alignCanvasContainer.classList.remove('crosshair');
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
  
  const rect = canvasContainer.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  
  zoomAtPoint(true, mouseX, mouseY);
});

canvasContainer.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (!naturalW) return;
  
  const rect = canvasContainer.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  
  const zoomIn = e.deltaY < 0;
  const step = zoomIn ? WHEEL_ZOOM_IN_STEP : WHEEL_ZOOM_OUT_STEP;
  zoomAtPoint(zoomIn, mouseX, mouseY, step);
});

zoomInBtn.addEventListener('click', () => {
  if (!naturalW) return;
  const centerX = canvasContainer.clientWidth / 2;
  const centerY = canvasContainer.clientHeight / 2;
  zoomAtPoint(true, centerX, centerY);
});

zoomOutBtn.addEventListener('click', () => {
  if (!naturalW) return;
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

document.getElementById('rectifyForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (points.length !== 4) return;
  
  const formData = new FormData(e.target);
  formData.set('enforce_axes', e.target.enforce_axes.checked ? 'true' : 'false');
  formData.set('points', JSON.stringify(points.map(p => ({ x: p.x, y: p.y }))));
  
  submitBtn.disabled = true;
  submitBtn.textContent = 'Processing...';
  rectifyResult.innerHTML = '';
  
  try {
    const response = await fetch('/rectify', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText);
    }
    
    rectifiedImageBlob = await response.blob();
    const url = URL.createObjectURL(rectifiedImageBlob);
    
    rectifyResult.innerHTML = `
      <div class="result-section">
        <h3>Rectified Image</h3>
        <img src="${url}" alt="Rectified result" id="rectifiedImg">
        <div class="result-buttons">
          <a href="${url}" download="rectified.png" class="btn-primary" style="display:inline-block; text-decoration:none;">
            Download Rectified Image
          </a>
          <button type="button" class="btn-success" id="startAlignBtn">+ Align Object Edge</button>
        </div>
      </div>
    `;
    
    document.getElementById('startAlignBtn').addEventListener('click', startAlignment);
    
  } catch (err) {
    rectifyResult.innerHTML = `<div class="error"><strong>Error:</strong> ${err.message}</div>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Rectify Image';
  }
});

// ========== STEP 2: ALIGNMENT ==========
const alignCanvas = document.getElementById('alignCanvas');
const alignCtx = alignCanvas.getContext('2d');
const alignCanvasContainer = document.getElementById('alignCanvasContainer');
const alignCanvasWrapper = document.getElementById('alignCanvasWrapper');
const alignMarkersContainer = document.getElementById('alignMarkersContainer');
const alignPointsDisplay = document.getElementById('alignPointsDisplay');
const alignUndoBtn = document.getElementById('alignUndoBtn');
const alignResetBtn = document.getElementById('alignResetBtn');
const applyAlignBtn = document.getElementById('applyAlignBtn');
const skipAlignBtn = document.getElementById('skipAlignBtn');
const alignZoomInBtn = document.getElementById('alignZoomInBtn');
const alignZoomOutBtn = document.getElementById('alignZoomOutBtn');
const alignZoomInfo = document.getElementById('alignZoomInfo');
const alignResult = document.getElementById('alignResult');

let alignImg = new Image();
let alignNaturalW = 0, alignNaturalH = 0;
let alignPoints = [];
let alignScale = 1.0;
let alignPanX = 0, alignPanY = 0;
let alignIsDragging = false;
let alignDragStartX = 0, alignDragStartY = 0;
let alignDragMoved = false;

const ALIGN_COLORS = ['#e53935', '#1e88e5'];
let alignMinScale = 0.1; // Will be updated based on image fit

function startAlignment() {
  document.getElementById('step2').classList.remove('hidden');
  document.getElementById('step2').scrollIntoView({ behavior: 'smooth' });
  
  alignPoints = [];
  alignResult.innerHTML = '';
  
  const url = URL.createObjectURL(rectifiedImageBlob);
  alignImg.onload = () => {
    alignNaturalW = alignImg.naturalWidth;
    alignNaturalH = alignImg.naturalHeight;
    alignCanvasContainer.classList.add('active');
    fitAlignImageToContainer();
    drawAlignCanvas();
  };
  alignImg.src = url;
}

function fitAlignImageToContainer() {
  const containerW = alignCanvasContainer.clientWidth;
  const containerH = alignCanvasContainer.clientHeight;
  const scaleW = containerW / alignNaturalW;
  const scaleH = containerH / alignNaturalH;
  const fitScale = Math.min(scaleW, scaleH, 1.0) * 0.9;
  
  // Set this as the minimum zoom level
  alignMinScale = fitScale;
  
  alignScale = fitScale;
  alignPanX = (containerW - alignNaturalW * alignScale) / 2;
  alignPanY = (containerH - alignNaturalH * alignScale) / 2;
}

function updateAlignZoomInfo() {
  alignZoomInfo.textContent = `Zoom: ${Math.round(alignScale * 100)}%`;
}

function drawAlignCanvas() {
  if (!alignNaturalW) return;

  alignCanvas.width = alignNaturalW;
  alignCanvas.height = alignNaturalH;
  
  alignCanvasWrapper.style.width = alignNaturalW + 'px';
  alignCanvasWrapper.style.height = alignNaturalH + 'px';
  alignCanvasWrapper.style.transform = `translate(${alignPanX}px, ${alignPanY}px) scale(${alignScale})`;
  
  alignCtx.clearRect(0, 0, alignNaturalW, alignNaturalH);
  alignCtx.drawImage(alignImg, 0, 0, alignNaturalW, alignNaturalH);
  
  updateAlignMarkers();
  updateAlignZoomInfo();
}

function updateAlignMarkers() {
  alignMarkersContainer.innerHTML = '';
  
  alignPoints.forEach((p, i) => {
    const marker = document.createElement('div');
    marker.className = 'marker';
    marker.style.left = p.x + 'px';
    marker.style.top = p.y + 'px';
    
    marker.innerHTML = `
      <div class="marker-outer"></div>
      <div class="marker-inner" style="background-color: ${ALIGN_COLORS[i]}"></div>
      <div class="marker-label">${i + 1}</div>
    `;
    
    alignMarkersContainer.appendChild(marker);
  });
  
  updateAlignPointsDisplay();
}

function updateAlignPointsDisplay() {
  const count = alignPoints.length;
  if (count === 0) {
    alignPointsDisplay.textContent = 'Points selected: 0 / 2';
  } else {
    const coords = alignPoints.map((p, i) => 
      `Point ${i+1}: (${Math.round(p.x)}, ${Math.round(p.y)})`
    ).join('  •  ');
    alignPointsDisplay.textContent = `Points selected: ${count} / 2  •  ${coords}`;
  }
  
  applyAlignBtn.disabled = (count !== 2);
}

function screenToAlignCanvas(screenX, screenY) {
  const rect = alignCanvasWrapper.getBoundingClientRect();
  const x = (screenX - rect.left) / alignScale;
  const y = (screenY - rect.top) / alignScale;
  return { x, y };
}

function alignZoomAtPoint(zoomIn, mouseX, mouseY, step = ZOOM_STEP) {
  const oldScale = alignScale;
  const newScale = zoomIn 
    ? Math.min(alignScale + step, MAX_SCALE)
    : Math.max(alignScale - step, alignMinScale);
  
  if (oldScale === newScale) return;
  
  const canvasX = (mouseX - alignPanX) / oldScale;
  const canvasY = (mouseY - alignPanY) / oldScale;
  
  alignScale = newScale;
  
  alignPanX = mouseX - canvasX * newScale;
  alignPanY = mouseY - canvasY * newScale;
  
  drawAlignCanvas();
}

alignCanvasContainer.addEventListener('mousedown', (e) => {
  alignIsDragging = true;
  alignDragMoved = false;
  alignDragStartX = e.clientX - alignPanX;
  alignDragStartY = e.clientY - alignPanY;
  alignCanvasContainer.classList.add('grabbing');
});

alignCanvasContainer.addEventListener('mousemove', (e) => {
  if (!alignIsDragging) return;
  
  const deltaX = Math.abs(e.clientX - alignPanX - alignDragStartX);
  const deltaY = Math.abs(e.clientY - alignPanY - alignDragStartY);
  
  if (deltaX > DRAG_THRESHOLD || deltaY > DRAG_THRESHOLD) {
    alignDragMoved = true;
  }
  
  alignPanX = e.clientX - alignDragStartX;
  alignPanY = e.clientY - alignDragStartY;
  drawAlignCanvas();
});

alignCanvasContainer.addEventListener('mouseup', (e) => {
  alignCanvasContainer.classList.remove('grabbing');
  
  if (!alignDragMoved && shiftHeld && alignNaturalW && alignPoints.length < 2) {
    const coords = screenToAlignCanvas(e.clientX, e.clientY);
    if (coords.x >= 0 && coords.x <= alignNaturalW && coords.y >= 0 && coords.y <= alignNaturalH) {
      alignPoints.push(coords);
      updateAlignMarkers();
    }
  }
  
  alignIsDragging = false;
});

alignCanvasContainer.addEventListener('mouseleave', () => {
  alignIsDragging = false;
  alignCanvasContainer.classList.remove('grabbing');
});

alignCanvasContainer.addEventListener('dblclick', (e) => {
  if (!alignNaturalW) return;
  
  const rect = alignCanvasContainer.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  
  alignZoomAtPoint(true, mouseX, mouseY);
});

alignCanvasContainer.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (!alignNaturalW) return;
  
  const rect = alignCanvasContainer.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  
  const zoomIn = e.deltaY < 0;
  const step = zoomIn ? WHEEL_ZOOM_IN_STEP : WHEEL_ZOOM_OUT_STEP;
  alignZoomAtPoint(zoomIn, mouseX, mouseY, step);
});

alignZoomInBtn.addEventListener('click', () => {
  if (!alignNaturalW) return;
  const centerX = alignCanvasContainer.clientWidth / 2;
  const centerY = alignCanvasContainer.clientHeight / 2;
  alignZoomAtPoint(true, centerX, centerY);
});

alignZoomOutBtn.addEventListener('click', () => {
  if (!alignNaturalW) return;
  const centerX = alignCanvasContainer.clientWidth / 2;
  const centerY = alignCanvasContainer.clientHeight / 2;
  alignZoomAtPoint(false, centerX, centerY);
});

alignUndoBtn.addEventListener('click', () => {
  alignPoints.pop();
  updateAlignMarkers();
});

alignResetBtn.addEventListener('click', () => {
  alignPoints = [];
  updateAlignMarkers();
});

skipAlignBtn.addEventListener('click', () => {
  // Just show the rectified image as final
  const url = URL.createObjectURL(rectifiedImageBlob);
  alignResult.innerHTML = `
    <div class="result-section">
      <h3>Final Image (No Alignment Applied)</h3>
      <img src="${url}" alt="Final result">
      <div class="result-buttons">
        <a href="${url}" download="final.png" class="btn-primary" style="display:inline-block; text-decoration:none;">
          Download Final Image
        </a>
      </div>
    </div>
  `;
  alignResult.scrollIntoView({ behavior: 'smooth' });
});

applyAlignBtn.addEventListener('click', async () => {
  if (alignPoints.length !== 2) return;
  
  const direction = document.querySelector('input[name="alignDirection"]:checked').value;
  
  const formData = new FormData();
  formData.append('image', rectifiedImageBlob, 'rectified.png');
  formData.append('points', JSON.stringify(alignPoints.map(p => ({ x: p.x, y: p.y }))));
  formData.append('direction', direction);
  
  applyAlignBtn.disabled = true;
  applyAlignBtn.textContent = 'Processing...';
  alignResult.innerHTML = '';
  
  try {
    const response = await fetch('/align', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText);
    }
    
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    
    alignResult.innerHTML = `
      <div class="result-section">
        <h3>Aligned Image</h3>
        <img src="${url}" alt="Aligned result">
        <div class="result-buttons">
          <a href="${url}" download="aligned.png" class="btn-primary" style="display:inline-block; text-decoration:none;">
            Download Aligned Image
          </a>
        </div>
      </div>
    `;
    
    alignResult.scrollIntoView({ behavior: 'smooth' });
    
  } catch (err) {
    alignResult.innerHTML = `<div class="error"><strong>Error:</strong> ${err.message}</div>`;
  } finally {
    applyAlignBtn.disabled = false;
    applyAlignBtn.textContent = 'Apply Alignment';
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

# ---------- Align API ----------
@app.post("/align")
async def align(
    image: UploadFile = File(...),
    points: str = Form(...),
    direction: str = Form(...),
):
    try:
        import numpy as np, cv2
        from PIL import Image

        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        im = np.array(pil)[:, :, ::-1]

        pts_list = json.loads(points)
        if not (isinstance(pts_list, list) and len(pts_list) == 2):
            raise HTTPException(status_code=400, detail="Provide exactly 2 points as [{x,y}, ...].")
        
        pt1 = (pts_list[0]["x"], pts_list[0]["y"])
        pt2 = (pts_list[1]["x"], pts_list[1]["y"])
        
        if direction not in ['horizontal', 'vertical']:
            raise HTTPException(status_code=400, detail="Direction must be 'horizontal' or 'vertical'.")

        aligned = _align_image_by_edge(im, pt1, pt2, direction)
        
        ok, buf = cv2.imencode(".png", aligned)
        if not ok:
            raise HTTPException(status_code=500, detail="PNG encode failed")
        return Response(content=buf.tobytes(), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
