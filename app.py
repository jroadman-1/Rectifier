# app.py
import os, io, sys, uuid, shutil, subprocess, json
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

app = FastAPI(title="Four-Dot Rectifier")

TMP = Path("/tmp")
TMP.mkdir(parents=True, exist_ok=True)

# ---------- Utilities ----------

def order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order 4 (x,y) points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def warp_by_corners(image_bgr: np.ndarray,
                    src_pts_xy: np.ndarray,
                    width_mm: float,
                    height_mm: float,
                    dpi: Optional[float] = 300.0,
                    margin_mm: float = 10.0,
                    enforce_axes: bool = True) -> np.ndarray:
    """Perspective-rectify using 4 source points to a known width×height (mm)."""
    px_per_mm = (dpi / 25.4) if dpi else 4.0
    W_rect = int(round(width_mm * px_per_mm))
    H_rect = int(round(height_mm * px_per_mm))
    M = int(round(margin_mm * px_per_mm))
    W = W_rect + 2 * M
    H = H_rect + 2 * M

    src = order_corners_tl_tr_br_bl(src_pts_xy.astype(np.float32))
    dst = np.array([
        [M,         M        ],   # TL
        [M+W_rect,  M        ],   # TR
        [M+W_rect,  M+H_rect ],   # BR
        [M,         M+H_rect ],   # BL
    ], dtype=np.float32)

    Hmat = cv2.getPerspectiveTransform(src, dst)
    rect = cv2.warpPerspective(image_bgr, Hmat, (W, H), flags=cv2.INTER_CUBIC)

    if enforce_axes:
        rect = cv2.resize(rect, (W, H), interpolation=cv2.INTER_CUBIC)

    return rect

def run_cli_rectifier(input_path: Path, output_path: Path, *,
                      width_mm: float, height_mm: float,
                      dpi: Optional[float],
                      corner_frac: Optional[float],
                      polarity: Optional[str],
                      enforce_axes: bool,
                      mask_path: Optional[Path] = None):
    """
    Call your existing CLI script (auto mode) safely.
    Adjust flags here to match your rectify_four_dots_improved.py.
    """
    script = "rectify_four_dots_improved.py"  # must be present in the repo root
    cmd = [
        sys.executable, script,
        "--input", str(input_path),
        "--output", str(output_path),
        "--width-mm", str(width_mm),
        "--height-mm", str(height_mm),
    ]
    if dpi is not None:
        cmd += ["--dpi", str(dpi)]
    if enforce_axes:
        cmd += ["--enforce-axes"]
    if corner_frac is not None:
        cmd += ["--corner-frac", str(corner_frac)]
    if polarity:
        cmd += ["--polarity", polarity]
    if mask_path is not None:
        cmd += ["--mask", str(mask_path)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Rectifier failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

# ---------- Routes ----------

@app.get("/")
def health():
    return {"ok": True, "routes": ["/rectify (POST)", "/ui", "/ui-manual", "/rectify_manual (POST)"]}

# ---- Auto mode API: calls your existing script ----
@app.post("/rectify", response_class=FileResponse)
async def rectify(
    image: UploadFile = File(..., description="Photo with 4 corner dots"),
    width_mm: float = Form(...),
    height_mm: float = Form(...),
    dpi: Optional[float] = Form(300),
    corner_frac: Optional[float] = Form(0.22),
    polarity: Optional[str] = Form("dark"),
    enforce_axes: bool = Form(True),
    mask: UploadFile | None = File(None)
):
    job = TMP / f"job_{uuid.uuid4().hex}"
    job.mkdir(parents=True, exist_ok=True)
    try:
        in_path = job / "input.jpg"
        with open(in_path, "wb") as f:
            f.write(await image.read())

        mask_path = None
        if mask is not None:
            mask_path = job / "mask.png"
            with open(mask_path, "wb") as f:
                f.write(await mask.read())

        out_path = job / "rectified.png"
        run_cli_rectifier(
            in_path, out_path,
            width_mm=width_mm, height_mm=height_mm, dpi=dpi,
            corner_frac=corner_frac, polarity=polarity,
            enforce_axes=enforce_axes, mask_path=mask_path
        )
        if not out_path.exists():
            raise HTTPException(status_code=500, detail="No output produced.")
        return FileResponse(str(out_path), media_type="image/png", filename="rectified.png")
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        shutil.rmtree(job, ignore_errors=True)

# ---- Simple browser UI for auto mode ----
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Four-Dot Rectifier (Auto)</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; margin:2rem; max-width:900px}
    fieldset{border:1px solid #ddd; padding:1rem}
    .row{display:flex; gap:1rem; flex-wrap:wrap}
    #out img{max-width:100%; height:auto; border:1px solid #ddd}
  </style>
</head>
<body>
  <h1>Four-Dot Rectifier — Auto</h1>
  <form id="frm">
    <fieldset>
      <legend>Upload</legend>
      <input type="file" name="image" accept="image/*" required />
    </fieldset>
    <fieldset class="row">
      <label>Width (mm)<br><input type="number" step="0.1" name="width_mm" value="381.0" required></label>
      <label>Height (mm)<br><input type="number" step="0.1" name="height_mm" value="228.6" required></label>
      <label>DPI<br><input type="number" step="1" name="dpi" value="300"></label>
      <label>Corner frac<br><input type="number" step="0.01" min="0.10" max="0.40" name="corner_frac" value="0.22"></label>
      <label>Polarity<br>
        <select name="polarity">
          <option value="dark" selected>dark (black dots)</option>
          <option value="light">light (white dots)</option>
        </select>
      </label>
      <label><input type="checkbox" name="enforce_axes" checked> Enforce axes</label>
    </fieldset>
    <button type="submit">Rectify</button>
  </form>
  <div id="out" style="margin-top:1.5rem;"></div>

<script>
const fileInput = document.getElementById('file');
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const ptsDiv = document.getElementById('pts');
const undoBtn = document.getElementById('undo');
const resetBtn = document.getElementById('reset');
const goBtn = document.getElementById('go');
const outDiv = document.getElementById('out');

let img = new Image();
let naturalW = 0, naturalH = 0;
let points = [];

let zoomMode = false;
let zoomCenter = null;
let zoomFactor = 4;   // how much to enlarge local region
let zoomSize = 150;   // half-size of zoom window in px (on screen)

function draw() {
  if (!naturalW) { canvas.width = 800; canvas.height = 450; ctx.clearRect(0,0,800,450); return; }
  const maxW = Math.min(1000, window.innerWidth - 64);
  const scale = Math.min(1, maxW / naturalW);
  const cw = Math.round(naturalW * scale);
  const ch = Math.round(naturalH * scale);
  canvas.width = cw; canvas.height = ch;
  ctx.clearRect(0,0,cw,ch);
  ctx.drawImage(img, 0, 0, cw, ch);

  ctx.lineWidth = 2;
  points.forEach((p, i) => {
    const sx = p.x * scale, sy = p.y * scale;
    ctx.strokeStyle = "#ffcc00";
    ctx.beginPath(); ctx.arc(sx, sy, 8, 0, 2*Math.PI); ctx.stroke();
    ctx.fillStyle = "#e53935";
    ctx.beginPath(); ctx.arc(sx, sy, 4, 0, 2*Math.PI); ctx.fill();
    ctx.fillStyle = "#000";
    ctx.font = "14px system-ui";
    ctx.fillText(String(i+1), sx + 10, sy - 10);
  });

  // draw zoom window if active
  if (zoomMode && zoomCenter) {
    const zx = zoomCenter.x, zy = zoomCenter.y;
    const zw = zoomSize * 2, zh = zoomSize * 2;
    const sx = zx * scale - zoomSize, sy = zy * scale - zoomSize;
    ctx.strokeStyle = "#00f";
    ctx.lineWidth = 1;
    ctx.strokeRect(sx, sy, zw, zh);
    ctx.drawImage(
      img,
      Math.max(0, zx - (zoomSize / scale)),
      Math.max(0, zy - (zoomSize / scale)),
      (zoomSize * 2) / scale,
      (zoomSize * 2) / scale,
      20, canvas.height - zw - 20, zw, zh
    );
    ctx.strokeStyle = "#000";
    ctx.strokeRect(20, canvas.height - zw - 20, zw, zh);
  }

  ptsDiv.textContent = points.length
    ? ("Points: " + points.map(p => `(${p.x.toFixed(1)}, ${p.y.toFixed(1)})`).join("  "))
    : "Points: (none)";
  goBtn.disabled = (points.length !== 4);
}

fileInput.addEventListener('change', () => {
  points = []; outDiv.innerHTML = "";
  const f = fileInput.files[0]; if (!f) return;
  const url = URL.createObjectURL(f);
  img.onload = () => { naturalW = img.naturalWidth; naturalH = img.naturalHeight; draw(); };
  img.src = url;
});

canvas.addEventListener('click', (e) => {
  if (!naturalW) return;
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  const scale = canvas.width / naturalW;
  const x = sx / scale, y = sy / scale;

  // first click -> enter zoom mode
  if (!zoomMode) {
    zoomMode = true;
    zoomCenter = {x, y};
    draw();
    return;
  }

  // second click inside zoom window -> confirm point
  if (zoomMode && zoomCenter) {
    points.push({x, y});
    zoomMode = false;
    zoomCenter = null;
    draw();
  }
});

undoBtn.addEventListener('click', () => { points.pop(); draw(); });
resetBtn.addEventListener('click', () => { points = []; zoomMode=false; draw(); });

document.getElementById('frm').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (points.length !== 4) return;
  const fd = new FormData(e.target);
  fd.set('enforce_axes', e.target.enforce_axes.checked ? 'true' : 'false');
  fd.set('points', JSON.stringify(points.map(p => ({x:p.x, y:p.y}))));
  const btn = document.getElementById('go'); btn.disabled = true; btn.textContent = 'Processing…';
  outDiv.innerHTML = "";
  try {
    const res = await fetch('/rectify_manual', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    outDiv.innerHTML = `<h2>Result</h2><a download="rectified.png" href="${url}">Download PNG</a><br><br><img src="${url}">`;
  } catch (err) {
    outDiv.innerHTML = '<p style="color:#b00020;">' + err.message + '</p>';
  } finally {
    btn.disabled = false; btn.textContent = 'Rectify';
  }
});
window.addEventListener('resize', draw);
</script>
</body>
</html>
    """

# ---- Manual rectification API (uses points from /ui-manual) ----
@app.post("/rectify_manual")
async def rectify_manual(
    image: UploadFile = File(...),
    points: str = Form(...),             # JSON string: [{x,y}, ...] length 4
    width_mm: float = Form(...),
    height_mm: float = Form(...),
    dpi: Optional[float] = Form(300),
    margin_mm: Optional[float] = Form(10.0),
    enforce_axes: bool = Form(True),
):
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        im = np.array(pil)[:, :, ::-1]  # to BGR
        pts_list = json.loads(points)
        if not (isinstance(pts_list, list) and len(pts_list) == 4):
            raise HTTPException(status_code=400, detail="Provide exactly 4 points.")
        src = np.array([[p["x"], p["y"]] for p in pts_list], dtype=np.float32)
        rect = warp_by_corners(im, src, width_mm, height_mm, dpi=dpi,
                               margin_mm=margin_mm or 0.0, enforce_axes=enforce_axes)
        ok, buf = cv2.imencode(".png", rect)
        if not ok: raise HTTPException(status_code=500, detail="PNG encode failed")
        return Response(content=buf.tobytes(), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
