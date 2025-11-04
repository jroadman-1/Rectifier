# app.py
import os, io, sys, uuid, shutil, subprocess, json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

app = FastAPI(title="Four-Dot Rectifier")

TMP = Path("/tmp")
TMP.mkdir(parents=True, exist_ok=True)

# ------------- Helpers (lightweight; heavy imports happen inside functions) -------------

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

def _run_cli_rectifier(input_path: Path, output_path: Path, *,
                       width_mm: float, height_mm: float,
                       dpi: Optional[float],
                       corner_frac: Optional[float],
                       polarity: Optional[str],
                       enforce_axes: bool,
                       mask_path: Optional[Path] = None):
    """
    Calls your existing CLI script in AUTO mode.
    Ensure 'rectify_four_dots_improved.py' is in the repo root.
    """
    script = "rectify_four_dots_improved.py"
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

# --------------------------------- Routes ---------------------------------

@app.get("/")
def health():
    return {"ok": True, "routes": ["/ui", "/rectify (POST)", "/ui-manual", "/rectify_manual (POST)"]}

# ---------- AUTO mode UI ----------
@app.get("/ui", response_class=HTMLResponse)
def ui_auto():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Four-Dot Rectifier — Auto</title>
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
const frm = document.getElementById('frm');
frm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(frm);
  fd.set('enforce_axes', frm.enforce_axes.checked ? 'true' : 'false');
  const btn = frm.querySelector('button');
  btn.disabled = true; btn.textContent = 'Processing…';
  try {
    const res = await fetch('/rectify', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('out').innerHTML =
      `<h2>Result</h2><a download="rectified.png" href="${url}">Download PNG</a><br><br><img src="${url}">`;
  } catch (err) {
    document.getElementById('out').innerHTML = '<p style="color:#b00020;">' + err.message + '</p>';
  } finally {
    btn.disabled = false; btn.textContent = 'Rectify';
  }
});
</script>
</body>
</html>
    """

# ---------- AUTO mode API (calls your script) ----------
@app.post("/rectify", response_class=FileResponse)
async def rectify_auto(
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
        _run_cli_rectifier(
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

# ---------- MANUAL picker UI (canvas with two-click zoom) ----------
@app.get("/ui-manual", response_class=HTMLResponse)
def ui_manual():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Four-Dot Rectifier — Manual Picker</title>
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; margin:2rem; max-width:1000px}
  #c{max-width:100%; height:auto; cursor: crosshair; border:1px solid #ddd}
  #pts{margin:.5rem 0; color:#333}
  button{padding:.5rem 1rem}
</style>
</head>
<body>
<h1>Four-Dot Rectifier — Manual Picker</h1>
<form id="frm">
  <div><input id="file" type="file" name="image" accept="image/*" required></div>
  <p style="color:#666">Click a rough spot to open a zoom window; click again to confirm the dot. Repeat 4 times. Undo/Reset if needed.</p>
  <canvas id="c"></canvas>
  <div id="pts">Points: (none)</div>
  <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:.75rem;">
    <label>Width (mm) <input type="number" step="0.1" name="width_mm" value="381.0" required></label>
    <label>Height (mm) <input type="number" step="0.1" name="height_mm" value="228.6" required></label>
    <label>DPI <input type="number" step="1" name="dpi" value="300"></label>
    <label>Margin (mm) <input type="number" step="0.1" name="margin_mm" value="10.0"></label>
    <label><input type="checkbox" name="enforce_axes" checked> Enforce axes</label>
  </div>
  <div style="margin-top:0.75rem;">
    <button type="button" id="undo">Undo</button>
    <button type="button" id="reset">Reset</button>
    <button type="submit" id="go" disabled>Rectify</button>
  </div>
</form>
<div id="out" style="margin-top:1rem;"></div>

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
let zoomSize = 150; // half-size of zoom square (screen px)

function draw() {
  if (!naturalW) { canvas.width = 800; canvas.height = 450; ctx.clearRect(0,0,800,450); return; }
  const maxW = Math.min(1000, window.innerWidth - 64);
  const scale = Math.min(1, maxW / naturalW);
  const cw = Math.round(naturalW * scale);
  const ch = Math.round(naturalH * scale);
  canvas.width = cw; canvas.height = ch;
  ctx.clearRect(0,0,cw,ch);
  ctx.drawImage(img, 0, 0, cw, ch);

  // existing points
  ctx.lineWidth = 2;
  points.forEach((p, i) => {
    const sx = p.x * scale, sy = p.y * scale;
    ctx.strokeStyle = "#ffcc00";
    ctx.beginPath(); ctx.arc(sx, sy, 8, 0, 2*Math.PI); ctx.stroke();
    ctx.fillStyle = "#e53935";
    ctx.beginPath(); ctx.arc(sx, sy, 4, 0, 2*Math.PI); ctx.fill();
    ctx.fillStyle = "#000"; ctx.font = "14px system-ui";
    ctx.fillText(String(i+1), sx + 10, sy - 10);
  });

  // zoom window preview
  if (zoomMode && zoomCenter) {
    const zw = zoomSize*2, zh = zoomSize*2;
    const sx = zoomCenter.x * scale - zoomSize;
    const sy = zoomCenter.y * scale - zoomSize;

    ctx.strokeStyle = "#00f"; ctx.lineWidth = 1; ctx.strokeRect(sx, sy, zw, zh);

    // draw magnified inset at bottom-left
    const srcW = (zw/scale), srcH = (zh/scale);
    const srcX = Math.max(0, zoomCenter.x - srcW/2);
    const srcY = Math.max(0, zoomCenter.y - srcH/2);
    ctx.drawImage(img, srcX, srcY, srcW, srcH,
                  20, canvas.height - zh - 20, zw, zh);
    ctx.strokeStyle = "#000";
    ctx.strokeRect(20, canvas.height - zh - 20, zw, zh);
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

  if (!zoomMode) { // first click -> open zoom
    zoomMode = true; zoomCenter = {x, y}; draw(); return;
  }
  // second click -> confirm
  points.push({x, y}); zoomMode = false; zoomCenter = null; draw();
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

# ---------- MANUAL mode API ----------
@app.post("/rectify_manual")
async def rectify_manual(
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
