import os, uuid, shutil, subprocess, sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

app = FastAPI(title="Four-Dot Rectifier")

from fastapi.responses import HTMLResponse

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Four-Dot Rectifier</title>
  <style>
    body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin:2rem; max-width:900px}
    fieldset{border:1px solid #ddd; padding:1rem}
    label{display:block; margin:.5rem 0 .25rem}
    input[type="number"]{width:10rem}
    .row{display:flex; gap:1rem; flex-wrap:wrap}
    #out{margin-top:1.5rem}
    img{max-width:100%; height:auto; border:1px solid #ddd}
    .muted{color:#666; font-size:.9rem}
  </style>
</head>
<body>
  <h1>Four-Dot Rectifier</h1>
  <form id="frm">
    <fieldset>
      <legend>Upload</legend>
      <label>Photo (with 4 corner dots)</label>
      <input type="file" name="image" accept="image/*" required />
    </fieldset>
    <fieldset class="row">
      <div>
        <label>Width (mm)</label>
        <input type="number" step="0.1" name="width_mm" value="381.0" required />
      </div>
      <div>
        <label>Height (mm)</label>
        <input type="number" step="0.1" name="height_mm" value="228.6" required />
      </div>
      <div>
        <label>DPI (optional)</label>
        <input type="number" step="1" name="dpi" value="300" />
      </div>
      <div>
        <label>Corner frac</label>
        <input type="number" step="0.01" min="0.10" max="0.40" name="corner_frac" value="0.22" />
      </div>
      <div>
        <label>Polarity</label>
        <select name="polarity">
          <option value="dark" selected>dark (black dots)</option>
          <option value="light">light (white dots)</option>
        </select>
      </div>
      <div>
        <label><input type="checkbox" name="enforce_axes" checked /> Enforce axes</label>
      </div>
    </fieldset>
    <p class="muted">Tip: Width/height for your 11×17″ target with 1″ inset are 381.0 mm × 228.6 mm.</p>
    <button type="submit">Rectify</button>
  </form>

  <div id="out"></div>

<script>
const frm = document.getElementById('frm');
frm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(frm);
  // Convert checkbox -> true/false string so FastAPI sees it
  fd.set('enforce_axes', frm.enforce_axes.checked ? 'true' : 'false');

  const btn = frm.querySelector('button');
  btn.disabled = true; btn.textContent = 'Processing…';

  try {
    const res = await fetch('/rectify', { method: 'POST', body: fd });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error('Server error: ' + txt);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('out').innerHTML =
      `<h2>Result</h2><a download="rectified.png" href="${url}">Download PNG</a><br/><br/><img src="${url}">`;
  } catch (err) {
    document.getElementById('out').innerHTML =
      '<p style="color:#b00020;">' + err.message + '</p>';
  } finally {
    btn.disabled = false; btn.textContent = 'Rectify';
  }
});
</script>
</body>
</html>
    """

# Where we do temp work on Render (ephemeral disk)
TMP = Path("/tmp")
TMP.mkdir(exist_ok=True, parents=True)

def run_script(input_path: Path, output_path: Path, *,
               width_mm: float, height_mm: float, dpi: float | None,
               corner_frac: float | None, polarity: str | None,
               enforce_axes: bool, mask_path: Path | None):
    """
    Calls your existing CLI script in automatic mode (no GUI).
    Adjust the flags here to match your script’s options.
    """
    cmd = [sys.executable, "rectify_four_dots_improved.py",
           "--input", str(input_path),
           "--output", str(output_path),
           "--width-mm", str(width_mm),
           "--height-mm", str(height_mm)]
    if dpi:
        cmd += ["--dpi", str(dpi)]
    if enforce_axes:
        cmd += ["--enforce-axes"]
    # Optional flags if your script supports them (safe to include if present):
    if corner_frac:
        cmd += ["--corner-frac", str(corner_frac)]
    if polarity:
        cmd += ["--polarity", polarity]
    if mask_path is not None:
        cmd += ["--mask", str(mask_path)]
    # Never use --manual on a server (no GUI available)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"rectifier failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

@app.post("/rectify", response_class=FileResponse)
async def rectify(
    image: UploadFile = File(..., description="Photo containing the four corner dots"),
    width_mm: float = Form(..., description="Real width between left/right dots (mm)"),
    height_mm: float = Form(..., description="Real height between top/bottom dots (mm)"),
    dpi: float | None = Form(None, description="Output DPI (optional)"),
    corner_frac: float | None = Form(0.22, description="Corner ROI fraction (e.g., 0.18–0.30)"),
    polarity: str | None = Form("dark", description="dot color: 'dark' or 'light'"),
    enforce_axes: bool = Form(True, description="Force exact width×height in output"),
    mask: UploadFile | None = File(None, description="Optional mask PNG (white=ignore, black=search)")
):
    # save uploads to /tmp
    job = TMP / f"job_{uuid.uuid4().hex}"
    job.mkdir(parents=True, exist_ok=True)
    try:
        in_path = job / "input.jpg"
        with open(in_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        mask_path = None
        if mask is not None:
            mask_path = job / "mask.png"
            with open(mask_path, "wb") as f:
                shutil.copyfileobj(mask.file, f)

        out_path = job / "rectified.png"

        run_script(
            in_path, out_path,
            width_mm=width_mm, height_mm=height_mm, dpi=dpi,
            corner_frac=corner_frac, polarity=polarity,
            enforce_axes=enforce_axes, mask_path=mask_path
        )

        if not out_path.exists():
            raise HTTPException(status_code=500, detail="Rectified image not produced")
        # Return the file; FastAPI will stream it
        return FileResponse(str(out_path), media_type="image/png", filename="rectified.png")
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Render’s disks are ephemeral; we can clean up to be nice
        try:
            shutil.rmtree(job, ignore_errors=True)
        except Exception:
            pass

@app.get("/")
def health():
    return {"ok": True, "hint": "POST /rectify with image + form fields"}
