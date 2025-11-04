import os, uuid, shutil, subprocess, sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

app = FastAPI(title="Four-Dot Rectifier")

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