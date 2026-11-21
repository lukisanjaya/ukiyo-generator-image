import os
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# =========================
# CONFIGURATION & PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
SAMPLES_DIR = BASE_DIR / "samples"
STATIC_DIR = BASE_DIR / "static"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

CPU_THREADS = int(os.environ.get("CPU_THREADS", "4"))
torch.set_num_threads(CPU_THREADS)

MODEL_ID = os.environ.get("SD_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SERVER_URL = os.environ.get("SERVER_URL", "").rstrip("/")

STYLE_POSITIVE = "ukiyo-e style, traditional Japanese woodblock print, Edo period art, intricate linework, vintage paper texture, Katsushika Hokusai and Hiroshige aesthetic, masterwork ukiyo-e print"
DEFAULT_NEGATIVE = "photorealistic, 3d render, photo, realistic photograph, modern, glossy, neon, blurry, distorted, bad anatomy, deformed fingers, extra limbs, watermark, signature, text"

# Load styles
STYLES_FILE = BASE_DIR / "styles.json"
STYLES_LIST: List[Dict[str, Any]] = []
STYLES_MAP: Dict[str, Dict[str, Any]] = {}

if STYLES_FILE.exists():
    try:
        with open(STYLES_FILE, "r", encoding="utf-8") as f:
            STYLES_LIST = json.load(f)
            STYLES_MAP = {s["id"]: s for s in STYLES_LIST}
        print(f"Loaded {len(STYLES_LIST)} styles from {STYLES_FILE.name}")
    except Exception as e:
        print(f"Error loading styles.json: {e}")

pipe: Optional[StableDiffusionPipeline] = None
lora_loaded: bool = False
generation_lock = asyncio.Lock()


def load_model():
    global pipe, lora_loaded
    print(f"Loading Stable Diffusion ({MODEL_ID}) on {DEVICE} with {CPU_THREADS} CPU threads...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    # Gunakan DPM-Solver++ Multistep Scheduler agar cepat (10-15 steps sudah bagus)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True
    )
    pipe.to(DEVICE)
    if DEVICE == "cpu":
        pipe.enable_attention_slicing()

    # Cek apakah ada file LoRA ukiyoe.safetensors
    lora_path = MODELS_DIR / "ukiyoe.safetensors"
    if lora_path.exists():
        try:
            print(f"Loading Ukiyo-e LoRA from {lora_path}...")
            pipe.load_lora_weights(str(MODELS_DIR), weight_name="ukiyoe.safetensors")
            lora_loaded = True
            print("LoRA ukiyoe.safetensors loaded successfully!")
        except Exception as e:
            print(f"Peringatan: Gagal memuat LoRA ({e}), tetap menggunakan base model.")
    else:
        print("LoRA models/ukiyoe.safetensors tidak ditemukan. Menggunakan Ukiyo-e prompt styling bawaan.")

    print("Model ready!")


# =========================
# OUTPUT AUTO-CLEANUP WORKER
# =========================
def run_output_cleanup(max_files: int = 500, max_age_days: int = 1) -> dict:
    """
    Membersihkan file output di folder outputs/ (folder samples TIDAK disentuh).
    1. Menghapus file yang lebih tua dari max_age_days (sudah lewat hari).
    2. Jika file masih lebih dari max_files (500), menghapus yang tertua hingga tersisa 500.
    3. Membersihkan direktori tanggal yang kosong.
    """
    if not OUTPUT_DIR.exists():
        return {"deleted": 0, "remaining": 0}

    now = datetime.now()
    cutoff = now - timedelta(days=max_age_days)
    all_files = list(OUTPUT_DIR.glob("**/*.png"))
    deleted_count = 0
    kept_files = []

    for f in all_files:
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                f.unlink(missing_ok=True)
                deleted_count += 1
            else:
                kept_files.append(f)
        except Exception:
            pass

    if len(kept_files) > max_files:
        kept_files.sort(key=lambda x: x.stat().st_mtime)
        excess = len(kept_files) - max_files
        for f in kept_files[:excess]:
            try:
                f.unlink(missing_ok=True)
                deleted_count += 1
            except Exception:
                pass

    # Bersihkan direktori kosong di dalam outputs/
    for root, dirs, files in os.walk(OUTPUT_DIR, topdown=False):
        for d in dirs:
            dp = Path(root) / d
            try:
                if not any(dp.iterdir()):
                    dp.rmdir()
            except Exception:
                pass

    remaining = len(list(OUTPUT_DIR.glob("**/*.png")))
    if deleted_count > 0:
        print(f"[Cleanup Scheduler] Dihapus: {deleted_count} file output usang. Tersisa: {remaining} file.")
    return {"deleted": deleted_count, "remaining": remaining}


async def cleanup_worker():
    """Background task yang berjalan berkala setiap 12 jam."""
    while True:
        try:
            await asyncio.sleep(12 * 3600)
            await run_in_threadpool(run_output_cleanup)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Cleanup Scheduler Error] {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    # Jalankan background cleaner task
    cleaner_task = asyncio.create_task(cleanup_worker())
    try:
        yield
    finally:
        cleaner_task.cancel()


app = FastAPI(
    title="Ukiyo-e Image Generator API",
    description="FastAPI for generating Ukiyo-e style images using Stable Diffusion 1.5",
    version="1.0.0",
    lifespan=lifespan
)

# Static files untuk mengakses gambar output, sample, dan assets web UI
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =========================
# SCHEMAS
# =========================
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Deskripsi gambar yang ingin dibuat")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt opsional")
    style_id: Optional[str] = Field(default=None, description="ID style dari styles.json")
    style_prompt: Optional[str] = Field(default=None, description="Custom style prompt override")
    ratio: str = Field(default="9:16", description="Rasio aspek: '9:16' (Shorts) atau '16:9' (Landscape)")
    steps: int = Field(default=15, ge=5, le=50, description="Inference steps (default: 15)")
    seed: Optional[int] = Field(default=None, description="Random seed untuk konsistensi visual")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="Guidance scale / CFG")


class SampleGenerateRequest(BaseModel):
    prompt: Optional[str] = Field(
        default="A brave warrior standing on a mountain cliff overlooking an ancient fantasy kingdom and misty valley, dramatic lighting, detailed scenery",
        description="Prompt standar untuk sample"
    )
    ratio: str = Field(default="9:16", description="Rasio aspek")
    steps: int = Field(default=12, ge=5, le=30, description="Inference steps")
    seed: int = Field(default=42, description="Fixed seed untuk komparasi konsisten")


# =========================
# CORE GENERATION FUNCTION
# =========================
def generate_image_sync(data: GenerateRequest) -> dict:
    if pipe is None:
        raise RuntimeError("Model is not initialized")

    # Resolusi aspek rasio
    if data.ratio == "9:16":
        width, height = 512, 768
    elif data.ratio == "16:9":
        width, height = 768, 512
    else:
        width, height = 512, 512

    # Seed
    if data.seed is None or data.seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    else:
        seed = data.seed

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Tentukan style prompt yang digunakan
    if data.style_prompt and data.style_prompt.strip():
        chosen_style = data.style_prompt.strip()
        active_style_name = "Custom Style"
    elif data.style_id and data.style_id in STYLES_MAP:
        chosen_style = STYLES_MAP[data.style_id]["prompt"]
        active_style_name = STYLES_MAP[data.style_id]["name"]
    else:
        chosen_style = STYLE_POSITIVE
        active_style_name = "Ukiyo-e Classic"

    user_prompt = data.prompt.strip()
    full_prompt = f"{user_prompt}, {chosen_style}"

    neg_prompt = (
        f"{data.negative_prompt.strip()}, {DEFAULT_NEGATIVE}"
        if data.negative_prompt
        else DEFAULT_NEGATIVE
    )

    result = pipe(
        prompt=full_prompt,
        negative_prompt=neg_prompt,
        width=width,
        height=height,
        num_inference_steps=data.steps,
        guidance_scale=data.guidance_scale,
        generator=generator,
    )

    image = result.images[0]

    # Simpan di struktur folder per tanggal: outputs/YYYY/MM/DD/{uuid}.png
    now = datetime.now()
    date_path = now.strftime("%Y/%m/%d")
    target_dir = OUTPUT_DIR / date_path
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.png"
    output_path = target_dir / filename
    image.save(output_path, format="PNG", optimize=True)

    del result

    return {
        "success": True,
        "filename": filename,
        "relative_url": f"/outputs/{date_path}/{filename}",
        "width": width,
        "height": height,
        "seed": seed,
        "ratio": data.ratio,
        "steps": data.steps,
        "style_name": active_style_name,
    }


def generate_sample_for_style_sync(style_id: str, data: SampleGenerateRequest) -> dict:
    """Generate gambar sample khusus dan simpan di folder samples/{style_id}.png"""
    if pipe is None:
        raise RuntimeError("Model is not initialized")
    if style_id not in STYLES_MAP:
        raise ValueError(f"Style ID '{style_id}' tidak ditemukan")

    style_obj = STYLES_MAP[style_id]
    style_prompt = style_obj["prompt"]
    style_name = style_obj["name"]

    if data.ratio == "9:16":
        width, height = 512, 768
    elif data.ratio == "16:9":
        width, height = 768, 512
    else:
        width, height = 512, 512

    generator = torch.Generator(device=DEVICE).manual_seed(data.seed)
    full_prompt = f"{data.prompt.strip()}, {style_prompt}"

    result = pipe(
        prompt=full_prompt,
        negative_prompt=DEFAULT_NEGATIVE,
        width=width,
        height=height,
        num_inference_steps=data.steps,
        guidance_scale=7.5,
        generator=generator,
    )

    image = result.images[0]
    filename = f"{style_id}.png"
    sample_path = SAMPLES_DIR / filename
    image.save(sample_path, format="PNG", optimize=True)

    del result

    return {
        "success": True,
        "style_id": style_id,
        "style_name": style_name,
        "filename": filename,
        "relative_url": f"/samples/{filename}"
    }


# =========================
# API ENDPOINTS
# =========================
@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Ukiyo-e Image Generator API",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "recent": "/recent",
            "styles": "/styles",
            "docs": "/docs"
        }
    }


@app.get("/api")
def api_info():
    return {
        "message": "Ukiyo-e Image Generator API",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "recent": "/recent",
            "styles": "/styles",
            "cleanup": "/cleanup",
            "docs": "/docs"
        }
    }


@app.get("/styles")
def get_styles(request: Request):
    base = SERVER_URL if SERVER_URL else str(request.base_url).rstrip("/")
    styles_result = []
    for s in STYLES_LIST:
        item = dict(s)
        sample_path = SAMPLES_DIR / f"{s['id']}.png"
        if sample_path.exists():
            item["sample_url"] = f"{base}/samples/{s['id']}.png"
        else:
            item["sample_url"] = None
        styles_result.append(item)
    return {
        "total": len(styles_result),
        "styles": styles_result
    }


@app.get("/recent")
def recent_outputs(request: Request, limit: int = 24):
    images = []
    base = SERVER_URL if SERVER_URL else str(request.base_url).rstrip("/")
    # Cari semua file .png secara rekursif dalam folder per tanggal
    all_files = sorted(OUTPUT_DIR.glob("**/*.png"), key=lambda f: f.stat().st_mtime, reverse=True)
    for p in all_files[:limit]:
        rel = p.relative_to(OUTPUT_DIR).as_posix()
        images.append({
            "filename": p.name,
            "image_url": f"{base}/outputs/{rel}",
            "created_at": p.stat().st_mtime
        })
    return {"images": images}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipe is not None,
        "lora_loaded": lora_loaded,
        "device": DEVICE,
        "cpu_threads": CPU_THREADS,
    }


@app.post("/generate")
async def generate(data: GenerateRequest, request: Request):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        async with generation_lock:
            result = await run_in_threadpool(generate_image_sync, data)

        base = SERVER_URL if SERVER_URL else str(request.base_url).rstrip("/")
        result["image_url"] = f"{base}{result['relative_url']}"
        del result["relative_url"]
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/samples/generate/{style_id}")
async def generate_sample(style_id: str, data: SampleGenerateRequest, request: Request):
    """Menghasilkan gambar sample untuk satu style_id tertentu dan menyimpannya di samples/{style_id}.png"""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    if style_id not in STYLES_MAP:
        raise HTTPException(status_code=404, detail=f"Style ID '{style_id}' not found")

    try:
        async with generation_lock:
            result = await run_in_threadpool(generate_sample_for_style_sync, style_id, data)

        base = SERVER_URL if SERVER_URL else str(request.base_url).rstrip("/")
        result["image_url"] = f"{base}{result['relative_url']}"
        del result["relative_url"]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cleanup")
def manual_cleanup():
    """Trigger manual pembersihan file outputs."""
    result = run_output_cleanup()
    return {"status": "ok", "result": result}


@app.get("/cleanup/status")
def cleanup_status():
    """Informasi status file di outputs/."""
    all_files = list(OUTPUT_DIR.glob("**/*.png"))
    sample_files = list(SAMPLES_DIR.glob("*.png"))
    return {
        "output_files_count": len(all_files),
        "sample_files_count": len(sample_files),
        "max_output_limit": 500,
        "retention_days": 1,
        "output_dir": str(OUTPUT_DIR),
        "samples_dir": str(SAMPLES_DIR)
    }
