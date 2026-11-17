import os
import uuid
import json
import asyncio
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
STATIC_DIR = BASE_DIR / "static"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

CPU_THREADS = int(os.environ.get("CPU_THREADS", "4"))
torch.set_num_threads(CPU_THREADS)

MODEL_ID = os.environ.get("SD_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SERVER_URL = os.environ.get("SERVER_URL", "").rstrip("/")

STYLE_POSITIVE = "ukiyo-e style, traditional Japanese woodblock print, Edo period art, intricate linework, vintage paper texture, Katsushika Hokusai and Hiroshige aesthetic, masterwork ukiyo-e print"
DEFAULT_NEGATIVE = "photorealistic, 3d render, photo, realistic photograph, modern, glossy, neon, blurry, distorted, bad anatomy, deformed fingers, extra limbs, watermark, signature, text"

# Load styles dari bang-story
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
    # Gunakan DPM-Solver++ Multistep Scheduler agar cepat (15-20 steps sudah bagus)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Ukiyo-e Image Generator API",
    description="FastAPI for generating Ukiyo-e style images using Stable Diffusion 1.5",
    version="1.0.0",
    lifespan=lifespan
)

# Static files untuk mengakses gambar output dan assets web UI
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =========================
# SCHEMAS
# =========================
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Deskripsi gambar yang ingin dibuat")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt opsional")
    style_id: Optional[str] = Field(default=None, description="ID style dari bang-story (misal: 'ukiyo_e_edo', 'stickman', 'ghibli_nostalgic')")
    style_prompt: Optional[str] = Field(default=None, description="Custom style prompt override")
    ratio: str = Field(default="9:16", description="Rasio aspek: '9:16' (Shorts) atau '16:9' (Landscape)")
    steps: int = Field(default=15, ge=5, le=50, description="Inference steps (default: 15)")
    seed: Optional[int] = Field(default=None, description="Random seed untuk konsistensi visual")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="Guidance scale / CFG")


# =========================
# CORE GENERATION FUNCTION
# =========================
def generate_image_sync(data: GenerateRequest) -> dict:
    if pipe is None:
        raise RuntimeError("Model is not initialized")

    # Resolusi hemat RAM VPS
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
    filename = f"{uuid.uuid4().hex}.png"
    output_path = OUTPUT_DIR / filename
    image.save(output_path, format="PNG", optimize=True)

    del result

    return {
        "success": True,
        "filename": filename,
        "relative_url": f"/outputs/{filename}",
        "width": width,
        "height": height,
        "seed": seed,
        "ratio": data.ratio,
        "steps": data.steps,
        "style_name": active_style_name,
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
            "docs": "/docs"
        }
    }


@app.get("/styles")
def get_styles():
    return {
        "total": len(STYLES_LIST),
        "styles": STYLES_LIST
    }


@app.get("/recent")
def recent_outputs(request: Request, limit: int = 24):
    images = []
    base = SERVER_URL if SERVER_URL else str(request.base_url).rstrip("/")
    for p in sorted(OUTPUT_DIR.glob("*.png"), key=lambda f: f.stat().st_mtime, reverse=True)[:limit]:
        images.append({
            "filename": p.name,
            "image_url": f"{base}/outputs/{p.name}",
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
        raise HTTPException(
            status_code=503,
            detail="Model not ready"
        )

    try:
        # Antrean satu proses agar CPU & RAM VPS tidak jebol
        async with generation_lock:
            result = await run_in_threadpool(generate_image_sync, data)

        # Bangun URL lengkap untuk image_url
        if SERVER_URL:
            base = SERVER_URL
        else:
            base = str(request.base_url).rstrip("/")

        result["image_url"] = f"{base}{result['relative_url']}"
        del result["relative_url"]
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
