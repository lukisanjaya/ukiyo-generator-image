#!/usr/bin/env python3
"""
generate_samples.py
Script untuk membuat gambar sample per visual style (104 styles).
Memanggil API lokal /samples/generate/{style_id} sehingga menggunakan model yang sudah ada di RAM.
Jika file sample sudah ada di folder samples/{style_id}.png, otomatis di-skip (resume-friendly).
"""
import os
import sys
import time
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STYLES_FILE = BASE_DIR / "styles.json"
SAMPLES_DIR = BASE_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PROMPT = (
    "A brave warrior standing on a mountain cliff overlooking an ancient fantasy kingdom and misty valley, "
    "dramatic lighting, cinematic composition"
)

def main():
    parser = argparse.ArgumentParser(description="Generate sample images for all styles.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Base URL API ukiyo-generator")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt standar untuk seluruh style")
    parser.add_argument("--steps", type=int, default=10, help="Jumlah inference steps (default: 10)")
    parser.add_argument("--ratio", default="9:16", help="Aspect ratio (9:16 atau 16:9)")
    parser.add_argument("--seed", type=int, default=42, help="Seed tetap untuk hasil komparasi yang konsisten")
    parser.add_argument("--force", action="store_true", help="Generate ulang meskipun file sample sudah ada")
    args = parser.parse_args()

    if not STYLES_FILE.exists():
        print(f"[Error] File styles.json tidak ditemukan di {STYLES_FILE}")
        sys.exit(1)

    with open(STYLES_FILE, "r", encoding="utf-8") as f:
        styles = json.load(f)

    total = len(styles)
    print(f"==================================================")
    print(f"Memulai Pembuatan Sample Style ({total} Styles)")
    print(f"API Target : {args.api_url}")
    print(f"Prompt     : {args.prompt}")
    print(f"Steps      : {args.steps} | Ratio: {args.ratio} | Seed: {args.seed}")
    print(f"Output     : {SAMPLES_DIR}")
    print(f"==================================================\n")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, s in enumerate(styles, 1):
        style_id = s["id"]
        style_name = s["name"]
        sample_file = SAMPLES_DIR / f"{style_id}.png"

        if sample_file.exists() and not args.force:
            print(f"[{idx}/{total}] SKIP: {style_name} ({style_id}) - Sample sudah ada.")
            skip_count += 1
            continue

        print(f"[{idx}/{total}] GENERATING: {style_name} ({style_id})...", end=" ", flush=True)
        start_time = time.time()

        payload = {
            "prompt": args.prompt,
            "ratio": args.ratio,
            "steps": args.steps,
            "seed": args.seed
        }
        url = f"{args.api_url.rstrip('/')}/generate-sample/{style_id}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                duration = time.time() - start_time
                res_data = json.loads(response.read().decode("utf-8"))
                if res_data.get("success"):
                    print(f"SUKSES ({duration:.1f}s)")
                    success_count += 1
                else:
                    print(f"GAGAL: {res_data}")
                    fail_count += 1
        except Exception as e:
            duration = time.time() - start_time
            print(f"ERROR ({duration:.1f}s): {e}")
            fail_count += 1

        # Istirahat 1 detik antar gambar
        time.sleep(1)

    print(f"\n==================================================")
    print(f"SELESAI!")
    print(f"Sukses Dibuat : {success_count}")
    print(f"Dilewati      : {skip_count}")
    print(f"Gagal         : {fail_count}")
    print(f"Total Style   : {total}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
