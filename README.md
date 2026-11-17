# Ukiyo-e Image Generator API (FastAPI + SD 1.5 CPU)

REST API untuk menghasilkan gambar bergaya **Ukiyo-e** (seni cetak kayu tradisional Jepang era Edo) menggunakan **Stable Diffusion 1.5**, dioptimalkan khusus untuk **VPS Ubuntu CPU 4-Core & RAM 8 GB**.

---

## 🌟 Fitur Utama
1. **Otomatis Gaya Ukiyo-e**: Prompt diperkaya secara otomatis dengan parameter estetika Ukiyo-e (tekstur kertas kuno, garis tinta khas Hokusai/Hiroshige), tanpa perlu menulis ulang berulang kali.
2. **Mendukung 2 Rasio Aspek**:
   - `9:16` (YouTube Shorts / TikTok): `512 x 768`
   - `16:9` (YouTube Landscape): `768 x 512`
3. **Queue Lock (Antrean Tunggal)**: Mencegah benturan inferensi paralel sehingga RAM 8 GB dan CPU VPS tidak kehabisan memori (OOM).
4. **LoRA Opsional**: Mendukung LoRA Ukiyo-e (`models/ukiyoe.safetensors`). Jika file tidak ada, sistem otomatis menggunakan base model + prompt engineering.
5. **DPM-Solver++ Scheduler**: Inferensi cepat (15 step sudah menghasilkan detail yang sangat tajam).
6. **Integrasi Mudah**: Dapat dipanggil via `curl`, Python, maupun aplikasi **Go**.

---

## 📁 Struktur Direktori
```text
ukiyo-generator/
├── app.py                     # Kode utama FastAPI & Stable Diffusion pipeline
├── requirements.txt           # Dependensi Python
├── test_generate.sh           # Script uji coba otomatis via cURL
├── ukiyo.service              # Konfigurasi systemd untuk VPS
├── models/
│   └── ukiyoe.safetensors     # (Opsional) File LoRA SD 1.5
├── outputs/                   # Folder hasil output gambar (.png)
├── examples/
│   └── go-client/
│       └── main.go            # Contoh pemanggilan dari Go
└── venv/                      # Virtual Environment Python
```

---

## 🚀 Panduan Menjalankan

### 1. Masuk ke Folder Project & Aktifkan venv
```bash
cd ~/ukiyo-generator
source venv/bin/activate
```

### 2. Jalankan Server API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
*Tunggu hingga log menampilkan:*
```text
Loading Stable Diffusion (stable-diffusion-v1-5/stable-diffusion-v1-5)...
Model ready!
INFO: Application startup complete.
```

---

## 📡 Dokumentasi Endpoint

### 1. `GET /health`
Mengecek status ketersediaan model dan konfigurasi server.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "lora_loaded": false,
  "device": "cpu",
  "cpu_threads": 4
}
```

---

### 2. `POST /generate`
Meminta generasi gambar baru.

**Payload:**
```json
{
  "prompt": "Ancient Kutai kingdom in Borneo, King Mulawarman standing in front of an ancient wooden palace, Mahakam river, traditional Indonesian clothing, historical atmosphere",
  "ratio": "9:16",
  "steps": 15,
  "seed": 12345
}
```

**Response:**
```json
{
  "success": true,
  "filename": "d3b4f1a2....png",
  "image_url": "http://localhost:8000/outputs/d3b4f1a2....png",
  "width": 512,
  "height": 768,
  "seed": 12345,
  "ratio": "9:16",
  "steps": 15
}
```

---

## 💻 Contoh Pemanggilan dari Golang

Jalankan contoh client Go:
```bash
cd ~/ukiyo-generator/examples/go-client
go run main.go
```

---

## ⚙️ Menjalankan sebagai Background Service (systemd)

Agar API tetap berjalan otomatis meskipun terminal ditutup atau VPS di-reboot:

```bash
sudo cp ~/ukiyo-generator/ukiyo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ukiyo
sudo systemctl status ukiyo
```
