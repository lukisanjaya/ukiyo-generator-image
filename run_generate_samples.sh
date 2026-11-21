#!/usr/bin/env bash
# Menjalankan generate_samples.py di background (nohup)
cd "$(dirname "$0")"
chmod +x generate_samples.py
echo "Memulai generate samples di background..."
nohup ./venv/bin/python generate_samples.py > generate_samples.log 2>&1 &
echo "Proses berjalan di background PID: $!"
echo "Untuk melihat log: tail -f generate_samples.log"
