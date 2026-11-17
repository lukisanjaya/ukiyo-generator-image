#!/usr/bin/env bash
set -e

BASE_URL="http://localhost:8000"

echo "=== 1. Testing /health ==="
curl -s "${BASE_URL}/health" | jq . || curl -s "${BASE_URL}/health"
echo -e "\n"

echo "=== 2. Testing /generate (YouTube Shorts 9:16) ==="
curl -s -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Ancient Kutai kingdom in Borneo, King Mulawarman standing in front of an ancient wooden palace, Mahakam river, traditional Indonesian clothing, historical atmosphere",
    "ratio": "9:16",
    "steps": 15,
    "seed": 12345
  }' | jq . || true
echo -e "\n"

echo "=== 3. Testing /generate (YouTube Landscape 16:9) ==="
curl -s -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Panoramic view of ancient Kutai kingdom port along Mahakam river, wooden boats, tropical Borneo rainforest background",
    "ratio": "16:9",
    "steps": 15,
    "seed": 67890
  }' | jq . || true
echo -e "\n"
