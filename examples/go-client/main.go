package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

type HealthResponse struct {
	Status      string `json:"status"`
	ModelLoaded bool   `json:"model_loaded"`
	LoraLoaded  bool   `json:"lora_loaded"`
	Device      string `json:"device"`
	CPUThreads  int    `json:"cpu_threads"`
}

type GenerateRequest struct {
	Prompt         string  `json:"prompt"`
	NegativePrompt string  `json:"negative_prompt,omitempty"`
	Ratio          string  `json:"ratio"` // "9:16" or "16:9"
	Steps          int     `json:"steps"`
	Seed           *int64  `json:"seed,omitempty"`
	GuidanceScale  float64 `json:"guidance_scale,omitempty"`
}

type GenerateResponse struct {
	Success  bool   `json:"success"`
	Filename string `json:"filename"`
	ImageURL string `json:"image_url"`
	Width    int    `json:"width"`
	Height   int    `json:"height"`
	Seed     int64  `json:"seed"`
	Ratio    string `json:"ratio"`
	Steps    int    `json:"steps"`
}

func main() {
	baseURL := "http://localhost:8000"
	client := &http.Client{Timeout: 300 * time.Second}

	// 1. Check Health
	fmt.Println("🔍 Memeriksa status Ukiyo-e API...")
	resp, err := client.Get(baseURL + "/health")
	if err != nil {
		fmt.Printf("❌ Error connecting to server: %v\n", err)
		return
	}
	defer resp.Body.Close()

	var health HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		fmt.Printf("❌ Gagal membaca respons health: %v\n", err)
		return
	}
	fmt.Printf("✅ Health check: Status=%s | ModelLoaded=%v | Device=%s | CPUThreads=%d\n",
		health.Status, health.ModelLoaded, health.Device, health.CPUThreads)

	if !health.ModelLoaded {
		fmt.Println("⚠️ Model belum siap, mohon tunggu beberapa saat...")
		return
	}

	// 2. Request Image Generation
	fmt.Println("🎨 Mengirim request generate gambar...")
	seedVal := int64(12345)
	reqBody := GenerateRequest{
		Prompt: "Ancient Kutai kingdom in Borneo, King Mulawarman standing in front of an ancient wooden palace, Mahakam river, traditional Indonesian clothing, historical atmosphere",
		Ratio:  "9:16",
		Steps:  15,
		Seed:   &seedVal,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		fmt.Printf("❌ Error serialize request: %v\n", err)
		return
	}

	start := time.Now()
	genResp, err := client.Post(baseURL+"/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Printf("❌ Error generate request: %v\n", err)
		return
	}
	defer genResp.Body.Close()

	if genResp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(genResp.Body)
		fmt.Printf("❌ Server mengembalikan status %d: %s\n", genResp.StatusCode, string(bodyBytes))
		return
	}

	var result GenerateResponse
	if err := json.NewDecoder(genResp.Body).Decode(&result); err != nil {
		fmt.Printf("❌ Gagal decode JSON result: %v\n", err)
		return
	}

	duration := time.Since(start)
	fmt.Printf("🎉 Gambar berhasil di-generate dalam %v!\n", duration)
	fmt.Printf("   - Filename : %s\n", result.Filename)
	fmt.Printf("   - Dimensi  : %dx%d (Rasio: %s)\n", result.Width, result.Height, result.Ratio)
	fmt.Printf("   - Seed     : %d\n", result.Seed)
	fmt.Printf("   - URL      : %s\n", result.ImageURL)

	// 3. Simpan file lokal sebagai verifikasi
	downloadResp, err := client.Get(result.ImageURL)
	if err == nil && downloadResp.StatusCode == http.StatusOK {
		defer downloadResp.Body.Close()
		outFile, err := os.Create("downloaded_" + result.Filename)
		if err == nil {
			defer outFile.Close()
			_, _ = io.Copy(outFile, downloadResp.Body)
			fmt.Printf("💾 Berhasil mengunduh gambar ke: downloaded_%s\n", result.Filename)
		}
	}
}
