# SiangTTS Go Fiber Queue Service (:8020)

High-performance, ultra-low latency Job Queue Gateway for SiangTTS and VoxCPM2 Voice Cloning, built with **Go (Golang)** and **Fiber v2** (`fasthttp`).

---

## Architecture

```
[Clients: Webhook (:8010) & Tone Studio (:8011)]
                       │
                       ▼ HTTP (:8020)
┌─────────────────────────────────────────────────────────────┐
│  Go Fiber Queue Gateway (:8020)                             │
│  • High-concurrency job dispatcher (Goroutines & Channels)  │
│  • Priority scheduling (Interactive burst vs Batch queue)   │
│  • Non-blocking state management (sync.RWMutex)             │
│  • Reverse proxy for voice caching & health endpoints       │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP (:8021) /v2/direct_render
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Python GPU Worker (:8021)                                  │
│  • Pure PyTorch / CUDA Inference (VoxCPM2 + Thai LoRA)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Features
* **Zero GIL Bottleneck**: Handles tens of thousands of concurrent polling connections with minimal RAM and CPU.
* **Fault Isolation**: If the Python GPU worker encounters CUDA OOM and crashes, the Go queue preserves all waiting jobs.
* **Multi-Lane Priority**: Studio interactive requests (`lane=interactive`) jump ahead of background batch requests (`lane=batch`).
* **100% Backward Compatible**: Implements the exact same `/v2/jobs` API contract as the original Python service.

---

## Quick Start

### 1. Build and Run
```bash
cd voice-clonning-queue-go
go run main.go
```
Or run `start_queue.bat` on Windows.

### 2. Environment Variables
| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8020` | Port for the Go Fiber Queue Gateway |
| `PYTHON_GPU_URL` | `http://127.0.0.1:8021` | Target URL for the Python PyTorch GPU Worker |
