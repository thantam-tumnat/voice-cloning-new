package main

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"

	"voice-cloning-queue/handlers"
	"voice-cloning-queue/queue"
	"voice-cloning-queue/worker"
)

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func main() {
	port := getEnv("PORT", "8020")
	pythonGPUURL := getEnv("PYTHON_GPU_URL", "http://127.0.0.1:8021")

	fmt.Println("================================================================")
	fmt.Println("🚀 SiangTTS High-Performance Go Fiber Queue Gateway (:8020)")
	fmt.Printf("👉 Python GPU Backend: %s\n", pythonGPUURL)
	fmt.Println("================================================================")

	// 1. Initialize Priority Queue & Worker
	pq := queue.NewPriorityQueue()
	w := worker.NewWorker(pq, pythonGPUURL)
	w.Start()
	defer w.Stop()
	defer pq.Close()

	// 2. Initialize Fiber App
	app := fiber.New(fiber.Config{
		AppName:               "SiangTTS Go Queue Gateway",
		BodyLimit:             50 * 1024 * 1024, // 50MB for reference audio uploads
		DisableStartupMessage: false,
	})

	// 3. Middlewares
	app.Use(recover.New())
	app.Use(logger.New(logger.Config{
		Format: "[${time}] ${status} - ${latency} ${method} ${path}\n",
	}))
	app.Use(cors.New(cors.Config{
		AllowOrigins: "*",
		AllowHeaders: "*",
		AllowMethods: "GET,POST,HEAD,PUT,DELETE,PATCH,OPTIONS",
	}))

	// 4. Handlers
	jobsH := handlers.NewJobsHandler(pq)
	proxyH := handlers.NewProxyHandler(pythonGPUURL)
	dashH := handlers.NewDashboardHandler(pq, pythonGPUURL)

	// 5. Job Queue Routes (Handled natively by Go)
	app.Post("/v2/jobs/render", jobsH.Render)
	app.Get("/v2/jobs", jobsH.List)
	app.Get("/v2/jobs/:job_id", jobsH.GetJob)
	app.Get("/v2/jobs/:job_id/result", jobsH.GetResult)
	app.Delete("/v2/jobs/:job_id", jobsH.Cancel)

	// 6. Voice, Speaker & Health Routes (Proxied to Python GPU :8021)
	app.Post("/v2/voices/resolve", proxyH.Forward)
	app.Post("/v2/voices", proxyH.Forward)
	app.Get("/v2/voices", proxyH.Forward)
	// Static paths first: Fiber matches in registration order, so ":handle"
	// and ":speaker_id" must not be allowed to swallow "seed".
	app.Post("/v2/voices/seed", proxyH.Forward)
	app.Delete("/v2/voices/seed", proxyH.Forward)
	app.Get("/v2/voices/:handle", proxyH.Forward)
	// Two segments -- ":handle" alone never matches this.
	app.Get("/v2/voices/:speaker_id/audio", proxyH.Forward)
	app.Delete("/v2/voices/:speaker_id", proxyH.Forward)
	app.Get("/health", proxyH.Forward)

	// 7. Web Dashboard UI
	app.Get("/", dashH.Index)

	// 8. Graceful Shutdown Setup
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		fmt.Println("\n[shutdown] Gracefully stopping Go Queue Gateway...")
		_ = app.Shutdown()
	}()

	// 8. Start Listening
	addr := fmt.Sprintf("127.0.0.1:%s", port)
	log.Printf("[ready] Go Queue Gateway listening on http://%s\n", addr)
	if err := app.Listen(addr); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}
