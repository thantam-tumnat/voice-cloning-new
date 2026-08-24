package worker

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"voice-cloning-queue/models"
	"voice-cloning-queue/queue"
)

// Worker continuously processes jobs from the PriorityQueue and dispatches to Python GPU Service.
type Worker struct {
	q            *queue.PriorityQueue
	pythonGPUURL string
	client       *http.Client
	stopChan     chan struct{}
}

// NewWorker initializes a new Worker.
func NewWorker(q *queue.PriorityQueue, pythonGPUURL string) *Worker {
	return &Worker{
		q:            q,
		pythonGPUURL: pythonGPUURL,
		client: &http.Client{
			Timeout: 10 * time.Minute,
		},
		stopChan: make(chan struct{}),
	}
}

// Start launches the background worker loop.
func (w *Worker) Start() {
	go w.run()
}

// Stop stops the worker gracefully.
func (w *Worker) Stop() {
	close(w.stopChan)
}

func (w *Worker) run() {
	fmt.Printf("[worker] Go Queue Worker started — forwarding to Python GPU at %s\n", w.pythonGPUURL)

	for {
		select {
		case <-w.stopChan:
			return
		default:
		}

		job := w.q.NextJob()
		if job == nil {
			return
		}

		w.processJob(job)
	}
}

func (w *Worker) processJob(job *models.RenderJob) {
	fmt.Printf("[worker] >>> Executing Job: %s (lane=%s, client=%s, chunks=%d)\n",
		job.JobID, job.Lane, job.Client, len(job.Chunks))

	for i, chunk := range job.Chunks {
		fmt.Printf("   [chunk %d/%d]: %s\n", i+1, len(job.Chunks), chunk)
	}

	start := time.Now()

	// Prepare payload for Python GPU service
	reqBody := map[string]interface{}{
		"job_id":     job.JobID,
		"chunks":     job.Chunks,
		"cfg_value":  job.CFGValue,
		"timesteps":  job.Timesteps,
		"lora":       job.LoRA,
		"output":     job.Output,
		"lane":       job.Lane,
		"client":     job.Client,
	}
	if job.Voice != nil {
		reqBody["voice"] = job.Voice
	}

	jsonBytes, err := json.Marshal(reqBody)
	if err != nil {
		w.q.MarkFailed(job.JobID, fmt.Sprintf("failed to marshal request: %v", err))
		return
	}

	// Dispatch to Python GPU service (/v2/direct_render or /v2/jobs/render?wait=600)
	targetURL := fmt.Sprintf("%s/v2/direct_render", w.pythonGPUURL)
	httpReq, err := http.NewRequest("POST", targetURL, bytes.NewBuffer(jsonBytes))
	if err != nil {
		w.q.MarkFailed(job.JobID, fmt.Sprintf("failed to create request: %v", err))
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := w.client.Do(httpReq)
	if err != nil {
		// Fallback to /v2/jobs/render?wait=600 if direct_render is not yet mounted
		fallbackURL := fmt.Sprintf("%s/v2/jobs/render?wait=600", w.pythonGPUURL)
		fallbackReq, err2 := http.NewRequest("POST", fallbackURL, bytes.NewBuffer(jsonBytes))
		if err2 == nil {
			fallbackReq.Header.Set("Content-Type", "application/json")
			resp, err = w.client.Do(fallbackReq)
		}
	}

	if err != nil {
		errMsg := fmt.Sprintf("GPU service unreachable: %v", err)
		fmt.Printf("[worker] <<< Job Failed %s: %s\n", job.JobID, errMsg)
		w.q.MarkFailed(job.JobID, errMsg)
		return
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		w.q.MarkFailed(job.JobID, fmt.Sprintf("failed to read response: %v", err))
		return
	}

	if resp.StatusCode >= 400 {
		var errObj map[string]interface{}
		errMsg := string(bodyBytes)
		if json.Unmarshal(bodyBytes, &errObj) == nil {
			if e, ok := errObj["error"].(string); ok {
				errMsg = e
			} else if d, ok := errObj["detail"].(string); ok {
				errMsg = d
			}
		}
		fmt.Printf("[worker] <<< Job Failed %s (status %d): %s\n", job.JobID, resp.StatusCode, errMsg)
		w.q.MarkFailed(job.JobID, errMsg)
		return
	}

	contentType := resp.Header.Get("Content-Type")

	var result map[string]interface{}
	var payload []byte

	if contentType == "application/octet-stream" {
		payload = bodyBytes
		result = map[string]interface{}{
			"mode":        "npz",
			"sample_rate": resp.Header.Get("X-Sample-Rate"),
			"chunks":      resp.Header.Get("X-Chunks"),
		}
	} else {
		// JSON response
		if err := json.Unmarshal(bodyBytes, &result); err != nil {
			w.q.MarkFailed(job.JobID, fmt.Sprintf("invalid JSON from GPU: %v", err))
			return
		}
		// Check if result object contains result wrapper
		if innerRes, ok := result["result"].(map[string]interface{}); ok {
			result = innerRes
		}
	}

	elapsed := time.Since(start).Seconds()
	fmt.Printf("[worker] <<< Job Completed: %s (took %.2fs)\n", job.JobID, elapsed)
	w.q.MarkCompleted(job.JobID, result, payload)
}
