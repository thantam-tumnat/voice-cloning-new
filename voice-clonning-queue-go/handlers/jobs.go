package handlers

import (
	"fmt"
	"sort"
	"strconv"
	"time"

	"github.com/gofiber/fiber/v2"

	"voice-cloning-queue/models"
	"voice-cloning-queue/queue"
)

// JobsHandler handles all /v2/jobs routes.
type JobsHandler struct {
	q *queue.PriorityQueue
}

// NewJobsHandler creates a new JobsHandler.
func NewJobsHandler(q *queue.PriorityQueue) *JobsHandler {
	return &JobsHandler{q: q}
}

// Render handles POST /v2/jobs/render.
func (h *JobsHandler) Render(c *fiber.Ctx) error {
	var req models.RenderRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": fmt.Sprintf("invalid JSON payload: %v", err)})
	}

	// Filter empty chunks
	validChunks := make([]string, 0, len(req.Chunks))
	for _, ch := range req.Chunks {
		if len(ch) > 0 {
			validChunks = append(validChunks, ch)
		}
	}
	if len(validChunks) == 0 {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "chunks is empty"})
	}
	req.Chunks = validChunks

	jobID := h.q.GenerateJobID()
	if req.JobID != nil && *req.JobID != "" {
		jobID = *req.JobID
	}

	job := models.NewRenderJob(req, jobID)
	h.q.Submit(job)

	// Check ?wait=N
	waitStr := c.Query("wait", "0")
	waitSec, _ := strconv.ParseFloat(waitStr, 64)

	if waitSec > 0 {
		if waitSec > 600 {
			waitSec = 600
		}
		finished := h.q.Wait(jobID, time.Duration(waitSec*float64(time.Second)))
		if finished != nil {
			if finished.Status == models.StatusFailed {
				return c.Status(fiber.StatusInternalServerError).JSON(finished.AsDict(nil))
			}
			if len(finished.Payload) > 0 {
				c.Set("Content-Type", "application/octet-stream")
				c.Set("X-Job-Id", finished.JobID)
				if sr, ok := finished.Result["sample_rate"]; ok {
					c.Set("X-Sample-Rate", fmt.Sprintf("%v", sr))
				}
				if ch, ok := finished.Result["chunks"]; ok {
					c.Set("X-Chunks", fmt.Sprintf("%v", ch))
				}
				payload := finished.Payload
				finished.Payload = nil // free memory
				return c.Send(payload)
			}
			return c.Status(fiber.StatusOK).JSON(finished.AsDict(nil))
		}
	}

	// Return 202 Accepted
	_, pos := h.q.GetJob(jobID)
	return c.Status(fiber.StatusAccepted).JSON(job.AsDict(pos))
}

// List handles GET /v2/jobs.
func (h *JobsHandler) List(c *fiber.Ctx) error {
	statusFilter := c.Query("status", "")
	limitStr := c.Query("limit", "100")
	limit, _ := strconv.Atoi(limitStr)
	if limit <= 0 {
		limit = 100
	}

	allJobs := h.q.ListJobs()
	counts, waiting, running := h.q.GetStats()
	positions := h.q.GetPositions()

	// Sort by created desc
	sort.Slice(allJobs, func(i, j int) bool {
		return allJobs[i].Created > allJobs[j].Created
	})

	filtered := make([]*models.RenderJob, 0)
	for _, j := range allJobs {
		if statusFilter == "" || string(j.Status) == statusFilter {
			filtered = append(filtered, j)
		}
	}

	if len(filtered) > limit {
		filtered = filtered[:limit]
	}

	jobDicts := make([]map[string]interface{}, 0, len(filtered))
	for _, j := range filtered {
		var pos *int
		if p, ok := positions[j.JobID]; ok {
			pos = &p
		}
		jobDicts = append(jobDicts, j.AsDict(pos))
	}

	var runningDict map[string]interface{}
	if running != nil {
		runningDict = running.AsDict(nil)
	}

	return c.JSON(fiber.Map{
		"counts":  counts,
		"running": runningDict,
		"waiting": waiting,
		"total":   len(allJobs),
		"jobs":    jobDicts,
	})
}

// GetJob handles GET /v2/jobs/:job_id.
func (h *JobsHandler) GetJob(c *fiber.Ctx) error {
	jobID := c.Params("job_id")
	job, pos := h.q.GetJob(jobID)
	if job == nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "unknown job"})
	}
	return c.JSON(job.AsDict(pos))
}

// GetResult handles GET /v2/jobs/:job_id/result.
func (h *JobsHandler) GetResult(c *fiber.Ctx) error {
	jobID := c.Params("job_id")
	job, _ := h.q.GetJob(jobID)
	if job == nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "unknown job"})
	}

	if job.Status != models.StatusCompleted {
		return c.Status(fiber.StatusConflict).JSON(fiber.Map{
			"error":  fmt.Sprintf("job is %s", job.Status),
			"status": job.Status,
			"detail": job.Error,
		})
	}

	if len(job.Payload) > 0 {
		c.Set("Content-Type", "application/octet-stream")
		c.Set("X-Job-Id", job.JobID)
		if sr, ok := job.Result["sample_rate"]; ok {
			c.Set("X-Sample-Rate", fmt.Sprintf("%v", sr))
		}
		if ch, ok := job.Result["chunks"]; ok {
			c.Set("X-Chunks", fmt.Sprintf("%v", ch))
		}
		payload := job.Payload
		job.Payload = nil // free memory
		return c.Send(payload)
	}

	if job.Result != nil {
		return c.JSON(job.Result)
	}

	return c.Status(fiber.StatusGone).JSON(fiber.Map{"error": "result already delivered"})
}

// Cancel handles DELETE /v2/jobs/:job_id.
func (h *JobsHandler) Cancel(c *fiber.Ctx) error {
	jobID := c.Params("job_id")
	job, _ := h.q.GetJob(jobID)
	if job == nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "unknown job"})
	}

	ok := h.q.Cancel(jobID)
	return c.JSON(fiber.Map{
		"job_id":    jobID,
		"cancelled": ok,
		"status":    job.Status,
	})
}
