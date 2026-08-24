package queue

import (
	"testing"
	"time"

	"voice-cloning-queue/models"
)

func TestPriorityQueue_BasicSubmitAndNext(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	req := models.RenderRequest{
		Chunks: []string{"test 1", "test 2"},
		Lane:   "batch",
	}
	job := models.NewRenderJob(req, "job_1")
	q.Submit(job)

	retrieved, pos := q.GetJob("job_1")
	if retrieved == nil {
		t.Fatalf("expected job_1 to exist")
	}
	if *pos != 0 {
		t.Errorf("expected position 0, got %d", *pos)
	}

	next := q.NextJob()
	if next == nil || next.JobID != "job_1" {
		t.Fatalf("expected next job to be job_1")
	}
	if next.Status != models.StatusRunning {
		t.Errorf("expected status running, got %s", next.Status)
	}

	q.MarkCompleted("job_1", map[string]interface{}{"status": "ok"}, nil)

	retrieved, _ = q.GetJob("job_1")
	if retrieved.Status != models.StatusCompleted {
		t.Errorf("expected status completed, got %s", retrieved.Status)
	}
}

func TestPriorityQueue_InteractivePriority(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	// Submit 2 batch jobs
	q.Submit(models.NewRenderJob(models.RenderRequest{Chunks: []string{"b1"}, Lane: "batch"}, "batch_1"))
	q.Submit(models.NewRenderJob(models.RenderRequest{Chunks: []string{"b2"}, Lane: "batch"}, "batch_2"))

	// Submit 1 interactive job
	q.Submit(models.NewRenderJob(models.RenderRequest{Chunks: []string{"i1"}, Lane: "interactive"}, "interactive_1"))

	// First job must be interactive_1
	j1 := q.NextJob()
	if j1.JobID != "interactive_1" {
		t.Errorf("expected interactive_1 first, got %s", j1.JobID)
	}

	// Second job must be batch_1
	j2 := q.NextJob()
	if j2.JobID != "batch_1" {
		t.Errorf("expected batch_1 second, got %s", j2.JobID)
	}
}

func TestPriorityQueue_CancelJob(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	job := models.NewRenderJob(models.RenderRequest{Chunks: []string{"test"}, Lane: "batch"}, "job_to_cancel")
	q.Submit(job)

	ok := q.Cancel("job_to_cancel")
	if !ok {
		t.Errorf("expected cancel to return true")
	}

	retrieved, _ := q.GetJob("job_to_cancel")
	if retrieved.Status != models.StatusCancelled {
		t.Errorf("expected status cancelled, got %s", retrieved.Status)
	}
}

func TestPriorityQueue_WaitTimeout(t *testing.T) {
	q := NewPriorityQueue()
	defer q.Close()

	job := models.NewRenderJob(models.RenderRequest{Chunks: []string{"test"}, Lane: "batch"}, "job_wait")
	q.Submit(job)

	// Wait with 50ms timeout (should return nil because job is still queued)
	finished := q.Wait("job_wait", 50*time.Millisecond)
	if finished != nil {
		t.Errorf("expected nil on timeout, got %v", finished)
	}

	// Mark completed
	go func() {
		time.Sleep(20 * time.Millisecond)
		q.MarkCompleted("job_wait", map[string]interface{}{"status": "ok"}, nil)
	}()

	finished = q.Wait("job_wait", 100*time.Millisecond)
	if finished == nil || finished.Status != models.StatusCompleted {
		t.Errorf("expected completed job, got %v", finished)
	}
}
