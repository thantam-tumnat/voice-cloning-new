package queue

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
	"time"

	"voice-cloning-queue/models"
)

const (
	InteractiveBurst = 3
	MaxHistory       = 500
)

// PriorityQueue manages multi-lane job scheduling and job persistence in memory.
type PriorityQueue struct {
	mu          sync.RWMutex
	cond        *sync.Cond
	jobs        map[string]*models.RenderJob
	interactive []*models.RenderJob
	batch       []*models.RenderJob
	burstCount  int
	running     *models.RenderJob
	closed      bool
}

// NewPriorityQueue creates an initialized PriorityQueue.
func NewPriorityQueue() *PriorityQueue {
	q := &PriorityQueue{
		jobs:        make(map[string]*models.RenderJob),
		interactive: make([]*models.RenderJob, 0),
		batch:       make([]*models.RenderJob, 0),
	}
	q.cond = sync.NewCond(&q.mu)
	return q
}

// GenerateJobID creates a unique timestamped job identifier.
func (q *PriorityQueue) GenerateJobID() string {
	ts := time.Now().Format("20060102_150405")
	b := make([]byte, 4)
	rand.Read(b)
	return fmt.Sprintf("job_%s_%s", ts, hex.EncodeToString(b))
}

// Submit adds a job to the appropriate lane queue.
func (q *PriorityQueue) Submit(job *models.RenderJob) {
	q.mu.Lock()
	defer q.mu.Unlock()

	q.jobs[job.JobID] = job
	if job.Lane == "interactive" {
		q.interactive = append(q.interactive, job)
	} else {
		q.batch = append(q.batch, job)
	}

	// Evict oldest finished jobs if over capacity
	if len(q.jobs) > MaxHistory {
		q.pruneOldJobs()
	}

	q.cond.Signal()
}

// NextJob blocks until a job is available according to the priority burst policy.
func (q *PriorityQueue) NextJob() *models.RenderJob {
	q.mu.Lock()
	defer q.mu.Unlock()

	for {
		if q.closed {
			return nil
		}

		// Scheduling policy: interactive jobs have priority up to InteractiveBurst times
		if len(q.interactive) > 0 && (q.burstCount < InteractiveBurst || len(q.batch) == 0) {
			job := q.interactive[0]
			q.interactive = q.interactive[1:]
			q.burstCount++
			now := float64(time.Now().UnixNano()) / 1e9
			job.Status = models.StatusRunning
			job.Started = &now
			q.running = job
			return job
		}

		if len(q.batch) > 0 {
			job := q.batch[0]
			q.batch = q.batch[1:]
			q.burstCount = 0
			now := float64(time.Now().UnixNano()) / 1e9
			job.Status = models.StatusRunning
			job.Started = &now
			q.running = job
			return job
		}

		q.cond.Wait()
	}
}

// MarkCompleted updates a job state upon successful synthesis.
func (q *PriorityQueue) MarkCompleted(jobID string, result map[string]interface{}, payload []byte) {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.jobs[jobID]
	if !exists {
		return
	}

	now := float64(time.Now().UnixNano()) / 1e9
	job.Status = models.StatusCompleted
	job.Finished = &now
	job.Result = result
	job.Payload = payload
	job.ChunksDone = job.TotalChunks

	if q.running != nil && q.running.JobID == jobID {
		q.running = nil
	}

	close(job.DoneChan)
}

// MarkFailed updates a job state upon failure.
func (q *PriorityQueue) MarkFailed(jobID string, errMsg string) {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.jobs[jobID]
	if !exists {
		return
	}

	now := float64(time.Now().UnixNano()) / 1e9
	job.Status = models.StatusFailed
	job.Finished = &now
	job.Error = &errMsg

	if q.running != nil && q.running.JobID == jobID {
		q.running = nil
	}

	close(job.DoneChan)
}

// Cancel marks a queued job as cancelled and removes it from waiting lanes.
func (q *PriorityQueue) Cancel(jobID string) bool {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.jobs[jobID]
	if !exists {
		return false
	}

	if job.Status != models.StatusQueued {
		return false
	}

	job.Status = models.StatusCancelled
	now := float64(time.Now().UnixNano()) / 1e9
	job.Finished = &now
	errStr := "cancelled by user"
	job.Error = &errStr

	// Remove from interactive queue
	for i, j := range q.interactive {
		if j.JobID == jobID {
			q.interactive = append(q.interactive[:i], q.interactive[i+1:]...)
			break
		}
	}

	// Remove from batch queue
	for i, j := range q.batch {
		if j.JobID == jobID {
			q.batch = append(q.batch[:i], q.batch[i+1:]...)
			break
		}
	}

	close(job.DoneChan)
	return true
}

// GetJob retrieves a job and its queue position.
func (q *PriorityQueue) GetJob(jobID string) (*models.RenderJob, *int) {
	q.mu.RLock()
	defer q.mu.RUnlock()

	job, exists := q.jobs[jobID]
	if !exists {
		return nil, nil
	}

	if job.Status != models.StatusQueued {
		return job, nil
	}

	pos := q.calculatePosition(job)
	return job, &pos
}

// ListJobs returns a snapshot of all jobs.
func (q *PriorityQueue) ListJobs() []*models.RenderJob {
	q.mu.RLock()
	defer q.mu.RUnlock()

	res := make([]*models.RenderJob, 0, len(q.jobs))
	for _, j := range q.jobs {
		res = append(res, j)
	}
	return res
}

// GetPositions calculates current queue indices for all waiting jobs.
func (q *PriorityQueue) GetPositions() map[string]int {
	q.mu.RLock()
	defer q.mu.RUnlock()

	posMap := make(map[string]int)
	pos := 0
	for _, j := range q.interactive {
		posMap[j.JobID] = pos
		pos++
	}
	for _, j := range q.batch {
		posMap[j.JobID] = pos
		pos++
	}
	return posMap
}

// GetStats returns summary counts of active, waiting, and completed jobs.
func (q *PriorityQueue) GetStats() (map[string]int, map[string]int, *models.RenderJob) {
	q.mu.RLock()
	defer q.mu.RUnlock()

	counts := make(map[string]int)
	for _, j := range q.jobs {
		counts[string(j.Status)]++
	}

	waiting := map[string]int{
		"interactive": len(q.interactive),
		"batch":       len(q.batch),
	}

	return counts, waiting, q.running
}

// Wait blocks until a job finishes or timeout expires.
func (q *PriorityQueue) Wait(jobID string, timeout time.Duration) *models.RenderJob {
	q.mu.RLock()
	job, exists := q.jobs[jobID]
	q.mu.RUnlock()

	if !exists {
		return nil
	}

	select {
	case <-job.DoneChan:
		return job
	case <-time.After(timeout):
		return nil
	}
}

// Close signals the queue to stop waiting goroutines.
func (q *PriorityQueue) Close() {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.closed = true
	q.cond.Broadcast()
}

func (q *PriorityQueue) calculatePosition(target *models.RenderJob) int {
	pos := 0
	for _, j := range q.interactive {
		if j.JobID == target.JobID {
			return pos
		}
		pos++
	}
	for _, j := range q.batch {
		if j.JobID == target.JobID {
			return pos
		}
		pos++
	}
	return pos
}

func (q *PriorityQueue) pruneOldJobs() {
	var oldestID string
	var oldestTime float64 = 1e18

	for id, j := range q.jobs {
		if (j.Status == models.StatusCompleted || j.Status == models.StatusFailed || j.Status == models.StatusCancelled) && j.Created < oldestTime {
			oldestTime = j.Created
			oldestID = id
		}
	}

	if oldestID != "" {
		delete(q.jobs, oldestID)
	}
}
