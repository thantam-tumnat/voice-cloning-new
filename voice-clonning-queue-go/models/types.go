package models

import "time"

// VoiceSpec mirrors the Python GPU service voice specification.
type VoiceSpec struct {
	Handle       *string `json:"handle,omitempty"`
	SpeakerID    *string `json:"speaker_id,omitempty"`
	RefText      *string `json:"ref_text,omitempty"`
	AllowSidecar bool    `json:"allow_sidecar"`
	Seed         bool    `json:"seed"`
}

// OutputSpec defines whether output is returned in-memory (npz/arrays) or files on disk.
type OutputSpec struct {
	Mode   string   `json:"mode"`
	JobDir *string  `json:"job_dir,omitempty"`
	Names  []string `json:"names,omitempty"`
}

// RenderRequest is the incoming payload for /v2/jobs/render.
type RenderRequest struct {
	JobID      *string     `json:"job_id,omitempty"`
	Chunks     []string    `json:"chunks"`
	Voice      *VoiceSpec  `json:"voice,omitempty"`
	CFGValue   float64     `json:"cfg_value"`
	Timesteps  int         `json:"timesteps"`
	LoRA       interface{} `json:"lora,omitempty"`
	Output     OutputSpec  `json:"output"`
	Lane       string      `json:"lane"`
	Client     string      `json:"client"`
}

// JobStatus represents the state of a render job.
type JobStatus string

const (
	StatusQueued    JobStatus = "queued"
	StatusRunning   JobStatus = "running"
	StatusCompleted JobStatus = "completed"
	StatusFailed    JobStatus = "failed"
	StatusCancelled JobStatus = "cancelled"
)

// RenderJob represents an active or completed job in the queue.
type RenderJob struct {
	JobID       string                 `json:"job_id"`
	Chunks      []string               `json:"chunks"`
	Voice       *VoiceSpec             `json:"voice,omitempty"`
	CFGValue    float64                `json:"cfg_value"`
	Timesteps   int                    `json:"timesteps"`
	LoRA        interface{}            `json:"lora,omitempty"`
	Output      OutputSpec             `json:"output"`
	Lane        string                 `json:"lane"`
	Client      string                 `json:"client"`
	Status      JobStatus              `json:"status"`
	Position    *int                   `json:"position,omitempty"`
	Error       *string                `json:"error,omitempty"`
	Result      map[string]interface{} `json:"result,omitempty"`
	Payload     []byte                 `json:"-"`
	ChunksDone  int                    `json:"chunks_done"`
	TotalChunks int                    `json:"total_chunks"`
	Created     float64                `json:"created"`
	Started     *float64               `json:"started,omitempty"`
	Finished    *float64               `json:"finished,omitempty"`
	DoneChan    chan struct{}          `json:"-"`
}

// NewRenderJob creates an initialized RenderJob.
func NewRenderJob(req RenderRequest, jobID string) *RenderJob {
	now := float64(time.Now().UnixNano()) / 1e9
	cfg := req.CFGValue
	if cfg <= 0 {
		cfg = 2.0
	}
	steps := req.Timesteps
	if steps <= 0 {
		steps = 10
	}
	lane := req.Lane
	if lane != "interactive" && lane != "batch" {
		lane = "batch"
	}
	outMode := req.Output.Mode
	if outMode == "" {
		outMode = "npz"
	}
	output := req.Output
	output.Mode = outMode

	return &RenderJob{
		JobID:       jobID,
		Chunks:      req.Chunks,
		Voice:       req.Voice,
		CFGValue:    cfg,
		Timesteps:   steps,
		LoRA:        req.LoRA,
		Output:      output,
		Lane:        lane,
		Client:      req.Client,
		Status:      StatusQueued,
		TotalChunks: len(req.Chunks),
		Created:     now,
		DoneChan:    make(chan struct{}),
	}
}

// AsDict converts a job into a JSON-friendly map with position.
func (j *RenderJob) AsDict(position *int) map[string]interface{} {
	now := float64(time.Now().UnixNano()) / 1e9
	waited := now - j.Created
	if j.Started != nil {
		waited = *j.Started - j.Created
	}

	var ran *float64
	if j.Started != nil {
		var r float64
		if j.Finished != nil {
			r = *j.Finished - *j.Started
		} else {
			r = now - *j.Started
		}
		ran = &r
	}

	res := map[string]interface{}{
		"job_id":       j.JobID,
		"status":       j.Status,
		"chunks":       j.Chunks,
		"chunks_done":  j.ChunksDone,
		"total_chunks": j.TotalChunks,
		"lane":         j.Lane,
		"client":       j.Client,
		"created":      j.Created,
		"started":      j.Started,
		"finished":     j.Finished,
		"waited_s":     waited,
		"ran_s":        ran,
	}

	if j.Status == StatusQueued && position != nil {
		res["position"] = *position
	}
	if j.Error != nil {
		res["error"] = *j.Error
	}
	if j.Result != nil {
		res["result"] = j.Result
	}
	return res
}
