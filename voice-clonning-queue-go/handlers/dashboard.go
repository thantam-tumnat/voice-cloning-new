package handlers

import (
	"github.com/gofiber/fiber/v2"

	"voice-cloning-queue/queue"
)

// DashboardHandler renders the HTML Web Dashboard for the Go Queue.
type DashboardHandler struct {
	q            *queue.PriorityQueue
	pythonGPUURL string
}

// NewDashboardHandler creates a new DashboardHandler.
func NewDashboardHandler(q *queue.PriorityQueue, pythonGPUURL string) *DashboardHandler {
	return &DashboardHandler{
		q:            q,
		pythonGPUURL: pythonGPUURL,
	}
}

// Index serves the real-time Queue Web Dashboard.
func (d *DashboardHandler) Index(c *fiber.Ctx) error {
	c.Set("Content-Type", "text/html; charset=utf-8")
	return c.SendString(dashboardHTML)
}

const dashboardHTML = `<!doctype html>
<html lang="th">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SiangTTS Go Queue Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #0b0f19;
      --bg-card: #111827;
      --bg-card-hover: #172236;
      --border-subtle: #1f293d;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent-cyan: #06b6d4;
      --accent-purple: #a855f7;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --accent-blue: #3b82f6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', 'Noto Sans Thai', system-ui, sans-serif;
      padding: 28px;
      min-height: 100vh;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border-subtle);
    }
    .title-group h1 {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge-go {
      background: linear-gradient(135deg, #00ADD8, #007d9c);
      color: #fff;
      font-size: 11px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .sub {
      color: var(--text-dim);
      font-size: 13px;
      margin-top: 4px;
    }
    .target-gpu {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: 'JetBrains Mono', monospace;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .dot-green { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
    .dot-red { background: var(--accent-red); box-shadow: 0 0 8px var(--accent-red); }

    /* Cards Grid */
    .cards-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .card:hover {
      transform: translateY(-2px);
      border-color: #334155;
    }
    .card-title {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .card-val {
      font-size: 26px;
      font-weight: 700;
      color: var(--text-main);
      font-family: 'JetBrains Mono', monospace;
    }
    .card-sub {
      font-size: 11px;
      color: var(--text-dim);
    }

    /* Running Banner */
    .running-banner {
      background: linear-gradient(90deg, rgba(6, 182, 212, 0.1), rgba(168, 85, 247, 0.05));
      border: 1px solid rgba(6, 182, 212, 0.25);
      border-radius: 12px;
      padding: 16px 20px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .running-info {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .pulse-ring {
      position: relative;
      width: 12px;
      height: 12px;
      background: var(--accent-cyan);
      border-radius: 50%;
      box-shadow: 0 0 10px var(--accent-cyan);
      animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.7); }
      70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(6, 182, 212, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(6, 182, 212, 0); }
    }

    /* Table */
    .table-container {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      overflow: hidden;
    }
    .table-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .table-title {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-main);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
    }
    th, td {
      padding: 12px 18px;
      border-bottom: 1px solid var(--border-subtle);
      font-size: 13px;
    }
    th {
      background: #0d1320;
      color: var(--text-muted);
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    tr:hover { background: var(--bg-card-hover); }
    .mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; }

    /* Badges */
    .pill {
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
    }
    .pill-queued { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .pill-running { background: rgba(6, 182, 212, 0.15); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.3); }
    .pill-completed { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .pill-failed { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .pill-cancelled { background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.3); }

    .lane-tag {
      font-size: 11px;
      font-weight: 600;
      padding: 2px 6px;
      border-radius: 4px;
    }
    .lane-interactive { background: rgba(168, 85, 247, 0.15); color: #c084fc; }
    .lane-batch { background: rgba(148, 163, 184, 0.1); color: #94a3b8; }
  </style>
</head>
<body>

  <div class="header">
    <div class="title-group">
      <h1>🚀 SiangTTS Central Queue Gateway <span class="badge-go">Go Fiber</span></h1>
      <div class="sub">High-Performance Non-blocking Traffic Controller & Priority Dispatcher (:8020)</div>
    </div>
    <div class="target-gpu" id="gpu-status">
      <span class="dot dot-green" id="gpu-dot"></span>
      <span id="gpu-label">Python GPU: Checking...</span>
    </div>
  </div>

  <div class="cards-grid">
    <div class="card">
      <div class="card-title">⚡ Interactive (Studio)</div>
      <div class="card-val" id="val-wait-interactive" style="color: var(--accent-purple);">0</div>
      <div class="card-sub">Priority Lane Queue</div>
    </div>
    <div class="card">
      <div class="card-title">📦 Batch (Webhook)</div>
      <div class="card-val" id="val-wait-batch" style="color: var(--accent-cyan);">0</div>
      <div class="card-sub">Background Queue</div>
    </div>
    <div class="card">
      <div class="card-title">⚙️ Running</div>
      <div class="card-val" id="val-running" style="color: var(--accent-amber);">0</div>
      <div class="card-sub">Active on GPU :8021</div>
    </div>
    <div class="card">
      <div class="card-title">✅ Completed</div>
      <div class="card-val" id="val-completed" style="color: var(--accent-green);">0</div>
      <div class="card-sub">Successful Jobs</div>
    </div>
    <div class="card">
      <div class="card-title">❌ Failed</div>
      <div class="card-val" id="val-failed" style="color: var(--accent-red);">0</div>
      <div class="card-sub">Errors or Cancelled</div>
    </div>
  </div>

  <div class="running-banner" id="running-container" style="display: none;">
    <div class="running-info">
      <div class="pulse-ring"></div>
      <div>
        <div style="font-weight: 700; font-size: 14px;" id="running-job-id">job_...</div>
        <div style="font-size: 12px; color: var(--text-muted);" id="running-details">Processing chunks on PyTorch CUDA...</div>
      </div>
    </div>
    <div class="mono" style="color: var(--accent-cyan); font-weight: 600;" id="running-time">0.0s</div>
  </div>

  <div class="table-container">
    <div class="table-header">
      <div class="table-title">Recent Jobs History</div>
      <div style="font-size: 12px; color: var(--text-dim);" id="last-sync">Syncing...</div>
    </div>
    <table>
      <thead>
        <tr>
          <th>Status</th>
          <th>Job ID</th>
          <th>Lane</th>
          <th>Client</th>
          <th>Chunks</th>
          <th>Waited</th>
          <th>Ran Time</th>
          <th>Created</th>
        </tr>
      </thead>
      <tbody id="jobs-tbody">
        <tr>
          <td colspan="8" style="text-align: center; color: var(--text-dim); padding: 24px;">No jobs in queue history yet.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <script>
    async function updateDashboard() {
      try {
        const res = await fetch('/v2/jobs');
        if (!res.ok) return;
        const data = await res.json();

        // Update counts
        const counts = data.counts || {};
        const waiting = data.waiting || {};
        document.getElementById('val-wait-interactive').innerText = waiting.interactive || 0;
        document.getElementById('val-wait-batch').innerText = waiting.batch || 0;
        document.getElementById('val-running').innerText = data.running ? 1 : 0;
        document.getElementById('val-completed').innerText = counts.completed || 0;
        document.getElementById('val-failed').innerText = (counts.failed || 0) + (counts.cancelled || 0);

        // Update Running banner
        const runBanner = document.getElementById('running-container');
        if (data.running) {
          runBanner.style.display = 'flex';
          document.getElementById('running-job-id').innerText = data.running.job_id;
          const chunksTxt = data.running.chunks && data.running.chunks[0] ? data.running.chunks[0].substring(0, 70) + '...' : '';
          document.getElementById('running-details').innerText = 'Lane: ' + data.running.lane + ' | ' + chunksTxt;
          document.getElementById('running-time').innerText = (data.running.ran_s ? data.running.ran_s.toFixed(1) : '0.0') + 's';
        } else {
          runBanner.style.display = 'none';
        }

        // Update Table
        var tbody = document.getElementById('jobs-tbody');
        if (!data.jobs || data.jobs.length === 0) {
          tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color: var(--text-dim); padding: 24px;">No jobs in queue history yet.</td></tr>';
        } else {
          var rowsHtml = '';
          for (var i = 0; i < data.jobs.length; i++) {
            var j = data.jobs[i];
            var statusClass = 'pill-' + j.status;
            var laneClass = 'lane-' + j.lane;
            var waited = (j.waited_s !== undefined && j.waited_s !== null) ? j.waited_s.toFixed(2) + 's' : '-';
            var ran = (j.ran_s !== undefined && j.ran_s !== null) ? j.ran_s.toFixed(2) + 's' : '-';
            var created = new Date(j.created * 1000).toLocaleTimeString();
            var totalChunks = j.total_chunks || (j.chunks ? j.chunks.length : 1);
            var clientName = j.client || '-';

            rowsHtml += '<tr>' +
              '<td><span class="pill ' + statusClass + '">' + j.status + '</span></td>' +
              '<td class="mono"><strong>' + j.job_id + '</strong></td>' +
              '<td><span class="lane-tag ' + laneClass + '">' + j.lane + '</span></td>' +
              '<td>' + clientName + '</td>' +
              '<td>' + totalChunks + '</td>' +
              '<td class="mono">' + waited + '</td>' +
              '<td class="mono">' + ran + '</td>' +
              '<td style="color: var(--text-dim);">' + created + '</td>' +
              '</tr>';
          }
          tbody.innerHTML = rowsHtml;
        }

        document.getElementById('last-sync').innerText = 'Live Auto-Sync: ' + new Date().toLocaleTimeString();
      } catch (err) {
        console.error('Failed to sync queue dashboard:', err);
      }
    }

    async function checkGPUHealth() {
      try {
        const res = await fetch('/health');
        const gpuDot = document.getElementById('gpu-dot');
        const gpuLabel = document.getElementById('gpu-label');
        if (res.ok) {
          const data = await res.json();
          gpuDot.className = 'dot dot-green';
          gpuLabel.innerText = 'Python GPU (:8021): ' + (data.device || 'Online') + ' | ' + (data.status || 'OK');
        } else {
          gpuDot.className = 'dot dot-red';
          gpuLabel.innerText = 'Python GPU (:8021): Offline';
        }
      } catch (e) {
        document.getElementById('gpu-dot').className = 'dot dot-red';
        document.getElementById('gpu-label').innerText = 'Python GPU (:8021): Unreachable';
      }
    }

    setInterval(updateDashboard, 1000);
    setInterval(checkGPUHealth, 3000);
    updateDashboard();
    checkGPUHealth();
  </script>
</body>
</html>`
