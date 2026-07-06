# MSP Hotsaw — Real-Time Blade & Item Cut Counter

A computer-vision production monitoring system for a steel hot-saw cutting line. It watches an
RTSP camera feed with a YOLO object detector, automatically counts blade cuts and per-item
production, and serves a live web dashboard with historical reporting — built to run unattended
on factory floor hardware for weeks at a time.

## Why this project

MSP Raigarh runs two hot-saw cutting stations. Operators previously logged blade resets and
item counts by hand on paper. This system replaces that with automatic, camera-based counting,
persistent session state (survives restarts/power loss), and exportable CSV/Excel reports —
with zero manual data entry.

## Features

- **Live YOLO detection** on an RTSP/GStreamer camera feed to detect blade cuts and
  distinguish front-face vs. end-face cutting events.
- **Real-time dashboard** (Flask + Server-Sent Events) showing current blade count, item
  counts, and a live annotated video feed — updates instantly on every cut, no page refresh.
- **Two independent report layers**:
  - *Blade Resets* — one row per blade lifecycle (start → end, total cuts, item breakdown).
  - *Item Wise* — one row per production session, one column per item type, updated live.
- **Crash-safe session state** — current counts are persisted to disk after every cut, so a
  power cut or restart resumes exactly where it left off instead of losing the shift's data.
- **CSV + combined Excel report export**, downloadable directly from the dashboard.
- **Self-healing report storage** — automatically detects and migrates outdated CSV schemas,
  and falls back to a legacy data folder if the primary one is empty, so historical records
  are never silently lost.

## Architecture

```
┌─────────────┐     RTSP/GStreamer     ┌──────────────────┐
│  IP Camera   │ ─────────────────────▶ │ Detection Thread  │
└─────────────┘                        │ (YOLO + OpenCV)   │
                                        └─────────┬─────────┘
                                                   │ cut events
                                                   ▼
                                        ┌──────────────────┐        ┌───────────────┐
                                        │  Shared State     │◀──────▶│  reports/      │
                                        │  (thread-safe)     │        │  CSV + JSON    │
                                        └─────────┬─────────┘        └───────────────┘
                                                   │ SSE push
                                                   ▼
                                        ┌──────────────────┐
                                        │  Flask Dashboard   │
                                        │  (live video +     │
                                        │   logs + export)   │
                                        └──────────────────┘
```

Two background threads run per station: one pulls frames from the camera and runs YOLO
inference, the other keeps a lightweight billet/DB refresh loop going. Flask serves the
dashboard, video stream, and JSON APIs on the main thread. All shared counters are guarded by
a lock, and every CSV write goes through a dedicated lock so concurrent events can never
corrupt the report files.

## Tech stack

| Layer            | Tech                                   |
|-------------------|-----------------------------------------|
| Computer vision   | YOLO (Ultralytics), OpenCV, GStreamer   |
| Backend           | Python, Flask, Server-Sent Events      |
| Storage           | CSV / JSON (append-only reports), SQLite |
| Reporting export  | openpyxl (Excel)                        |
| Frontend          | Vanilla JS + HTML/CSS (no build step)   |

## Project structure

```
.
├── merged88.py        # Hotsaw 2 station (single 2-class YOLO model)
├── merged89.py         # Hotsaw 1 station (separate endface/frontface models)
├── requirements.txt
├── .env.example        # Camera credential template (copy to .env, never commit .env)
└── reports/            # Generated at runtime - CSV/XLSX/JSON reports (gitignored)
```

`merged88.py` and `merged89.py` are two instances of the same dashboard, each pointed at a
different physical camera/hot-saw station (they run as separate processes on separate ports).

## Setup

1. **Clone and install dependencies**
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure your camera**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your actual camera username/password/IP. This file is gitignored and
   stays local to your machine.

3. **Add your YOLO weights**
   Place your trained `.pt` model file(s) next to the script (paths are configured at the top
   of each file). Model weights are not tracked in git — store them separately (e.g. Git LFS,
   cloud storage, or copy manually to the deployment machine).

4. **Run**
   ```bash
   python3 merged88.py     # Hotsaw 2, dashboard on http://localhost:5025
   python3 merged89.py     # Hotsaw 1, dashboard on http://localhost:5024
   ```

## API endpoints

| Endpoint                     | Description                              |
|-------------------------------|-------------------------------------------|
| `GET /`                        | Main dashboard UI                        |
| `GET /vision`                  | Raw annotated video feed                 |
| `GET /events`                   | Server-Sent Events stream (live updates) |
| `GET /api/state`               | Current session state (JSON)             |
| `GET /api/blade_logs`          | Blade reset history (JSON)               |
| `GET /api/item_logs`           | Item-wise session history (JSON)         |
| `POST /reset_blade`            | Reset blade count, start new session     |
| `POST /reset_item`             | Save current item report, switch item    |
| `POST /change_item`            | Switch item without resetting counts     |
| `GET /download/combined_excel` | Download full report as one workbook     |

## Notes on security

Camera credentials are read from environment variables (`.env`, gitignored) rather than being
hardcoded — this repo is safe to make public. If you fork this for your own deployment, never
commit a `.env` file or paste real credentials into the source.

## License

MIT — see [LICENSE](LICENSE).
