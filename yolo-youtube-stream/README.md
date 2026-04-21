# YouTube Live Streaming Scripts

This directory contains two streaming scripts:

1. **`run-youtube-stream.sh`** — YOLO inference + YouTube Live streaming (detection, counting, streaming)
2. **`run-passthrough-stream.sh`** — Passthrough YouTube Live streaming (no inference, just restream)

---

## Passthrough Stream (no inference)

Streams any video source directly to YouTube Live without running YOLO detection. Useful for cameras that only need live streaming without vehicle counting.

### Features

- Any OpenCV-compatible video source (RTSP, file, webcam)
- YouTube Live streaming via FFmpeg/RTMP
- Auto-restart before YouTube's 12-hour limit (for 24/7 streaming)
- Stream registration API for frontend listing
- RTSP auto-reconnect on connection loss
- No GPU required

### Prerequisites

1. **YouTube API credentials**: `client_secrets.json` in project root
2. **Python dependencies**: `opencv-python`, `requests`, `google-api-python-client`, `google-auth-oauthlib`

### Environment Variables

#### Required

| Variable                | Description                                 |
| ----------------------- | ------------------------------------------- |
| `REGISTRATION_USERNAME` | Username for stream registration Basic Auth |
| `REGISTRATION_PASSWORD` | Password for stream registration Basic Auth |

#### Optional

| Variable               | Default                                  | Description                                      |
| ---------------------- | ---------------------------------------- | ------------------------------------------------ |
| `RTSP_SOURCE`          | `rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV` | RTSP stream URL                                  |
| `CAMERA_ID`            | `jalan-masuk-utama`                      | Camera identifier (used in YouTube title)        |
| `REGISTRATION_API_URL` | `https://ecocampus-proxy.up.railway.app` | Stream registration API URL                      |
| `YOUTUBE_TITLE`        | `CCTV {CAMERA_ID} - Live Stream`         | YouTube broadcast title                          |
| `YOUTUBE_PRIVACY`      | `unlisted`                               | YouTube privacy: `public`, `private`, `unlisted` |

### Usage

#### Basic Usage

```bash
export REGISTRATION_USERNAME="admin"
export REGISTRATION_PASSWORD="secret"

./run-passthrough-stream.sh
```

#### Custom RTSP Source

```bash
export REGISTRATION_USERNAME="admin"
export REGISTRATION_PASSWORD="secret"
export RTSP_SOURCE="rtsp://192.168.1.100:554/stream"
export CAMERA_ID="parking-lot"

./run-passthrough-stream.sh
```

#### Using the Python Script Directly

```bash
# RTSP source
python passthrough_youtube_stream.py \
    --source "rtsp://192.168.1.100:554/stream" \
    --youtube-title "Parking Lot Cam" \
    --youtube-privacy unlisted \
    --youtube-resolution 720p \
    --target-fps 30

# Video file
python passthrough_youtube_stream.py \
    --source "/path/to/video.mp4" \
    --youtube-title "Recorded Video Stream"

# Webcam (device index)
python passthrough_youtube_stream.py \
    --source 0 \
    --youtube-title "Webcam Stream"
```

#### With Stream Registration

```bash
python passthrough_youtube_stream.py \
    --source "rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV" \
    --registration-api-url "https://ecocampus-proxy.up.railway.app" \
    --registration-username "admin" \
    --registration-password "secret" \
    --youtube-title "CCTV Live"
```

### How It Works

1. **Startup**: Opens video source with OpenCV, authenticates with YouTube API, creates broadcast
2. **Streaming**: Reads frames and pushes them to YouTube via FFmpeg/RTMP
3. **Stream Registration**: POSTs stream info so frontend can list active streams
4. **Auto-restart**: Before hitting YouTube's 12-hour limit, automatically restarts with a new broadcast
5. **Reconnect**: If RTSP connection drops, automatically retries
6. **Cleanup**: On stop/restart, DELETEs stream registration from frontend

---

## YOLO + YouTube Live Stream + Hono Server Integration

Runs the vehicle detection and counting system with YouTube Live streaming output and server integration.

### Features

- Real-time object detection and tracking (YOLO + DeepSORT)
- Line crossing counting (people, motorcycles, cars)
- YouTube Live streaming via FFmpeg/RTMP
- Auto-restart before YouTube's 12-hour limit (for 24/7 streaming)
- Counter API integration for persisting counts
- Stream registration API for frontend listing

### Prerequisites

1. **YouTube API credentials**: `client_secrets.json` in project root
2. **CUDA-enabled GPU** for inference
3. **Running Hono server** for stream registration (optional)

### Environment Variables

#### Required

| Variable                | Description                                 |
| ----------------------- | ------------------------------------------- |
| `COUNTER_API_KEY`       | API key for counter ingest endpoint         |
| `REGISTRATION_USERNAME` | Username for stream registration Basic Auth |
| `REGISTRATION_PASSWORD` | Password for stream registration Basic Auth |

#### Optional

| Variable               | Default                                                      | Description                                      |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| `RTSP_SOURCE`          | `rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV`                    | RTSP stream URL                                  |
| `CAMERA_ID`            | `jalan-masuk-utama`                                          | Camera identifier                                |
| `COUNTER_API_URL`      | `https://cctv-vehicle-counter-api-production.up.railway.app` | Counter API URL                                  |
| `REGISTRATION_API_URL` | `https://ecocampus-proxy.up.railway.app`                     | Stream registration API URL                      |
| `YOUTUBE_TITLE`        | `CCTV {CAMERA_ID} - Vehicle Detection`                       | YouTube broadcast title                          |
| `YOUTUBE_PRIVACY`      | `unlisted`                                                   | YouTube privacy: `public`, `private`, `unlisted` |

### Usage

#### Basic Usage

```bash
export COUNTER_API_KEY="your-api-key"
export REGISTRATION_USERNAME="admin"
export REGISTRATION_PASSWORD="secret"

./run-youtube-stream.sh
```

#### Custom RTSP Source

```bash
export COUNTER_API_KEY="your-api-key"
export REGISTRATION_USERNAME="admin"
export REGISTRATION_PASSWORD="secret"
export RTSP_SOURCE="rtsp://192.168.1.100:554/stream"
export CAMERA_ID="parking-lot"

./run-youtube-stream.sh
```

#### With Custom API URLs

```bash
export COUNTER_API_KEY="your-api-key"
export REGISTRATION_USERNAME="admin"
export REGISTRATION_PASSWORD="secret"
export COUNTER_API_URL="http://localhost:3001"
export REGISTRATION_API_URL="http://localhost:3000"

./run-youtube-stream.sh
```

### How It Works

1. **Startup**: Authenticates with YouTube API, creates a new broadcast/stream
2. **Detection**: Runs YOLO inference on RTSP frames, tracks objects, counts line crossings
3. **Streaming**: Encodes and pushes frames to YouTube via FFmpeg/RTMP
4. **Counter Ingest**: POSTs counts to counter API every 10 seconds
5. **Stream Registration**: POSTs stream info to registration API so frontend can list active streams
6. **Auto-restart**: Before hitting YouTube's 12-hour limit, automatically restarts with new stream
7. **Cleanup**: On stop/restart, DELETEs stream registration so frontend removes it from list

### API Endpoints Used

#### Counter Ingest API

```
POST {COUNTER_API_URL}/api/ingest
Headers: X-API-Key: {COUNTER_API_KEY}
Body: {
  "camera_id": "jalan-masuk-utama",
  "counts": {
    "people_in": 10, "people_out": 5,
    "motor_in": 20, "motor_out": 15,
    "car_in": 8, "car_out": 6
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Stream Registration API

```
POST {REGISTRATION_API_URL}/api/streaming/
Headers: Authorization: Basic {base64(username:password)}
Body: {
  "streamingId": "youtube-broadcast-id",
  "title": "CCTV jalan-masuk-utama - Vehicle Detection",
  "description": "..."
}

DELETE {REGISTRATION_API_URL}/api/streaming/{streamingId}
Headers: Authorization: Basic {base64(username:password)}
```

---

## Troubleshooting (both scripts)

### "No module named 'src'"

Run from project root or ensure the script changes to the correct directory.

### YouTube authentication fails

Delete `youtube_token.pickle` and re-authenticate.

### FFmpeg crashes

Check that the RTSP source is accessible. For the inference script, also check GPU memory.

### Stream registration fails

Verify the Hono server is running and credentials are correct.

### Passthrough: "Could not open video source"

Verify the RTSP URL is reachable (`ffprobe rtsp://...`) or that the file path exists.
