# YOLO + YouTube Live Stream + Hono Server Integration

This script runs the vehicle detection and counting system with YouTube Live streaming output and server integration.

## Features

- Real-time object detection and tracking (YOLO + DeepSORT)
- Line crossing counting (people, motorcycles, cars)
- YouTube Live streaming via FFmpeg/RTMP
- Auto-restart before YouTube's 12-hour limit (for 24/7 streaming)
- Counter API integration for persisting counts
- Stream registration API for frontend listing

## Prerequisites

1. **YouTube API credentials**: `client_secrets.json` in project root
2. **CUDA-enabled GPU** for inference
3. **Running Hono server** for stream registration (optional)

## Environment Variables

### Required

| Variable                | Description                                 |
| ----------------------- | ------------------------------------------- |
| `COUNTER_API_KEY`       | API key for counter ingest endpoint         |
| `REGISTRATION_USERNAME` | Username for stream registration Basic Auth |
| `REGISTRATION_PASSWORD` | Password for stream registration Basic Auth |

### Optional

| Variable               | Default                                                      | Description                                      |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| `RTSP_SOURCE`          | `rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV`                    | RTSP stream URL                                  |
| `CAMERA_ID`            | `jalan-masuk-utama`                                          | Camera identifier                                |
| `COUNTER_API_URL`      | `https://cctv-vehicle-counter-api-production.up.railway.app` | Counter API URL                                  |
| `REGISTRATION_API_URL` | `https://ecocampus-proxy.up.railway.app`                     | Stream registration API URL                      |
| `YOUTUBE_TITLE`        | `CCTV {CAMERA_ID} - Vehicle Detection`                       | YouTube broadcast title                          |
| `YOUTUBE_PRIVACY`      | `unlisted`                                                   | YouTube privacy: `public`, `private`, `unlisted` |

## Usage

### Basic Usage

```bash
export COUNTER_API_KEY="your-api-key"
export REGISTRATION_USERNAME="admin"
export REGISTRATION_PASSWORD="secret"

./run-youtube-stream.sh
```

### Custom RTSP Source

```bash
export COUNTER_API_KEY="your-api-key"
export REGISTRATION_USERNAME="admin"
export REGISTRATION_PASSWORD="secret"
export RTSP_SOURCE="rtsp://192.168.1.100:554/stream"
export CAMERA_ID="parking-lot"

./run-youtube-stream.sh
```

### With Custom API URLs

```bash
export COUNTER_API_KEY="your-api-key"
export REGISTRATION_USERNAME="admin"
export REGISTRATION_PASSWORD="secret"
export COUNTER_API_URL="http://localhost:3001"
export REGISTRATION_API_URL="http://localhost:3000"

./run-youtube-stream.sh
```

## How It Works

1. **Startup**: Authenticates with YouTube API, creates a new broadcast/stream
2. **Detection**: Runs YOLO inference on RTSP frames, tracks objects, counts line crossings
3. **Streaming**: Encodes and pushes frames to YouTube via FFmpeg/RTMP
4. **Counter Ingest**: POSTs counts to counter API every 10 seconds
5. **Stream Registration**: POSTs stream info to registration API so frontend can list active streams
6. **Auto-restart**: Before hitting YouTube's 12-hour limit, automatically restarts with new stream
7. **Cleanup**: On stop/restart, DELETEs stream registration so frontend removes it from list

## API Endpoints Used

### Counter Ingest API

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

### Stream Registration API

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

## Troubleshooting

### "No module named 'src'"

Run from project root or ensure the script changes to the correct directory.

### YouTube authentication fails

Delete `youtube_token.pickle` and re-authenticate.

### FFmpeg crashes

Check that the RTSP source is accessible and the GPU has enough memory.

### Stream registration fails

Verify the Hono server is running and credentials are correct.
