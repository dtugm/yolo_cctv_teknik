#!/bin/bash
#
# YOLO + YouTube Live Stream + Hono Server Integration
#
# This script runs the vehicle detection and counting system with:
# - YOLO object detection and tracking
# - YouTube Live streaming output
# - Counter API integration for persistence
# - Stream registration for frontend listing
# - Auto-restart before YouTube's 12-hour limit
#
# Required environment variables:
#   COUNTER_API_KEY        - API key for counter ingest endpoint
#   REGISTRATION_USERNAME  - Username for stream registration Basic Auth
#   REGISTRATION_PASSWORD  - Password for stream registration Basic Auth
#
# Optional environment variables:
#   RTSP_SOURCE           - RTSP stream URL (default: rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV)
#   CAMERA_ID             - Camera identifier (default: jalan-masuk-utama)
#   COUNTER_API_URL       - Counter API URL (default: https://cctv-vehicle-counter-api-production.up.railway.app)
#   REGISTRATION_API_URL  - Stream registration API URL (default: http://localhost:3000)
#   YOUTUBE_TITLE         - YouTube broadcast title
#   YOUTUBE_PRIVACY       - YouTube privacy setting: public, private, unlisted (default: unlisted)
#

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

# Default values
RTSP_SOURCE="${RTSP_SOURCE:-rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV}"
CAMERA_ID="${CAMERA_ID:-jalan-masuk-utama}"
COUNTER_API_URL="${COUNTER_API_URL:-https://cctv-vehicle-counter-api-production.up.railway.app}"
REGISTRATION_API_URL="${REGISTRATION_API_URL:-https://ecocampus-proxy.up.railway.app}"
YOUTUBE_TITLE="${YOUTUBE_TITLE:-CCTV ${CAMERA_ID} - Vehicle Detection}"
YOUTUBE_PRIVACY="${YOUTUBE_PRIVACY:-unlisted}"

# Validate required environment variables
if [ -z "$COUNTER_API_KEY" ]; then
    echo "Error: COUNTER_API_KEY environment variable is required"
    exit 1
fi

if [ -z "$REGISTRATION_USERNAME" ]; then
    echo "Error: REGISTRATION_USERNAME environment variable is required"
    exit 1
fi

if [ -z "$REGISTRATION_PASSWORD" ]; then
    echo "Error: REGISTRATION_PASSWORD environment variable is required"
    exit 1
fi

echo "=============================================================="
echo "  YOLO + YouTube Live Stream + Hono Server"
echo "=============================================================="
echo "  RTSP Source:       $RTSP_SOURCE"
echo "  Camera ID:         $CAMERA_ID"
echo "  Counter API:       $COUNTER_API_URL"
echo "  Registration API:  $REGISTRATION_API_URL"
echo "  YouTube Title:     $YOUTUBE_TITLE"
echo "  YouTube Privacy:   $YOUTUBE_PRIVACY"
echo "=============================================================="
echo ""

# Run with restart loop (handles crashes, not YouTube 12h limit which is handled internally)
while true; do
    echo "Starting YOLO + YouTube stream at $(date)"

    taskset -c 0-1 python predict_youtube_stream.py \
        --source "$RTSP_SOURCE" \
        --device cuda:0 \
        --camera-id "$CAMERA_ID" \
        --counter-api-url "$COUNTER_API_URL" \
        --counter-api-key "$COUNTER_API_KEY" \
        --registration-api-url "$REGISTRATION_API_URL" \
        --registration-username "$REGISTRATION_USERNAME" \
        --registration-password "$REGISTRATION_PASSWORD" \
        --youtube-title "$YOUTUBE_TITLE" \
        --youtube-privacy "$YOUTUBE_PRIVACY" \
        --youtube-resolution 720p \
        --youtube-bitrate 2500

    EXIT_CODE=$?
    echo "Process exited with code $EXIT_CODE at $(date). Restarting in 5 seconds..."
    sleep 5
done
