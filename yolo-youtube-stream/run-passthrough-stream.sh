#!/bin/bash
#
# Passthrough YouTube Live Stream (no inference)
#
# This script streams a video source directly to YouTube Live without
# any YOLO inference or object detection. Useful for cameras that only
# need live streaming without vehicle counting.
#
# Features:
# - Any OpenCV-compatible video source (RTSP, file, webcam)
# - YouTube Live streaming output
# - Stream registration for frontend listing
# - Auto-restart before YouTube's 12-hour limit
#
# Required environment variables:
#   REGISTRATION_USERNAME  - Username for stream registration Basic Auth
#   REGISTRATION_PASSWORD  - Password for stream registration Basic Auth
#
# Optional environment variables:
#   RTSP_SOURCE           - RTSP stream URL (default: rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV)
#   CAMERA_ID             - Camera identifier (default: jalan-masuk-utama)
#   REGISTRATION_API_URL  - Stream registration API URL (default: https://ecocampus-proxy.up.railway.app)
#   YOUTUBE_TITLE         - YouTube broadcast title
#   YOUTUBE_PRIVACY       - YouTube privacy setting: public, private, unlisted (default: unlisted)
#

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

# Default values
RTSP_SOURCE="${RTSP_SOURCE:-rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV}"
CAMERA_ID="${CAMERA_ID:-jalan-masuk-utama}"
REGISTRATION_API_URL="${REGISTRATION_API_URL:-https://ecocampus-proxy.up.railway.app}"
YOUTUBE_TITLE="${YOUTUBE_TITLE:-CCTV ${CAMERA_ID} - Live Stream}"
YOUTUBE_PRIVACY="${YOUTUBE_PRIVACY:-unlisted}"

# Validate required environment variables
if [ -z "$REGISTRATION_USERNAME" ]; then
    echo "Error: REGISTRATION_USERNAME environment variable is required"
    exit 1
fi

if [ -z "$REGISTRATION_PASSWORD" ]; then
    echo "Error: REGISTRATION_PASSWORD environment variable is required"
    exit 1
fi

echo "=============================================================="
echo "  Passthrough YouTube Live Stream (no inference)"
echo "=============================================================="
echo "  RTSP Source:       $RTSP_SOURCE"
echo "  Camera ID:         $CAMERA_ID"
echo "  Registration API:  $REGISTRATION_API_URL"
echo "  YouTube Title:     $YOUTUBE_TITLE"
echo "  YouTube Privacy:   $YOUTUBE_PRIVACY"
echo "=============================================================="
echo ""

# Run with restart loop (handles crashes, not YouTube 12h limit which is handled internally)
while true; do
    echo "Starting passthrough stream at $(date)"

    python passthrough_youtube_stream.py \
        --source "$RTSP_SOURCE" \
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
