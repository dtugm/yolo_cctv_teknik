#!/bin/bash

while true; do
    echo "Starting YOLO inference at $(date)"
    taskset -c 0-1 python predict_live_stream.py \
        --source rtsp://10.2.10.70:7447/rNVL3fqfqxknwEJt \
        --device cuda:0 \
        --port 5050 \
        --camera-id gerbang-eric \
        --counter-api-url "https://cctv-vehicle-counter-api-production.up.railway.app" \
        --counter-api-key "${COUNTER_API_KEY}"
    
    echo "Process stopped at $(date). Restarting in 5 seconds..."
    sleep 5
done
