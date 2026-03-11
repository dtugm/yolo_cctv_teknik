#!/bin/bash

while true; do
    echo "Starting YOLO inference at $(date)"
    taskset -c 0-1 python predict_live_stream.py \
        --source rtsp://10.2.10.70:7447/AOAw5Ce31UIYCVDH \
        --device cuda:0 \
        --port 5051 \
        --camera-id tugu-teknik \
        --counter-api-url "https://cctv-vehicle-counter-api-production.up.railway.app"
    
    echo "Process stopped at $(date). Restarting in 5 seconds..."
    sleep 5
done
