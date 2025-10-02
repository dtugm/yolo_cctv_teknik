# Quick Start Guide - YOLO CCTV System

Get up and running in 5 minutes! 🚀

## 📦 Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download model (optional - auto-downloads on first run)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## 🎬 Basic Usage

### Option 1: Video File

```bash
python predict_simple.py --source datasets/test.mp4 --show
```

### Option 2: RTSP Stream

```bash
python predict_simple.py --source "rtsp://10.2.10.70:7447/stream"
```

### Option 3: Live HTTP Streaming

```bash
python predict_live_stream.py --source datasets/test.mp4 --port 5050
# Open browser: http://localhost:5050/_yolo_stream/
```

## ⚙️ Common Commands

### Different Models

```bash
# Fast (nano)
python predict_simple.py --source video.mp4 --model yolov8n.pt

# Accurate (large)
python predict_simple.py --source video.mp4 --model yolov8l.pt
```

### Device Selection

```bash
# GPU (default)
python predict_simple.py --source video.mp4 --device cuda

# CPU
python predict_simple.py --source video.mp4 --device cpu
```

### Confidence Threshold

```bash
# More detections
python predict_simple.py --source video.mp4 --conf 0.15

# Fewer, confident detections
python predict_simple.py --source video.mp4 --conf 0.5
```

### Save Results

```bash
# Save to file
python predict_simple.py --source video.mp4 --save

# Results saved in: runs/detect/predict/
```

## 🌐 Streaming Options

### Default Port

```bash
python predict_live_stream.py --source rtsp://camera/stream
# Access: http://localhost:5050/_yolo_stream/
```

### Custom Port

```bash
python predict_live_stream.py --source video.mp4 --port 8080
# Access: http://localhost:8080/_yolo_stream/
```

### Access from Other Devices

```bash
# Listen on all interfaces
python predict_live_stream.py --source video.mp4 --host 0.0.0.0 --port 5050
# Access: http://your-ip:5050/_yolo_stream/
```

### RTSP Not Working

```bash
# Test with test script first
python test_connection_rtsp.py
# Update RTSP URL in the script
```

## 📁 Project Structure (Key Files)

```
yolo_cctv_teknik/
├── predict_simple.py          # Basic inference
├── predict_live_stream.py     # With HTTP streaming
├── src/                        # Refactored modular code
│   ├── config/                # Settings
│   ├── core/                  # Inference engine
│   ├── tracking/              # Object tracking
│   ├── visualization/         # UI rendering
│   └── streaming/             # HTTP server
├── datasets/                   # Your videos
└── runs/                      # Output results
```

## 🔧 Configuration

Edit `src/config/settings.py` to customize:

```python
# Example: Change confidence threshold
@dataclass
class InferenceConfig:
    confidence_threshold: float = 0.30  # Changed from 0.25

# Example: Adjust counting line position
@dataclass
class VisualizationConfig:
    counting_line_start: Tuple[float, float] = (0.2, 0.5)  # x, y fractions
```
