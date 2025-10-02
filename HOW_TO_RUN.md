# How to Run - YOLO CCTV System

## 🚀 Quick Command Reference

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Commands

#### 1. Video File - Simple Prediction

```bash
# Basic usage
python predict_simple.py --source datasets/test.mp4

# With display window
python predict_simple.py --source datasets/test.mp4 --show

# Save results
python predict_simple.py --source datasets/test.mp4 --save
```

#### 2. RTSP Stream - Simple Prediction

```bash
# Basic RTSP
python predict_simple.py --source "rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV"

# With custom confidence
python predict_simple.py \
    --source "rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV" \
    --conf 0.3 \
    --device cuda
```

#### 3. Live HTTP Streaming

```bash
# Video file with streaming
python predict_live_stream.py --source datasets/test.mp4 --port 5050

# RTSP with streaming
python predict_live_stream.py \
    --source "rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV" \
    --port 5050

# Then open: http://localhost:5050/_yolo_stream/
```

### Advanced Usage

#### Different Models

```bash
# Nano (fastest)
python predict_simple.py --source video.mp4 --model yolov8n.pt

# Small
python predict_simple.py --source video.mp4 --model yolov8s.pt

# Medium
python predict_simple.py --source video.mp4 --model yolov8m.pt

# Large (most accurate)
python predict_simple.py --source video.mp4 --model yolov8l.pt

# Extra Large
python predict_simple.py --source video.mp4 --model yolov8x.pt
```

#### Device Selection

```bash
# GPU (default)
python predict_simple.py --source video.mp4 --device cuda

# Specific GPU
python predict_simple.py --source video.mp4 --device cuda:0

# CPU
python predict_simple.py --source video.mp4 --device cpu
```

#### Adjust Detection Sensitivity

```bash
# Lower confidence = More detections (may include false positives)
python predict_simple.py --source video.mp4 --conf 0.1

# Default confidence
python predict_simple.py --source video.mp4 --conf 0.25

# Higher confidence = Fewer, more certain detections
python predict_simple.py --source video.mp4 --conf 0.5
```

#### Image Size

```bash
# Smaller = Faster, less accurate
python predict_simple.py --source video.mp4 --imgsz 416

# Default
python predict_simple.py --source video.mp4 --imgsz 640

# Larger = Slower, more accurate
python predict_simple.py --source video.mp4 --imgsz 1280
```

### Saved Results

```
runs/detect/predict/
├── test.mp4          # Annotated video
└── labels/           # Detection labels (if --save-txt)
```

### HTTP Stream Access

- **Main page**: `http://localhost:5050/_yolo_stream/`
- **Direct feed**: `http://localhost:5050/_yolo_stream/video_feed`
- **Snapshot**: `http://localhost:5050/_yolo_stream/snapshot.jpg`

## 🔧 Python API Usage

### Basic Example

```python
from src.core.inference import InferenceEngine
from src.config.settings import InferenceConfig, TrackingConfig, VisualizationConfig

# Configure
inference_config = InferenceConfig(
    model_path="yolov8n.pt",
    confidence_threshold=0.25,
    device="cuda"
)

# Create hydra config
hydra_config = {
    'source': 'datasets/test.mp4',
    'model': 'yolov8n.pt',
    'show': False,
    'save': True
}

# Run inference
engine = InferenceEngine(
    inference_config=inference_config,
    hydra_config=hydra_config
)

results = engine()
```

### With Streaming

```python
from src.core.inference import InferenceEngine
from src.streaming.server import StreamingServer
from src.config.settings import InferenceConfig, StreamingConfig

# Setup streaming
streaming_config = StreamingConfig(enabled=True, port=5050)
streaming_server = StreamingServer(streaming_config)
streaming_server.start()

# Configure inference
inference_config = InferenceConfig(model_path="yolov8n.pt")
hydra_config = {'source': 'rtsp://camera/stream'}

# Create custom engine with streaming
class StreamingInference(InferenceEngine):
    def __init__(self, streaming_server, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streaming_server = streaming_server

    def write_results(self, idx, preds, batch):
        log_string = super().write_results(idx, preds, batch)
        if hasattr(self, 'annotator'):
            frame = self.annotator.result()
            self.streaming_server.add_frame(frame)
        return log_string

# Run
engine = StreamingInference(
    streaming_server=streaming_server,
    inference_config=inference_config,
    hydra_config=hydra_config
)
engine()
```

### Check CUDA Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Find Available Ports

```bash
# Linux/macOS
lsof -i :5050

# Windows
netstat -ano | findstr :5050
```

### Kill Process on Port

```bash
# Linux/macOS
lsof -ti:5050 | xargs kill -9

# Windows
# Find PID first, then:
taskkill /PID <PID> /F
```

## 📁 File Locations

### Input

- Video files: Place in `datasets/`
- YOLO models: Project root or specify path
- DeepSORT checkpoint: `ultralytics/yolo/v8/detect/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7`

### Output

- Predictions: `runs/detect/predict*/`
- Labels: `runs/detect/predict*/labels/` (if `--save-txt`)
- Logs: Console output

### Configuration

- Main configs: `src/config/settings.py`
- Model configs: `ultralytics/models/v8/*.yaml`
- DeepSORT config: `ultralytics/yolo/v8/detect/deep_sort_pytorch/configs/deep_sort.yaml`

## 📝 Common Workflows

### Workflow 1: Test on Sample Video

```bash
# 1. Run prediction
python predict_simple.py --source datasets/test.mp4 --show

# 2. Check results
ls -lh runs/detect/predict/
```

### Workflow 2: Stream RTSP Camera

```bash
# 1. Test connection
python test_connection_rtsp.py

# 2. Start streaming
python predict_live_stream.py \
    --source "rtsp://camera-ip/stream" \
    --port 5050

# 3. Open browser
open http://localhost:5050/_yolo_stream/
```

### Workflow 3: Compare Models

```bash
# Test different models
for model in yolov8n.pt yolov8s.pt yolov8m.pt; do
    echo "Testing $model..."
    python predict_simple.py \
        --source datasets/test.mp4 \
        --model $model \
        --save
done

# Compare results in runs/detect/
```
