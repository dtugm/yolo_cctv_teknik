#!/usr/bin/env python3
"""
Live streaming prediction script with HTTP server.
Runs YOLO inference and broadcasts results via HTTP MJPEG stream.
"""

import argparse
import sys
from pathlib import Path
import time

# Add paths for src and deep_sort_pytorch
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics" / "yolo" / "v8" / "detect"))

from src.core.inference import InferenceEngine
from src.config.settings import (
    InferenceConfig, TrackingConfig, 
    VisualizationConfig, StreamingConfig
)
from src.streaming.server import StreamingServer


class LiveStreamInference(InferenceEngine):
    """Extended inference engine with HTTP streaming support."""
    
    def __init__(self, streaming_server: StreamingServer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streaming_server = streaming_server
    
    def write_results(self, idx: int, preds, batch):
        """Override to push frames to streaming server."""
        log_string = super().write_results(idx, preds, batch)
        
        # Get the annotated frame from the annotator
        if hasattr(self, 'annotator') and self.annotator is not None:
            annotated_frame = self.annotator.result()
            # Push to streaming server
            if self.streaming_server and self.streaming_server.running:
                try:
                    self.streaming_server.add_frame(annotated_frame)
                except Exception as e:
                    print(f"Error pushing frame to stream: {e}")
        
        return log_string


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Object Detection with Live HTTP Streaming'
    )
    parser.add_argument('--source', type=str, required=True,
                       help='Video file path or RTSP stream URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLO model (default: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='IoU threshold for NMS (default: 0.7)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on: cuda or cpu (default: cuda)')
    parser.add_argument('--port', type=int, default=5050,
                       help='HTTP streaming port (default: 5050)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='HTTP server host (default: 0.0.0.0)')
    parser.add_argument('--show', action='store_true',
                       help='Also display results in window')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    parser.add_argument('--counter-direction', type=str, default='north_enter',
                       choices=['north_enter', 'south_enter'],
                       help='Counter direction: north_enter (North=Enter, South=Exit) or south_enter (South=Enter, North=Exit) (default: north_enter)')
    parser.add_argument('--show-info-panel', action='store_true', default=True,
                       help='Show system information panel (default: True)')
    parser.add_argument('--hide-info-panel', action='store_true',
                       help='Hide system information panel')
    
    args = parser.parse_args()
    
    # Handle info panel visibility logic
    show_info_panel = args.show_info_panel and not args.hide_info_panel
    
    # Create configurations
    inference_config = InferenceConfig(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    tracking_config = TrackingConfig()
    visualization_config = VisualizationConfig(
        counter_direction=args.counter_direction,
        show_info_panel=show_info_panel
    )
    
    streaming_config = StreamingConfig(
        enabled=True,
        host=args.host,
        port=args.port
    )
    
    # Hydra config for BasePredictor
    # Ensure imgsz is a list for compatibility
    imgsz = [640, 640]  # Default image size as list
    
    hydra_config = {
        'model': args.model,
        'source': args.source,
        'conf': args.conf,
        'iou': args.iou,
        'device': args.device,
        'imgsz': imgsz,
        'show': args.show,
        'save': args.save,
    }
    
    print(f"🚀 Starting YOLO Live Streaming Inference...")
    print(f"   Model: {args.model}")
    print(f"   Source: {args.source}")
    print(f"   Device: {args.device}")
    print(f"   Stream: http://{args.host}:{args.port}/_yolo_stream/")
    print()
    
    # Initialize streaming server
    streaming_server = StreamingServer(streaming_config)
    streaming_server.start()
    
    # Wait a moment for server to initialize
    time.sleep(1)
    
    # Initialize inference engine with streaming
    engine = LiveStreamInference(
        streaming_server=streaming_server,
        inference_config=inference_config,
        tracking_config=tracking_config,
        visualization_config=visualization_config,
        hydra_config=hydra_config
    )
    
    try:
        # Run prediction
        results = engine()
        print("\n✅ Inference completed!")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        streaming_server.stop()


if __name__ == "__main__":
    main()

