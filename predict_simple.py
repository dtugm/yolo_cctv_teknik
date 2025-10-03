#!/usr/bin/env python3
"""
Simple prediction script for running YOLO inference on video files or streams.
This is the most basic usage - just detection and tracking without HTTP streaming.
"""

import argparse
import sys
from pathlib import Path

# Add paths for src and deep_sort_pytorch
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics" / "yolo" / "v8" / "detect"))

from src.core.inference import InferenceEngine
from src.config.settings import InferenceConfig, TrackingConfig, VisualizationConfig


def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection and Tracking')
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
    parser.add_argument('--show', action='store_true',
                       help='Display results in window')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save results to file (default: True)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Inference image size (default: 640)')
    
    args = parser.parse_args()
    
    # Create configurations
    inference_config = InferenceConfig(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        image_size=args.imgsz
    )
    
    tracking_config = TrackingConfig()
    visualization_config = VisualizationConfig()
    
    # Create Hydra config for BasePredictor
    # Ensure imgsz is a list for compatibility
    imgsz = [args.imgsz, args.imgsz] if isinstance(args.imgsz, int) else args.imgsz
    
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
    
    print(f"🚀 Starting YOLO inference...")
    print(f"   Model: {args.model}")
    print(f"   Source: {args.source}")
    print(f"   Device: {args.device}")
    print(f"   Confidence: {args.conf}")
    print()
    
    # Initialize and run inference
    engine = InferenceEngine(
        inference_config=inference_config,
        tracking_config=tracking_config,
        visualization_config=visualization_config,
        hydra_config=hydra_config
    )
    
    # Run prediction
    results = engine()
    
    print("\n✅ Inference completed!")


if __name__ == "__main__":
    main()

