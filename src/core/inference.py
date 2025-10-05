"""Core inference engine integrating YOLO detection with tracking and visualization."""

import sys
from pathlib import Path
import time
from typing import Optional, Dict, Any
import numpy as np
import torch
import cv2

# Add paths for ultralytics and deep_sort_pytorch
ULTRALYTICS_PATH = Path(__file__).parent.parent.parent / "ultralytics"
DEEP_SORT_PATH = Path(__file__).parent.parent.parent / "ultralytics" / "yolo" / "v8" / "detect"
sys.path.insert(0, str(ULTRALYTICS_PATH))
sys.path.insert(0, str(DEEP_SORT_PATH))

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator

from src.config.settings import InferenceConfig, TrackingConfig, VisualizationConfig
from src.tracking.tracker import ObjectTracker
from src.visualization.renderer import DetectionRenderer
from src.utils.geometry import xyxy_to_xywh


class InferenceEngine(BasePredictor):
    """
    Main inference engine that combines YOLO detection with tracking and visualization.
    Extends BasePredictor to integrate with existing YOLO pipeline.
    """
    
    def __init__(
        self,
        inference_config: Optional[InferenceConfig] = None,
        tracking_config: Optional[TrackingConfig] = None,
        visualization_config: Optional[VisualizationConfig] = None,
        hydra_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the inference engine.
        
        Args:
            inference_config: Inference configuration
            tracking_config: Tracking configuration
            visualization_config: Visualization configuration
            hydra_config: Hydra configuration dict for BasePredictor
        """
        # Initialize configs
        self.inference_config = inference_config or InferenceConfig()
        self.tracking_config = tracking_config or TrackingConfig()
        self.visualization_config = visualization_config or VisualizationConfig()
        
        # Initialize BasePredictor with hydra config
        if hydra_config is None:
            hydra_config = self._create_default_hydra_config()
        super().__init__(config=DEFAULT_CONFIG, overrides=hydra_config)
        
        # Initialize tracker and renderer
        self.tracker = ObjectTracker(self.tracking_config)
        self.renderer = DetectionRenderer(self.visualization_config)
        
    def _create_default_hydra_config(self) -> Dict[str, Any]:
        """Create default Hydra configuration from inference config."""
        # Ensure imgsz is a list/tuple for compatibility
        imgsz = self.inference_config.image_size
        if isinstance(imgsz, int):
            imgsz = [imgsz, imgsz]
        
        return {
            'model': self.inference_config.model_path,
            'conf': self.inference_config.confidence_threshold,
            'iou': self.inference_config.iou_threshold,
            'max_det': self.inference_config.max_detections,
            'imgsz': imgsz,
            'device': self.inference_config.device,
            'half': self.inference_config.half_precision,
        }
    
    def get_annotator(self, img: np.ndarray) -> Annotator:
        """Get annotator for the image (required by BasePredictor)."""
        return Annotator(
            img, 
            line_width=self.visualization_config.line_thickness, 
            example=str(self.model.names)
        )
    
    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            img: Input image as numpy array
            
        Returns:
            Preprocessed tensor
        """
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255.0  # Normalize to 0-1
        return img
    
    def postprocess(
        self, 
        preds: torch.Tensor, 
        img: torch.Tensor, 
        orig_img: np.ndarray
    ) -> torch.Tensor:
        """
        Post-process predictions with NMS.
        
        Args:
            preds: Raw predictions from model
            img: Preprocessed image tensor
            orig_img: Original image
            
        Returns:
            Post-processed predictions
        """
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det
        )
        
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        
        return preds
    
    def write_results(self, idx: int, preds: torch.Tensor, batch: tuple) -> str:
        """
        Process predictions, run tracking, and draw visualizations.
        
        Args:
            idx: Batch index
            preds: Predictions from postprocess
            batch: Batch data (path, preprocessed_img, original_img)
            
        Returns:
            Log string with detection info
        """
        frame_start_time = time.time()
        
        p, im, im0 = batch
        log_string = ""
        
        if len(im.shape) == 3:
            im = im[None]  # Expand for batch dim
        
        self.seen += 1
        im0 = im0.copy()
        
        if self.webcam:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        
        self.data_path = p
        save_path = str(self.save_dir / p.name)
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + (
            '' if self.dataset.mode == 'image' else f'_{frame}'
        )
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)
        
        det = preds[idx]
        
        if len(det) == 0:
            # No detections, still draw UI
            annotated_frame = self.renderer.draw_detections(im0, np.array([]), np.array([]), np.array([]), self.model.names)
            # Store the final annotated frame for external access
            self.annotated_frame = annotated_frame
            self._update_performance_metrics(frame_start_time)
            return log_string
        
        # Log detections
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        
        # Prepare data for tracker
        xywh_bboxs = []
        confs = []
        oids = []
        
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_bboxs.append([x_c, y_c, bbox_w, bbox_h])
            confs.append([conf.item()])
            oids.append(int(cls))
        
        # Convert to tensors
        if len(xywh_bboxs) > 0:
            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)
        else:
            xywhs = torch.Tensor([])
            confss = torch.Tensor([])
        
        # Update tracker
        outputs = self.tracker.update(xywhs, confss, oids, im0)
        
        # Draw results
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            annotated_frame = self.renderer.draw_detections(im0, bbox_xyxy, identities, object_id, self.model.names)
        else:
            annotated_frame = self.renderer.draw_detections(im0, np.array([]), np.array([]), np.array([]), self.model.names)
        
        # Store the final annotated frame for external access
        self.annotated_frame = annotated_frame
        
        # Handle keyboard input if showing window
        if self.args.show:
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Key was pressed
                self.renderer.handle_keyboard_input(key)
        
        self._update_performance_metrics(frame_start_time)
        
        return log_string
    
    def _update_performance_metrics(self, frame_start_time: float) -> None:
        """Update FPS and latency metrics."""
        frame_processing_time = time.time() - frame_start_time
        self.renderer.update_fps(frame_processing_time)
    
    def predict_video(self, source: str, show: bool = False, save: bool = True):
        """
        Run prediction on video source.
        
        Args:
            source: Video file path or stream URL
            show: Whether to display results
            save: Whether to save results
        """
        self.args.source = source
        self.args.show = show
        self.args.save = save
        return self()
    
    def reset(self) -> None:
        """Reset tracker and renderer state."""
        self.tracker.reset()
        self.renderer.reset()

