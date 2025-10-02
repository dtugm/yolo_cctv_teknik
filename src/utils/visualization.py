"""Visualization utilities for drawing boxes, labels, and UI elements."""

import cv2
import numpy as np
from numpy import random
from typing import Tuple, Optional


# Color palette for labels
PALETTE = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label: int) -> Tuple[int, int, int]:
    """
    Generate a fixed color for a specific class label.
    
    Args:
        label: Class label ID
        
    Returns:
        RGB color tuple
    """
    color_map = {
        0: (85, 45, 255),    # Person
        1: (255, 0, 0),       # Bicycle
        2: (222, 82, 175),    # Car
        3: (0, 204, 255),     # Motorcycle
        4: (255, 255, 0),     # Airplane
        5: (0, 149, 255),     # Bus
        6: (0, 255, 0),       # Train
        7: (255, 0, 255),     # Truck
    }
    
    if label in color_map:
        return color_map[label]
    else:
        label = int(label) % 255
        color = [int((p * (label * 2 + 1)) % 255) for p in PALETTE]
        return tuple(color)


def draw_border(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], 
                color: Tuple[int, int, int], thickness: int, r: int, d: int) -> np.ndarray:
    """
    Draw a rounded border rectangle.
    
    Args:
        img: Input image
        pt1: Top-left corner
        pt2: Bottom-right corner
        color: Border color
        thickness: Line thickness
        r: Corner radius
        d: Corner line length
        
    Returns:
        Image with drawn border
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)
    
    return img


def UI_box(x: list, img: np.ndarray, color: Optional[Tuple[int, int, int]] = None, 
           label: Optional[str] = None, line_thickness: Optional[int] = None) -> None:
    """
    Draw a bounding box with optional label on the image.
    
    Args:
        x: Bounding box coordinates [x1, y1, x2, y2]
        img: Input image
        color: Box color (auto-generated if None)
        label: Text label to display
        line_thickness: Line thickness (auto-calculated if None)
    """
    tl = line_thickness or round(0.003 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    if label:
        tf = max(tl - 1, 1)
        font_scale = tl / 2.5
        t_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=tf)[0]
        
        img = draw_border(
            img, 
            (c1[0], c1[1] - t_size[1] - 3), 
            (c1[0] + t_size[0], c1[1] + 3), 
            color, 1, 8, 2
        )
        
        cv2.putText(
            img, label, (c1[0], c1[1] - 2), 0, font_scale, 
            [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA
        )

