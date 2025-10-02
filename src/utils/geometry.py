"""Geometric utility functions for bounding boxes and tracking."""

import math
from typing import List, Tuple


def xyxy_to_xywh(*xyxy) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from (x1, y1, x2, y2) format to (x_center, y_center, width, height).
    
    Args:
        *xyxy: Four coordinates (x1, y1, x2, y2)
        
    Returns:
        Tuple of (x_center, y_center, width, height)
    """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy: List) -> List[List[int]]:
    """
    Convert bounding boxes from xyxy to top-left-width-height format.
    
    Args:
        bbox_xyxy: List of bounding boxes in xyxy format
        
    Returns:
        List of bounding boxes in tlwh format
    """
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def ccw(A: Tuple, B: Tuple, C: Tuple) -> bool:
    """Check if three points are in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A: Tuple, B: Tuple, C: Tuple, D: Tuple) -> bool:
    """
    Check if line segment AB intersects with line segment CD.
    
    Args:
        A, B: Points defining first line segment
        C, D: Points defining second line segment
        
    Returns:
        True if segments intersect, False otherwise
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def get_direction(point1: Tuple[int, int], point2: Tuple[int, int]) -> str:
    """
    Get cardinal direction from point1 to point2.
    
    Args:
        point1: Starting point (x, y)
        point2: Ending point (x, y)
        
    Returns:
        String describing direction (e.g., "North", "SouthWest")
    """
    direction_str = ""

    # Calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"

    # Calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"

    return direction_str


def estimate_speed(location1: Tuple[int, int], location2: Tuple[int, int], 
                  pixels_per_meter: int = 15, time_constant: float = 15 * 3.6) -> int:
    """
    Estimate speed based on movement between two locations.
    
    Args:
        location1: Starting location (x, y)
        location2: Ending location (x, y)
        pixels_per_meter: Calibration factor for pixel to meter conversion
        time_constant: Time conversion constant (default converts to km/h)
        
    Returns:
        Estimated speed in km/h
    """
    # Euclidean distance in pixels
    d_pixel = math.sqrt(
        math.pow(location2[0] - location1[0], 2) + 
        math.pow(location2[1] - location1[1], 2)
    )
    
    # Convert to meters
    d_meters = d_pixel / pixels_per_meter
    
    # Calculate speed
    speed = d_meters * time_constant
    
    return int(speed)

