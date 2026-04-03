import cv2
import numpy as np

def is_point_in_polygon(point, polygon):
    """
    Check if a point (x, y) is within a polygon.
    :param point: Tuple (x, y)
    :param polygon: List of tuples/lists [(x1, y1), (x2, y2), ...]
    :return: Boolean
    """
    result = cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False)
    return result >= 0

def draw_roi(frame, polygon, color=(0, 255, 0), thickness=2):
    """
    Draw a translucent ROI on the frame.
    :param frame: The image frame.
    :param polygon: List of points.
    :param color: BGR color.
    """
    pts = np.array(polygon, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, thickness)
    
    # Overlay translucent filling
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    return frame
