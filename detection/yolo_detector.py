import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.2):
        """
        Initialize the YOLOv8 detector.
        :param model_path: Path to the .pt model weights.
        :param confidence: Confidence threshold for detection.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame, classes=None):
        """
        Detect objects in a frame.
        :param frame: The input frame from video source.
        :param classes: List of integer class IDs to detect.
        :return: Ultralytics Results object.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=classes,
            verbose=False,
            device="0" if self.model.device == "cuda" else "cpu"
        )
        return results[0] # Return result for the single frame

    def track(self, frame, classes=None, tracker="bytetrack.yaml"):
        """
        Detect and track objects across frames.
        :param frame: The input frame.
        :param tracker: High-level tracker name or path to config.
        :return: Results object with tracking IDs.
        """
        results = self.model.track(
            source=frame,
            conf=self.confidence,
            classes=classes,
            persist=True,
            tracker=tracker,
            verbose=False
        )
        return results[0]
