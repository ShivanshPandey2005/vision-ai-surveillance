import cv2
import yaml
from detection.yolo_detector import YOLODetector
from utils.roi import is_point_in_polygon
from utils.logger import EventLogger
from alerts.telegram_alert import TelegramAlerter
import numpy as np

def test_imports():
    print("Testing imports...")
    try:
        # Load config
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        print("✅ Config loaded.")

        # Init detector
        detector = YOLODetector(model_path="yolov8n.pt", confidence=0.5)
        print("✅ YOLOv8 Detector initialized.")

        # Test ROI logic
        point = (200, 200)
        roi = [[100, 100], [300, 100], [300, 300], [100, 300]]
        assert is_point_in_polygon(point, roi) == True
        print("✅ ROI Logic verified.")

        # Test Logger
        logger = EventLogger(log_path="outputs/test_events.json", snapshots_dir="outputs/test_snapshots/")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        logger.log_event(dummy_frame, "Test_Event", 1, 0.99)
        print("✅ Logger verified.")

        print("\nAll core modules are ready for production! 🚀")
        print("Run 'streamlit run app.py' to start the dashboard.")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    test_imports()
