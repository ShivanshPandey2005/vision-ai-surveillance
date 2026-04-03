import os
import json
import sqlite3
import cv2
from datetime import datetime

class EventLogger:
    def __init__(self, log_path="outputs/events.json", snapshots_dir="outputs/snapshots/"):
        """
        Initialize the logger.
        :param log_path: Path for event JSON file.
        :param snapshots_dir: Path for snapshot images.
        """
        self.log_path = log_path
        self.snapshots_dir = snapshots_dir
        
        # Ensure directories exist
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Load existing logs if they exist
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                try:
                    self.logs = json.load(f)
                except json.JSONDecodeError:
                    self.logs = []
        else:
            self.logs = []

    def log_event(self, frame, event_type, object_id=None, confidence=0.0):
        """
        Log an event and save a snapshot.
        :param frame: The image frame where event occurred.
        :param event_type: Description of the event (Intrusion, Loitering, etc.).
        :param object_id: ID of the tracked object.
        :param confidence: Detection confidence.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{event_type.lower()}_{object_id}.jpg"
        snapshot_path = os.path.join(self.snapshots_dir, file_name)
        
        # Save snapshot
        cv2.imwrite(snapshot_path, frame)
        
        # Add to logs
        event = {
            "timestamp": timestamp,
            "event_type": event_type,
            "object_id": str(object_id) if object_id is not None else None,
            "confidence": f"{float(confidence):.2f}",
            "snapshot": snapshot_path
        }
        self.logs.append(event)
        
        # Save to JSON
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, indent=4)
        
        return event

    def get_recent_logs(self, limit=10):
        """
        Return the most recent N logs.
        """
        return self.logs[-limit:]
