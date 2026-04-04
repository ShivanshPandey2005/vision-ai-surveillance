import streamlit as st
import cv2
import yaml
import numpy as np
import time
from datetime import datetime
import subprocess
import threading
import queue
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from detection.yolo_detector import YOLODetector
from utils.roi import is_point_in_polygon, draw_roi
from utils.logger import EventLogger
from alerts.telegram_alert import TelegramAlerter

# Load Config
def load_config():
    # Priority: st.secrets (Cloud) > local config.yaml
    try:
        if "alerts" in st.secrets:
            return st.secrets
    except:
        pass
    
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# UI Layout
st.set_page_config(page_title="AI Smart Surveillance", layout="wide")

# Custom Premium CSS Styling
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0b1120;
    }
    /* Title Gradient */
    h1 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 10px;
    }
    /* Glowing Buttons */
    div[data-testid="stButton"] > button {
        background: linear-gradient(90deg, #3b82f6 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Vision AI Surveillance")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-top: -20px; margin-bottom: 30px;'>Advanced Open-Vocabulary Security Engine</p>", unsafe_allow_html=True)

# Sidebar - Settings
st.sidebar.header("🔧 Settings")
model_path = st.sidebar.selectbox("YOLOv8 Model", ["yolov8m-worldv2.pt", "yolov8s-world.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.15)
st.sidebar.info("Detected classes: 50+ Optimized Security & Household Objects" if "world" in model_path else "Detected classes: All (80 COCO objects)")

# Feature Toggles
st.sidebar.header("🔍 Features")
enable_roi = st.sidebar.checkbox("Intrusion Detection (ROI)", config['features']['roi']['enabled'])
enable_loitering = st.sidebar.checkbox("Loitering Detection", config['features']['loitering']['enabled'])
loitering_limit = st.sidebar.number_input("Loitering (sec)", 1, 60, config['features']['loitering']['threshold_seconds'])

# Alert Settings
st.sidebar.header("🔔 Alerts")
enable_alerts = st.sidebar.toggle("Enable Telegram Alerts", config['alerts']['telegram']['enabled'])

# AI Video Processor for WebRTC
class VideoTransformer(VideoTransformerBase):
    def __init__(self, detector, logger, alerter, config, roi_points):
        self.detector = detector
        self.logger = logger
        self.alerter = alerter
        self.config = config
        self.roi_points = roi_points
        self.tracker_history = {}
        self.last_spoken = {}
        self.last_spoken_cls = {}
        self.result_queue = queue.Queue()
        self.frame_count = 0
        self.last_results = None

    def transform(self, frame):
        img_raw = frame.to_ndarray(format="bgr24")
        frame_display = img_raw.copy()
        self.frame_count += 1
        
        # AI Activity Indicator
        cv2.putText(frame_display, "AI ENGINE: ACTIVE (OPTIMIZED)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        # Process every 5th frame to save CPU
        if self.frame_count % 5 == 0 or self.last_results is None:
            # Downscale for AI speed (Inference only)
            img_ai = cv2.resize(img_raw, (320, 320))
            
            # Detection (Use predict for 2x speed over track)
            target_classes = None if "world" in self.config['model']['path'] else self.config['model']['classes']
            self.last_results = self.detector.detect(img_ai, classes=target_classes)
            
        # Draw and handle using last_results
        if self.last_results is not None and self.last_results.boxes is not None:
            boxes = self.last_results.boxes.xyxy.cpu().numpy()
            confs = self.last_results.boxes.conf.cpu().numpy()
            classes = self.last_results.boxes.cls.int().cpu().numpy()
            
            # Rescale boxes back to original size
            h, w = img_raw.shape[:2]
            scale_x, scale_y = w / 320, h / 320
            
            # Check if IDs exist (tracking), otherwise use -1 (detection only)
            ids = self.last_results.boxes.id.int().cpu().numpy() if self.last_results.boxes.id is not None else [-1] * len(boxes)
            
            for box, track_id, conf, cls in zip(boxes, ids, confs, classes):
                # Scale coordinates
                x1, y1, x2, y2 = int(box[0]*scale_x), int(box[1]*scale_y), int(box[2]*scale_x), int(box[3]*scale_y)
                center = (int((x1+x2)/2), int((y1+y2)/2))
                cls_name = self.detector.model.names[cls]
                
                # Draw bounding box (Red for tracked, Yellow for detection-only)
                color = (0, 0, 255) if track_id != -1 else (0, 255, 255)
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{track_id} {cls_name}" if track_id != -1 else f"{cls_name} {conf:.2f}"
                cv2.putText(frame_display, label, (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Voice Alert logic (passed to UI via queue) - Only for tracked OR high confidence detection
                if track_id != -1 or conf > 0.4:
                    current_time_sec = time.time()
                    if not self.last_spoken.get(track_id if track_id != -1 else cls_name, False):
                        if current_time_sec - self.last_spoken_cls.get(cls_name, 0) > 15:
                            self.last_spoken[track_id if track_id != -1 else cls_name] = True
                            self.last_spoken_cls[cls_name] = current_time_sec
                            self.result_queue.put({"type": "speech", "text": cls_name})
                    
                # Intrusion/Loitering logic
                if is_point_in_polygon(center, self.roi_points):
                    event = self.logger.log_event(img_raw, "Intrusion", track_id, conf)
                    self.alerter.send_alert("Intrusion", track_id, conf, event['snapshot'])
        
        return frame_display

@st.cache_resource
def init_detector(path, conf):
    det = YOLODetector(model_path=path, confidence=conf)
    if "world" in path:
        custom_classes = [
            "person", "cigarette", "cell phone", "fan", "almirah", "chair", "table", "desk", 
            "laptop", "bottle", "cup", "backpack", "handbag", "book", "clock", "television", 
            "remote", "keyboard", "mouse", "car", "motorcycle", "bicycle", "dog", "cat", 
            "door", "window", "bed", "sofa", "refrigerator", "microwave", "sink", 
            "toothbrush", "hair dryer", "knife", "spoon", "fork", "bowl", "potted plant", 
            "umbrella", "shoes", "glasses", "watch", "pen", "tablet", "wallet", "keys"
        ]
        det.model.set_classes(list(set(custom_classes)))
    return det

with st.spinner("Loading AI Engine & Vocabulary (Takes ~30 sec first time)..."):
    detector = init_detector(model_path, conf_threshold)

logger = EventLogger(config['storage']['logs_path'], config['storage']['snapshots_dir'])
alerter = TelegramAlerter(
    config['alerts']['telegram']['token'], 
    config['alerts']['telegram']['chat_id'], 
    enabled=enable_alerts
)

# State Management
if 'tracker_history' not in st.session_state:
    st.session_state.tracker_history = {} # {track_id: start_time}
if 'last_spoken' not in st.session_state:
    st.session_state.last_spoken = {} # {track_id: bool}
if 'last_spoken_cls' not in st.session_state:
    st.session_state.last_spoken_cls = {} # {cls_name: timestamp}

# App Columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎥 Live Feed")
    
    # Browser-based TTS (JavaScript)
    speech_placeholder = st.empty()
    
    webrtc_ctx = webrtc_streamer(
        key="surveillance",
        video_transformer_factory=lambda: VideoTransformer(
            detector, logger, alerter, config, config['features']['roi']['points']
        ),
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"width": 320, "height": 240}, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_transformer:
        # Handle results from the transformer thread
        try:
            while True:
                result = webrtc_ctx.video_transformer.result_queue.get_nowait()
                if result["type"] == "speech":
                    # Inject JS to speak in browser
                    cls_name = result["text"]
                    speech_placeholder.markdown(f"""
                        <script>
                            var msg = new SpeechSynthesisUtterance('Object detected, it is a {cls_name}');
                            window.speechSynthesis.speak(msg);
                        </script>
                    """, unsafe_allow_html=True)
        except queue.Empty:
            pass

with col2:
    st.subheader("📜 Recent Events")
    event_log_placeholder = st.empty()
    
    # Show recent logs
    recent_logs = logger.get_recent_logs(8)
    if recent_logs:
        event_log_placeholder.table(recent_logs[::-1])
