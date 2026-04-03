import streamlit as st
import cv2
import yaml
import numpy as np
import time
from datetime import datetime
import subprocess
from detection.yolo_detector import YOLODetector
from utils.roi import is_point_in_polygon, draw_roi
from utils.logger import EventLogger
from alerts.telegram_alert import TelegramAlerter

# Load Config
def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

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
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, config['model']['confidence'])
st.sidebar.info("Detected classes: 1200+ Common Everyday Objects" if "world" in model_path else "Detected classes: All (80 COCO objects)")

# Feature Toggles
st.sidebar.header("🔍 Features")
enable_roi = st.sidebar.checkbox("Intrusion Detection (ROI)", config['features']['roi']['enabled'])
enable_loitering = st.sidebar.checkbox("Loitering Detection", config['features']['loitering']['enabled'])
loitering_limit = st.sidebar.number_input("Loitering (sec)", 1, 60, config['features']['loitering']['threshold_seconds'])

# Alert Settings
st.sidebar.header("🔔 Alerts")
enable_alerts = st.sidebar.toggle("Enable Telegram Alerts", config['alerts']['telegram']['enabled'])

# Initialize Components
@st.cache_resource
def init_detector(path, conf):
    det = YOLODetector(model_path=path, confidence=conf)
    if "world" in path:
        # Instead of 1200 words (which freezes CPU for 5 minutes), load the top most essential terms.
        custom_classes = [
            "person", "cigarette", "cell phone", "fan", "almirah", "chair", "table", "desk", 
            "laptop", "bottle", "cup", "backpack", "handbag", "book", "clock", "television", 
            "remote", "keyboard", "mouse", "car", "motorcycle", "bicycle", "dog", "cat", 
            "door", "window", "bed", "sofa", "refrigerator", "microwave", "sink", 
            "toothbrush", "hair dryer", "knife", "spoon", "fork", "bowl", "potted plant", 
            "umbrella", "shoes", "glasses", "watch", "pen", "tablet", "wallet", "keys"
        ]
        # Remove duplicates
        selected_classes = list(set([c for c in custom_classes if c]))
        det.model.set_classes(selected_classes)
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
    placeholder = st.empty()
    
    col_start, col_stop = st.columns(2)
    start_button = col_start.button("Start Surveillance")
    stop_button = col_stop.button("Stop Surveillance")
    
    if start_button:
        st.session_state["run_surveillance"] = True
    if stop_button:
        st.session_state["run_surveillance"] = False

with col2:
    st.subheader("📜 Recent Events")
    event_log_placeholder = st.empty()

# Main Loop
run = st.session_state.get("run_surveillance", False)

if run:
    # Source (0 for webcam)
    video_source = 0
    cap = cv2.VideoCapture(video_source)

    # Default ROI from config
    roi_points = config['features']['roi']['points']
    
    try:
        while cap.isOpened() and st.session_state.get("run_surveillance", False):
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture Video Stream.")
                break
                
            frame_display = frame.copy()
            
            # Draw ROI if enabled
            if enable_roi:
                draw_roi(frame_display, roi_points)
                
            # Detection & Tracking
            results = detector.track(frame, classes=config['model']['classes'])
            
            current_time = datetime.now()
            frame_events = []
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.int().cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.int().cpu().numpy()
                
                for box, track_id, conf, cls in zip(boxes, ids, confs, classes):
                    x1, y1, x2, y2 = map(int, box)
                    center = (int((x1+x2)/2), int((y1+y2)/2))
                    cls_name = detector.model.names[cls]
                    
                    # Draw bounding box
                    label = f"ID:{track_id} {cls_name}"
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_display, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Voice Alert with dual-layer anti-spam logic
                    current_time_sec = time.time()
                    if not st.session_state.last_spoken.get(track_id, False):
                        if current_time_sec - st.session_state.last_spoken_cls.get(cls_name, 0) > 15:
                            st.session_state.last_spoken[track_id] = True
                            st.session_state.last_spoken_cls[cls_name] = current_time_sec
                            # Run powershell TTS in background so it doesn't block video feed
                            tts_cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('Object detected, it is a {cls_name}')"
                            subprocess.Popen(["powershell", "-Command", tts_cmd], creationflags=subprocess.CREATE_NO_WINDOW)
                    
                    # Intrusion Logic
                    if enable_roi and is_point_in_polygon(center, roi_points):
                        event = logger.log_event(frame, "Intrusion", track_id, conf)
                        alerter.send_alert("Intrusion", track_id, conf, event['snapshot'])
                        frame_events.append(event)
                        
                    # Loitering Logic
                    if enable_loitering:
                        if track_id not in st.session_state.tracker_history:
                            st.session_state.tracker_history[track_id] = current_time
                        else:
                            duration = (current_time - st.session_state.tracker_history[track_id]).total_seconds()
                            if duration > loitering_limit:
                                event = logger.log_event(frame, "Loitering", track_id, conf)
                                alerter.send_alert("Loitering", track_id, conf, event['snapshot'])
                                frame_events.append(event)
            
            # Update UI
            placeholder.image(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Update Event List
            recent_logs = logger.get_recent_logs(8)
            if recent_logs:
                event_log_placeholder.table(recent_logs[::-1])

    finally:
        cap.release()
else:
    st.write("Surveillance Stopped.")
