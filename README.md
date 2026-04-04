# Vision AI Smart Surveillance System 🛡️

A production-ready, **Open-Vocabulary** AI-powered surveillance engine built with Python, YOLO-World, and Streamlit. This professional-grade system allows zero-shot detection of virtually any object with real-time audio voice alerts and Telegram notifications.

![Streamlit Architecture](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![YOLO-World](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-blue?style=for-the-badge&logo=streamlit)](https://vision-ai-surveillance-m9x5lmwibh8cgwjbqcrnib.streamlit.app/)

## 🌐 Live URL
[vision-ai-surveillance.streamlit.app](https://vision-ai-surveillance-m9x5lmwibh8cgwjbqcrnib.streamlit.app/)

## 🚀 Key Features

*   **🌐 Open-Vocabulary Zero-Shot Detection**: Powered by `yolov8m-worldv2`, the system acts beyond the standard 80 COCO objects. It can be easily configured to detect **any specific item** (e.g., cigarette, wallet, laptop, specific tools) by utilizing built-in semantic text embedding.
*   **🗣️ Real-time TTS Voice Alerts**: Employs an intelligent anti-spam dual-layer cache mechanism to audibly announce detected threats ("Object detected, it is a person") via native OS TTS, strictly limiting overlapping noise.
*   **📍 ROI & Intrusion Zones**: Define custom polygonal Region of Interest (ROI) boundaries directly via tracking logic to secure specific corners or frames.
*   **🕵️ Loitering Detection**: Monitors the precise timestamps of individually tracked objects (using ByteTrack/YOLO Tracking) to raise alarms when subjects over-stay.
*   **📱 Dynamic Telegram Alarms**: Sends real-time intrusion and loitering snapshots mapped to JSON confidence graphs via Telegram bot integration.
*   **✨ Premium Glassmorphism UI**: High-end Streamlit dashboard featuring immersive dark-mode aesthetics, responsive layouts, and gradient accents.

## 🛠️ Architecture Setup

- **AI Inference:** Ultralytics YOLOv8 / YOLO-World
- **Vision Parsing:** OpenCV
- **Web Dashboard:** Streamlit
- **Alert Dispatch Mechanism:** Telegram API

## 📦 Installation

**1. Clone the Repository:**
```bash
git clone https://github.com/ShivanshPandey2005/vision-ai-surveillance
cd vision-ai-surveillance
```

**2. Install Core Dependencies:**
```bash
pip install -r requirements.txt
```

**3. Configure Environment Variables (`config/config.yaml`):**
Set up your Telegram Bot API Token & Chat ID, and default features.
```yaml
alerts:
  telegram:
    enabled: true
    token: "YOUR_TELEGRAM_BOT_TOKEN"
    chat_id: "YOUR_CHAT_ID"
```

## 🖥️ Running the Application

Launch the premium Streamlit dashboard with:
```bash
streamlit run app.py
```
*Note: The first execution will download the `yolov8m-worldv2.pt` weights and embed the dynamic vocabulary pipeline (takes ~30-60 secs base setup).*

## 🔐 Configuration Highlights
You can customize the objects strictly processed by altering the hard-coded baseline in `app.py`. The architecture intelligently ignores out-of-context subjects drastically reducing hallucinations.
