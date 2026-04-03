import requests
import os
from datetime import datetime

class TelegramAlerter:
    def __init__(self, token, chat_id, enabled=False):
        """
        Initialize Telegram Alert System.
        :param token: Bot API token.
        :param chat_id: Target chat ID.
        :param enabled: Is alert system active.
        """
        self.bot_url = f"https://api.telegram.org/bot{token}/"
        self.chat_id = chat_id
        self.enabled = enabled
        self.cooldowns = {} # Stores timestamps of last alerts by event_type

    def send_alert(self, event_type, object_id, confidence, snapshot_path=None, cooldown=30):
        """
        Send a message and photo to Telegram.
        :param event_type: Description of the alert.
        :param object_id: Tracked object ID.
        :param confidence: Confidence score.
        :param snapshot_path: File path of the image.
        :param cooldown: Threshold in seconds to avoid spamming the same event.
        """
        if not self.enabled:
            return False
            
        # Check cooldown
        if event_type in self.cooldowns:
            if (datetime.now() - self.cooldowns[event_type]).total_seconds() < cooldown:
                return False
        
        caption = (
            f"🚨 *AI Surveillance Alert* 🚨\n\n"
            f"📅 *Time*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"🔥 *Event*: {event_type}\n"
            f"👤 *ID*: {object_id}\n"
            f"📊 *Conf*: {float(confidence)*100:.1f}%"
        )
        
        try:
            # Send photo first
            if snapshot_path and os.path.exists(snapshot_path):
                with open(snapshot_path, 'rb') as photo:
                    files = {'photo': photo}
                    data = {'chat_id': self.chat_id, 'caption': caption, 'parse_mode': 'Markdown'}
                    response = requests.post(self.bot_url + "sendPhoto", data=data, files=files)
            else:
                # Send text only if no photo
                data = {'chat_id': self.chat_id, 'text': caption, 'parse_mode': 'Markdown'}
                response = requests.post(self.bot_url + "sendMessage", data=data)
            
            if response.status_code == 200:
                self.cooldowns[event_type] = datetime.now()
                return True
        except Exception as e:
            print(f"Failed to send Telegram alert: {e}")
        
        return False
