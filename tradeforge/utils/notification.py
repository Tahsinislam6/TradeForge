import requests
import os
from dotenv import load_dotenv

load_dotenv()

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

def send_notification(message):
    """Sends a message via Telegram Bot API with Markdown formatting."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"  # Enable Markdown formatting
    }
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        print(f"\n[INFO] Notification sent successfully: {response.json().get('result', {}).get('text')}")
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Failed to send notification: {e}")
