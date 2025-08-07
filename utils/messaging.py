# utils/messaging.py

import traceback
from twilio.rest import Client
import os

# Initialize Twilio client once here
_twilio_client = Client(
    os.environ.get("TWILIO_ACCOUNT_SID", ""),
    os.environ.get("TWILIO_AUTH_TOKEN", ""),
)
TWILIO_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
WA_SANDBOX_NUMBER = os.environ.get("WHATSAPP_SANDBOX_NUMBER", "")


def send_message(to_number: str, message: str, is_whatsapp: bool = False):
    """
    UNIFIED: Send message via SMS or WhatsApp using same logic.
    """
    if is_whatsapp:
        tw_to = f"whatsapp:{to_number}"
        from_number = WA_SANDBOX_NUMBER
        channel = "WhatsApp"
    else:
        tw_to = to_number
        from_number = TWILIO_NUMBER
        channel = "SMS"

    try:
        msg = _twilio_client.messages.create(body=message,
                                             from_=from_number,
                                             to=tw_to)
        print(
            f"📤 Sent {channel} to {to_number} | SID: {msg.sid} | Status: {msg.status}"
        )
    except Exception:
        print(f"❌ Failed to send {channel} to {to_number}")
        traceback.print_exc()
