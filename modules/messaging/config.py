
# modules/messaging/config.py

import os

# Twilio configuration
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
WA_SANDBOX_NUMBER = os.environ.get("WHATSAPP_SANDBOX_NUMBER", "")

# Test configuration
TEST_MODE = os.getenv("TEST_MODE", "False").lower() == "true"
DIGEST_TEST_PHONE = os.getenv("DIGEST_TEST_PHONE")
