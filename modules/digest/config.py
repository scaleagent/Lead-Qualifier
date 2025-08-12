
# modules/digest/config.py - Configuration constants for digest functionality

import os

# PDF Security
PDF_SECRET_KEY = os.environ.get("PDF_SECRET_KEY", "change-this-in-production")
PDF_TOKEN_EXPIRY_HOURS = 24

# Digest Settings
DEFAULT_DIGEST_HOUR = 18
DEFAULT_TIMEZONE = 'Europe/London'
DEFAULT_REPEAT_UNTIL_TAKEOVER = True

# URLs
APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://your-app.herokuapp.com")

# Test overrides
DIGEST_TEST_PHONE = os.getenv("DIGEST_TEST_PHONE")
TEST_MODE = os.getenv("TEST_MODE", "False").lower() == "true"
