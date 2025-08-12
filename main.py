# main.py

import os
import json
import traceback
import asyncio
import re
import logging
from datetime import datetime, timedelta

from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# --- Database setup ---
from repos.database import async_engine
from repos.models import Base

from modules.digest.digest_service import run_daily_digest

# Set up clean logging format
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.INFO)
# Create logger for this module
logger = logging.getLogger(__name__)

# Log startup
logger.info("=" * 60)
logger.info("SMS Lead Qualification Bot Starting")
logger.info(
    f"Environment: {'Production' if os.getenv('DATABASE_URL') else 'Development'}"
)
logger.info(f"Test Mode: {os.getenv('TEST_MODE', 'False')}")
logger.info(f"Test Phone: {os.getenv('DIGEST_TEST_PHONE', 'Not set')}")
logger.info("=" * 60)

app = FastAPI()

# Include all API routers
from modules.digest.api import router as digest_router
from api.webhooks import router as webhooks_router
from api.contractors import router as contractors_router
from api.health import router as health_router

app.include_router(health_router)
app.include_router(webhooks_router)
app.include_router(contractors_router)
app.include_router(digest_router)


async def extract_qualification_data(history_string: str) -> dict:
    """
    Extract qualification data from conversation history using OpenAI.
    """
    system_prompt = (
        "Extract exactly these fields from the customer's messages: job_type, property_type, urgency, address, access, notes.\n"
        "- If NOT mentioned by the customer, set value to empty string.\n"
        "- Do NOT guess or infer information.\n"
        "- Put all other customer comments into 'notes'.\n"
        "Respond ONLY with a JSON object with exactly these six keys.")

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": f"Conversation:\n{history_string}"
        }],
        temperature=0,
        max_tokens=300,
    )

    try:
        data = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        print("❗️ JSON parse error:", resp.choices[0].message.content)
        data = {}

    # Ensure all required keys exist
    for key in REQUIRED_FIELDS + ["notes"]:
        data.setdefault(key, "")

    return data


async def apply_correction_data(current: dict, correction: str) -> dict:
    """
    Apply user corrections to existing qualification data using OpenAI.
    """
    system_prompt = (
        "You are a JSON assistant. Given existing job data and a user correction, "
        "return the updated JSON with the same six keys only.")

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": system_prompt
        }, {
            "role":
            "user",
            "content":
            f"Existing data: {json.dumps(current)}\nCorrection: {correction}\nRespond ONLY with the full updated JSON."
        }],
        temperature=0,
        max_tokens=300,
    )

    try:
        updated = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        print("❗️ Correction JSON parse error:",
              resp.choices[0].message.content)
        updated = current

    # Ensure all required keys exist
    for key in REQUIRED_FIELDS + ["notes"]:
        updated.setdefault(key, current.get(key, ""))

    return updated


async def classify_message(message_text: str, history_string: str) -> str:
    """
    Classify incoming message into conversation states.
    """
    # Placeholder for actual message classification logic
    # This would involve using an LLM or rule-based system
    # to determine the intent of the message (e.g., provide info, confirm, correct)
    print(f" Classifying message: '{message_text}'")
    return "QUALIFYING"  # Default for now


def is_affirmative(text: str) -> bool:
    """Check if text indicates affirmative response."""
    return bool(
        re.match(r"^(yes|yep|yeah|correct|that is correct)\b",
                 text.strip().lower()))


def is_negative(text: str) -> bool:
    """Check if text indicates negative/completion response."""
    return bool(
        re.match(r"^(no|nope|all done|that'?s (all|everything))\b",
                 text.strip().lower()))


def is_qualified(data: dict) -> bool:
    """Check if all required fields are filled."""
    return all(data.get(k) for k in REQUIRED_FIELDS)


# === FastAPI setup ===
@app.on_event("startup")
async def on_startup():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(run_daily_digest, 'cron', minute=0)
    scheduler.start()
    logging.info("Scheduled daily digest job every hour on the hour")








# ===== Takeover Command Handler =====
# Add this to your SMS webhook handler


# === Takeover Command Handling (moved to modules/messaging/webhook_handler.py) ===





# In your SMS webhook, add this check for contractor messages:
"""
# After identifying contractor...
if contractor:
    # Check for takeover command
    takeover_response = await handle_takeover_command(Body, contractor, session)
    if takeover_response:
        send_message(from_phone_clean, takeover_response, is_whatsapp)
        return Response(status_code=204)

    # Continue with existing "reach out to" logic...
"""


# @app.get("/pdf/{convo_id}")
# def generate_pdf(convo_id: str):
#     """Generate PDF report for a conversation."""
#     path = f"/tmp/{convo_id}.pdf"
#     return FileResponse(path, media_type="application/pdf")


# === Daily Digest System (COMMENTED OUT - NOT IMPLEMENTED YET) ===
# async def run_daily_digest():
#     """Generate and send daily digest of leads to contractors."""
#     pass

# === Background Scheduler Setup (COMMENTED OUT - NOT IMPLEMENTED YET) ===
# scheduler = BackgroundScheduler()
# scheduler.add_job(
#     lambda: asyncio.create_task(run_daily_digest()),
#     'cron',
#     hour=18,  # 6 PM daily
#     minute=0
# )
# scheduler.start()

print("🚀 SMS Lead Qualification Bot started successfully!")
print("📱 Supporting both SMS and WhatsApp channels with COMPLETE separation")
print("🔧 Debug mode enabled with enhanced logging")
# print("⏰ Daily digest scheduled for 6 PM")  # Commented out