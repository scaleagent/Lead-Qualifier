# main.py

import os
import json
import traceback
import asyncio
import re
import logging
from datetime import datetime, timedelta

from fastapi import FastAPI, Depends, Form, Response
from fastapi.responses import PlainTextResponse, FileResponse, Response
from sqlalchemy.future import select

from openai import OpenAI
from twilio.rest import Client
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# --- Database & Repos imports ---
from repos.database import async_engine, AsyncSessionLocal
from repos.models import Base, Contractor, ConversationData
from repos.contractor_repo import ContractorRepo
from repos.conversation_repo import ConversationRepo
from repos.message_repo import MessageRepo
from repos.conversation_data_repo import ConversationDataRepo

from modules.digest.digest_service import run_daily_digest
from modules.digest.api import router as digest_router
from utils.messaging import send_message

# --- Qualification Module Imports ---
from modules.qualification import QualificationService

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

# Include digest module router
app.include_router(digest_router)

# === External API Clients ===
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
twilio_client = Client(
    os.environ.get("TWILIO_ACCOUNT_SID", ""),
    os.environ.get("TWILIO_AUTH_TOKEN", ""),
)
TWILIO_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
WA_SANDBOX_NUMBER = os.environ.get("WHATSAPP_SANDBOX_NUMBER", "")

# === Phone Number Formatting for Channel Separation ===


# === Helpers ===
# def send_message(to_number: str, message: str, is_whatsapp: bool = False):
#     """
#     UNIFIED: Send message via SMS or WhatsApp using same logic.
#     """
#     # Format recipient based on channel
#     if is_whatsapp:
#         tw_to = f"whatsapp:{to_number}"
#         from_number = WA_SANDBOX_NUMBER
#         channel = "WhatsApp"
#     else:
#         tw_to = to_number
#         from_number = TWILIO_NUMBER
#         channel = "SMS"

#     try:
#         msg = twilio_client.messages.create(body=message,
#                                             from_=from_number,
#                                             to=tw_to)
#         print(
#             f"📤 Sent {channel} to {to_number} | SID: {msg.sid} | Status: {msg.status}"
#         )
#     except Exception:
#         print(f"❌ Failed to send {channel} to {to_number}")
#         traceback.print_exc()


# === Message Classification (moved to modules/messaging/message_classifier.py) ===


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


@app.get("/", response_class=PlainTextResponse)
def read_root():
    return "✅ SMS-Lead-Qual API is running."


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


@app.get("/contractors", response_class=PlainTextResponse)
async def list_contractors(session=Depends(get_session)):
    """Debug endpoint to view stored contractor profiles"""
    contractor_repo = ContractorRepo(session)
    contractors = await contractor_repo.get_all()

    if not contractors:
        return "❌ No contractors found in database"

    result = ["📋 STORED CONTRACTORS:"]
    for c in contractors:
        result.append(f"  ID: {c.id}")
        result.append(f"  Name: {c.name}")
        result.append(f"  Phone: {c.phone}")
        result.append(f"  Address: {c.address or 'Not set'}")
        result.append(f"  Created: {c.created_at}")
        result.append("  " + "-" * 30)

    return "\n".join(result)


# ===== Takeover Command Handler =====
# Add this to your SMS webhook handler


# === Takeover Command Handling (moved to modules/messaging/webhook_handler.py) ===


@app.post("/sms", response_class=PlainTextResponse)
async def sms_webhook(From: str = Form(...),
                      To: str = Form(...),
                      Body: str = Form(...),
                      session=Depends(get_session)):
    """SMS/WhatsApp webhook handler - now using modular messaging system"""
    from modules.messaging import MessageWebhookHandler

    handler = MessageWebhookHandler(session)
    return await handler.handle_webhook(From, To, Body)


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