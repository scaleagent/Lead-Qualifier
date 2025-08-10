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

from services.digest import run_daily_digest

from utils.messaging import send_message

## imports for pdf generation
import hashlib
import hmac
from urllib.parse import quote

# Set up clean logging format
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.INFO)

app = FastAPI()

# === External API Clients ===
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
twilio_client = Client(
    os.environ.get("TWILIO_ACCOUNT_SID", ""),
    os.environ.get("TWILIO_AUTH_TOKEN", ""),
)
TWILIO_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
WA_SANDBOX_NUMBER = os.environ.get("WHATSAPP_SANDBOX_NUMBER", "")

# === Qualification Schema ===
REQUIRED_FIELDS = ["job_type", "property_type", "urgency", "address", "access"]


# === Phone Number Formatting for Channel Separation ===
def normalize_phone_for_db(phone: str, is_whatsapp: bool) -> str:
    """
    Normalize phone numbers for database storage with channel separation:
    - WhatsApp: wa:+447742001014
    - SMS: +447742001014
    """
    # Remove any existing whatsapp: prefix first
    clean_phone = phone.split(":",
                              1)[1] if phone.startswith("whatsapp:") else phone

    if is_whatsapp:
        return f"wa:{clean_phone}"
    else:
        return clean_phone


def get_system_number_for_db(is_whatsapp: bool) -> str:
    """Get the system's number in the correct format for database storage"""
    if is_whatsapp:
        # Remove whatsapp: prefix and add wa: prefix
        clean_number = WA_SANDBOX_NUMBER.split(
            ":", 1)[1] if WA_SANDBOX_NUMBER.startswith(
                "whatsapp:") else WA_SANDBOX_NUMBER
        return f"wa:{clean_number}"
    else:
        return TWILIO_NUMBER


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


async def classify_message(message_text: str, history_string: str) -> str:
    """
    Classify incoming message as NEW, CONTINUATION, or UNSURE using OpenAI.
    """
    # Extract the last AI message if it exists (usually contains the question)
    last_ai_message = ""
    if history_string:
        lines = history_string.strip().split('\n')
        for line in reversed(lines):
            if line.startswith("AI:"):
                last_ai_message = line.replace("AI:", "").strip()
                break

    system_prompt = (
        "You are a message classifier for a contractor's AI assistant. Your job is to determine if an incoming message is part of an existing conversation or starting a new one.\n\n"
        "Classification rules:\n"
        "1. CONTINUATION - Use when:\n"
        "   - The message answers or responds to a question from the conversation history\n"
        "   - The message provides information related to the ongoing topic\n"
        "   - The message acknowledges the previous message (e.g., 'thanks', 'yes', 'sure') AND provides relevant information\n"
        "   - The message corrects or adds to previously discussed information\n"
        "2. NEW - Use when:\n"
        "   - The message asks about a completely different job/service\n"
        "   - The message mentions a new address/location not previously discussed\n"
        "   - The message explicitly states starting over or new request\n"
        "   - The message is unrelated to any previous questions or topics\n"
        "3. UNSURE - Use when:\n"
        "   - The message is ambiguous or could go either way\n"
        "   - Simple greetings without context\n"
        "   - Very short responses that don't clearly relate to previous messages\n\n"
        "IMPORTANT: Focus on whether the message logically follows from or responds to the conversation history.\n"
        "Respond with exactly one word: NEW, CONTINUATION, or UNSURE.")

    # Structure the data more clearly
    user_prompt = f"""Analyze this message classification task:

CONVERSATION HISTORY:
{history_string if history_string else "[No previous messages]"}

LAST QUESTION/STATEMENT FROM AI:
{last_ai_message if last_ai_message else "[No previous AI message]"}

NEW INCOMING MESSAGE FROM CUSTOMER:
"{message_text}"

ANALYSIS CHECKLIST:
1. Does the new message answer or respond to the last AI question? 
2. Does the new message provide information that was requested?
3. Is the new message about the same topic/job as the conversation history?
4. Does the new message explicitly mention a different job or location?

Based on the analysis, classify as: NEW, CONTINUATION, or UNSURE"""

    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": user_prompt
        }],
        temperature=0,
        max_tokens=10,
    )

    result = resp.choices[0].message.content.strip().upper()

    # Failsafe: ensure we only return valid classifications
    if result not in ["NEW", "CONTINUATION", "UNSURE"]:
        print(
            f"⚠️ Invalid classification result: {result}, defaulting to UNSURE"
        )
        return "UNSURE"

    return result


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
        model="gpt-3.5-turbo",
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
        model="gpt-3.5-turbo",
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


async def handle_takeover_command(body: str, contractor: Contractor,
                                  session) -> str | None:
    """
    Handle contractor takeover commands like "stop daily digest for plumbing job"
    Returns response message if command was handled, None otherwise.
    """
    # Check for takeover command patterns
    patterns = [
        r"stop\s+(?:daily\s+)?digest\s+for\s+(.+)", r"takeover\s+(.+)",
        r"take\s+over\s+(.+)", r"claim\s+(.+)"
    ]

    for pattern in patterns:
        match = re.match(pattern, body.lower().strip())
        if match:
            job_query = match.group(1).strip()

            # Find matching conversation
            data_repo = ConversationDataRepo(session)
            lead = await data_repo.find_by_job_title_fuzzy(
                contractor.id, job_query)

            if lead:
                # Mark as opted out
                await data_repo.mark_digest_opt_out(lead.conversation_id)

                return (f"✅ Stopped daily digest for '{lead.job_title}'. "
                        f"You won't receive further updates about this lead.")
            else:
                return (f"❌ Couldn't find a lead matching '{job_query}'. "
                        f"Please check the job title and try again.")

    return None  # Not a takeover command


@app.post("/sms", response_class=PlainTextResponse)
async def sms_webhook(From: str = Form(...),
                      To: str = Form(...),
                      Body: str = Form(...),
                      session=Depends(get_session)):
    From, To, Body = From.strip(), To.strip(), Body.strip()
    print(f"🔔 INCOMING SMS/WhatsApp:")
    print(f"   From: {From}")
    print(f"   To: {To}")
    print(f"   Body: {Body!r}")

    # Unified channel detection and normalization
    is_whatsapp = From.startswith("whatsapp:")
    from_phone_clean = From.split(":", 1)[1] if is_whatsapp else From

    # CRITICAL: Normalize phone numbers for database with channel separation
    customer_phone_db = normalize_phone_for_db(from_phone_clean, is_whatsapp)
    system_phone_db = get_system_number_for_db(is_whatsapp)

    print(f"🔍 PARSED:")
    print(f"   is_whatsapp: {is_whatsapp}")
    print(f"   from_phone_clean: {from_phone_clean}")
    print(f"   customer_phone_db: {customer_phone_db}")
    print(f"   system_phone_db: {system_phone_db}")

    # Initialize repositories
    contractor_repo = ContractorRepo(session)
    conv_repo = ConversationRepo(session)
    msg_repo = MessageRepo(session)
    data_repo = ConversationDataRepo(session)

    # 1) Handle contractor-initiated "reach out" commands
    # NOTE: Contractors are stored with clean phone numbers (no wa: prefix)
    contractor = await contractor_repo.get_by_phone(from_phone_clean)
    if contractor:
        print(
            f"✅ CONTRACTOR IDENTIFIED: {contractor.name} (ID: {contractor.id})"
        )
        # ===== DETERMINE WHERE TO SEND RESPONSES =====
        # Use test phone if configured, otherwise send back to contractor
        response_phone = os.getenv("DIGEST_TEST_PHONE") or from_phone_clean

        if os.getenv("DIGEST_TEST_PHONE"):
            print(f"📱 Using TEST PHONE for response: {response_phone}")

        # ===== CHECK FOR TAKEOVER COMMAND FIRST =====
        takeover_response = await handle_takeover_command(
            Body, contractor, session)
        if takeover_response:
            print(f"🎯 TAKEOVER COMMAND processed")

            # Log the command
            await msg_repo.create_message(
                sender=customer_phone_db,
                receiver=system_phone_db,
                body=Body,
                direction="inbound",
                conversation_id=None  # Command, not part of a conversation
            )

            # Send response
            send_message(from_phone_clean, takeover_response, is_whatsapp)

            # Log the response
            await msg_repo.create_message(sender=system_phone_db,
                                          receiver=customer_phone_db,
                                          body=takeover_response,
                                          direction="outbound",
                                          conversation_id=None)

            return Response(status_code=204)
        # ===== CHECK FOR "REACH OUT TO" COMMAND =====
        # Match both UK and international phone numbers
        m = re.match(r'^\s*reach out to (\+\d{10,15})\s*$', Body,
                     re.IGNORECASE)
        if m:
            target_phone_clean = m.group(1)
            target_phone_db = normalize_phone_for_db(target_phone_clean,
                                                     is_whatsapp)
            print(
                f"📞 REACH OUT COMMAND to: {target_phone_clean} (DB: {target_phone_db})"
            )

            # Close any existing conversation for this customer ON THIS CHANNEL
            old_convo = await conv_repo.get_active_conversation(
                contractor.id, target_phone_db)
            if old_convo:
                print(f"🔄 CLOSING existing conversation: {old_convo.id}")
                await conv_repo.close_conversation(old_convo.id)

            # Create new conversation
            convo = await conv_repo.create_conversation(
                contractor_id=contractor.id,
                customer_phone=target_phone_db  # Store with channel prefix
            )
            print(f"🆕 CREATED new conversation: {convo.id}")

            # Send introduction message
            intro = (
                f"Hi! I'm {contractor.name}'s assistant. "
                "To get started, please tell me the type of job you need.")

            await msg_repo.create_message(
                sender=system_phone_db,  # Store with channel prefix
                receiver=target_phone_db,  # Store with channel prefix
                body=intro,
                direction="outbound",
                conversation_id=convo.id)
            send_message(target_phone_clean, intro,
                         is_whatsapp)  # Send with clean phone
            print(f"📤 SENT intro message to customer: {target_phone_clean}")

        return Response(status_code=204)

    # 2) Handle customer-initiated messages

    print(f"👤 CUSTOMER MESSAGE from: {customer_phone_db}")

    # Lookup which contractor owns the assistant number (To:)
    contractor = await contractor_repo.get_by_assistant_phone(system_phone_db)
    if not contractor:
        print(
            f"⚠️ Unrecognized assistant number: {system_phone_db}; dropping message."
        )
        return Response(status_code=204)

    print(
        f"🏢 Routing customer message to contractor: {contractor.name} (ID: {contractor.id})"
    )

    print(f"🏢 USING contractor: {contractor.name} (ID: {contractor.id})")

    # Get any active conversation for this customer ON THIS CHANNEL
    old_convo = await conv_repo.get_active_conversation(
        contractor.id, customer_phone_db)
    if old_convo:
        print(
            f"💬 FOUND active conversation: {old_convo.id} (status: {old_convo.status})"
        )
    else:
        print(
            f"❌ NO active conversation found for customer: {customer_phone_db}"
        )

    # 3) Handle CONFIRMING and COLLECTING_NOTES states with priority
    if old_convo and old_convo.status in ("CONFIRMING", "COLLECTING_NOTES"):
        print(f"🔄 HANDLING {old_convo.status} state")

        # Log the incoming message
        await msg_repo.create_message(sender=customer_phone_db,
                                      receiver=system_phone_db,
                                      body=Body,
                                      direction="inbound",
                                      conversation_id=old_convo.id)

        # CONFIRMING state: handle confirmation or corrections
        if old_convo.status == "CONFIRMING":
            if is_affirmative(Body):
                print("✅ CONFIRMED - moving to notes collection")
                # Move to collecting additional notes
                follow = (
                    "Thanks! If there's any other important info—parking, pets, special access—"
                    "just reply here. When you're done, reply 'No'.")
                await msg_repo.create_message(sender=system_phone_db,
                                              receiver=customer_phone_db,
                                              body=follow,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_message(from_phone_clean, follow, is_whatsapp)
                old_convo.status = "COLLECTING_NOTES"
                await session.commit()
            else:
                print("🔧 CORRECTIONS needed")
                # Handle corrections
                full_msgs = await msg_repo.get_all_conversation_messages(
                    old_convo.id)
                history = "\n".join(
                    f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                    for d, b in full_msgs)
                data = await extract_qualification_data(history)
                updated = await apply_correction_data(data, Body)

                await data_repo.upsert(
                    conversation_id=old_convo.id,
                    contractor_id=contractor.id,
                    customer_phone=
                    customer_phone_db,  # Store with channel prefix
                    data_dict=updated,
                    qualified=is_qualified(updated),
                    job_title=updated.get("job_type"))

                # Show updated summary
                bullets = [
                    f"• {f.replace('_',' ').title()}: {updated[f]}"
                    for f in REQUIRED_FIELDS
                ]
                summary = "Got it! Here's the updated info:\n" + "\n".join(
                    bullets) + "\nIs that correct?"

                await msg_repo.create_message(sender=system_phone_db,
                                              receiver=customer_phone_db,
                                              body=summary,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_message(from_phone_clean, summary, is_whatsapp)

            return Response(status_code=204)

        # COLLECTING_NOTES state: collect additional information
        if old_convo.status == "COLLECTING_NOTES":
            if is_negative(Body):
                print("✅ COMPLETED - closing conversation")
                # Complete the conversation
                closing = "Great—thanks! I'll pass this along to your contractor. ✅"
                await msg_repo.create_message(sender=system_phone_db,
                                              receiver=customer_phone_db,
                                              body=closing,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_message(from_phone_clean, closing, is_whatsapp)
                await conv_repo.close_conversation(old_convo.id)
            else:
                print("📝 ADDING to notes")
                # Append to notes
                cd: ConversationData = await session.get(
                    ConversationData, old_convo.id)
                if cd:
                    existing_notes = cd.data_json.get("notes", "")
                    combined_notes = (existing_notes + "\n" + Body).strip()
                    cd.data_json["notes"] = combined_notes
                    cd.last_updated = datetime.utcnow()
                    await session.commit()

                prompt = "Anything else to add? If not, reply 'No'."
                await msg_repo.create_message(sender=system_phone_db,
                                              receiver=customer_phone_db,
                                              body=prompt,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_message(from_phone_clean, prompt, is_whatsapp)

            return Response(status_code=204)

    # 4) Save the incoming message FIRST (before classification)
    # This ensures we never lose messages
    if old_convo:
        await msg_repo.create_message(sender=customer_phone_db,
                                      receiver=system_phone_db,
                                      body=Body,
                                      direction="inbound",
                                      conversation_id=old_convo.id)

    # 5) Message Classification
    print(f"🧠 STARTING message classification...")

    # Special case: If this is right after a "reach out" command, always CONTINUATION
    # Check if the last message in the conversation was the AI intro
    if old_convo:
        last_messages = await msg_repo.get_recent_messages(
            customer=customer_phone_db,
            contractor=system_phone_db,
            conversation_id=old_convo.id,  # Only from this conversation
            limit=2)

        # If only 1 message and it's the AI intro, this must be a continuation
        if len(last_messages) == 1 and last_messages[0][
                0] == "outbound" and "To get started, please tell me the type of job" in last_messages[
                    0][1]:
            print("🔄 First response after reach out - forcing CONTINUATION")
            classification = "CONTINUATION"
        else:
            # Normal classification using only active conversation history
            recent_msgs = await msg_repo.get_recent_messages(
                customer=customer_phone_db,
                contractor=system_phone_db,
                conversation_id=old_convo.id,  # Only from this conversation
                limit=10)

            print(f"🔍 CLASSIFICATION DEBUG:")
            print(f"   Querying conversation: {old_convo.id}")
            print(f"   Found {len(recent_msgs)} recent messages")
            print(f"   Recent messages raw: {recent_msgs}")

            history = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                                for d, b in recent_msgs)
            print(f"   History for classification: {history!r}")

            classification = await classify_message(Body, history)
            print(f"🧠 CLASSIFICATION RESULT: {classification}")
    else:
        # No active conversation - must be NEW
        print("❌ No active conversation - treating as NEW")
        classification = "NEW"

    # Handle UNSURE classification
    if classification == "UNSURE":
        print("❓ UNSURE classification - asking for clarification")
        job_context = ""
        if old_convo:
            cd: ConversationData = await session.get(ConversationData,
                                                     old_convo.id)
            if cd:
                job_context = cd.job_title or cd.data_json.get("job_type", "")

        prompt = (
            f"Is this about your previous '{job_context}' job or a new one?" if
            job_context else "Is this about your previous job or a new one?")

        await msg_repo.create_message(
            sender=system_phone_db,
            receiver=customer_phone_db,
            body=prompt,
            direction="outbound",
            conversation_id=old_convo.id if old_convo else None)
        send_message(from_phone_clean, prompt, is_whatsapp)
        return Response(status_code=204)

    # Handle NEW classification
    if classification == "NEW":
        print("🆕 NEW classification - creating fresh conversation")
        if old_convo:
            print(f"   Closing old conversation: {old_convo.id}")
            await conv_repo.close_conversation(old_convo.id)

        convo = await conv_repo.create_conversation(
            contractor_id=contractor.id, customer_phone=customer_phone_db)
        print(f"   Created new conversation: {convo.id}")

        # Save the message that triggered the new conversation
        await msg_repo.create_message(sender=customer_phone_db,
                                      receiver=system_phone_db,
                                      body=Body,
                                      direction="inbound",
                                      conversation_id=convo.id)

        intro = f"Hi! I'm {contractor.name}'s assistant. To get started, please tell me the type of job you need."

        await msg_repo.create_message(sender=system_phone_db,
                                      receiver=customer_phone_db,
                                      body=intro,
                                      direction="outbound",
                                      conversation_id=convo.id)
        send_message(from_phone_clean, intro, is_whatsapp)
        return Response(status_code=204)

    # Handle CONTINUATION classification
    if classification == "CONTINUATION":
        print("➡️ CONTINUATION classification")
        if not old_convo:
            print("   No active convo - creating new one")
            # Create new conversation if none exists
            convo = await conv_repo.create_conversation(
                contractor_id=contractor.id, customer_phone=customer_phone_db)

            # Save the message
            await msg_repo.create_message(sender=customer_phone_db,
                                          receiver=system_phone_db,
                                          body=Body,
                                          direction="inbound",
                                          conversation_id=convo.id)

            intro = f"Hi! I'm {contractor.name}'s assistant. To get started, please tell me the type of job you need."

            await msg_repo.create_message(sender=system_phone_db,
                                          receiver=customer_phone_db,
                                          body=intro,
                                          direction="outbound",
                                          conversation_id=convo.id)
            send_message(from_phone_clean, intro, is_whatsapp)
            return Response(status_code=204)
        else:
            print(f"   Using existing conversation: {old_convo.id}")
            convo = old_convo

    # 6) Continue with qualification process
    print(f"📋 CONTINUING qualification process...")

    # Note: Message was already saved above, so we don't save it again here

    # Extract qualification data from full conversation
    full_msgs = await msg_repo.get_all_conversation_messages(convo.id)
    history = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                        for d, b in full_msgs)
    data = await extract_qualification_data(history)

    # ... rest of the qualification logic remains the same

    print(f"   Extracted data: {data}")

    # Store/update the qualification data
    await data_repo.upsert(
        conversation_id=convo.id,
        contractor_id=contractor.id,
        customer_phone=customer_phone_db,  # Store with channel prefix
        data_dict=data,
        qualified=is_qualified(data),
        job_title=data.get("job_type"))

    # Check for missing fields and prompt accordingly
    missing = [k for k in REQUIRED_FIELDS if not data[k]]
    print(f"   Missing fields: {missing}")

    if missing:
        if all(data[k] == "" for k in REQUIRED_FIELDS):
            ask = "Please provide your job type."
        else:
            next_fields = missing[:2]  # Ask for up to 2 fields at once
            labels = [f.replace("_", " ") for f in next_fields]
            ask = (f"Please provide your {labels[0]}." if len(labels) == 1 else
                   f"Please provide your {labels[0]} and {labels[1]}.")

        print(f"   Asking for: {ask}")
        await msg_repo.create_message(sender=system_phone_db,
                                      receiver=customer_phone_db,
                                      body=ask,
                                      direction="outbound",
                                      conversation_id=convo.id)
        send_message(from_phone_clean, ask, is_whatsapp)
        return Response(status_code=204)

    # If all fields are collected and we're in QUALIFYING state, show summary
    if convo.status == "QUALIFYING":
        print("✅ ALL FIELDS COLLECTED - showing summary")
        bullets = [
            f"• {f.replace('_',' ').title()}: {data[f]}"
            for f in REQUIRED_FIELDS
        ]
        summary = "Here's what I have so far:\n" + "\n".join(
            bullets) + "\nIs that correct?"

        await msg_repo.create_message(sender=system_phone_db,
                                      receiver=customer_phone_db,
                                      body=summary,
                                      direction="outbound",
                                      conversation_id=convo.id)
        send_message(from_phone_clean, summary, is_whatsapp)

        convo.status = "CONFIRMING"
        await session.commit()

    return Response(status_code=204)


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


@app.get("/pdf/{convo_id}")
def generate_pdf(convo_id: str):
    """Generate PDF report for a conversation."""
    path = f"/tmp/{convo_id}.pdf"
    return FileResponse(path, media_type="application/pdf")


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

# ===== PDF Security Functions =====

PDF_SECRET_KEY = os.environ.get("PDF_SECRET_KEY", "change-this-in-production")


def generate_pdf_token(conversation_id: str,
                       expires_in_hours: int = 24) -> str:
    """
    Generate a time-limited signed token for secure PDF access.
    Token format: conversation_id:expiry_timestamp:signature
    """
    expiry = int(
        (datetime.utcnow() + timedelta(hours=expires_in_hours)).timestamp())
    message = f"{conversation_id}:{expiry}"
    signature = hmac.new(PDF_SECRET_KEY.encode(), message.encode(),
                         hashlib.sha256).hexdigest()
    return f"{message}:{signature}"


def verify_pdf_token(token: str) -> tuple[bool, str]:
    """
    Verify PDF access token and extract conversation_id.
    Returns: (is_valid, conversation_id)
    """
    try:
        parts = token.split(':')
        if len(parts) != 3:
            logging.warning(f"Invalid token format: {token}")
            return False, ""

        conversation_id, expiry_str, provided_signature = parts
        expiry = int(expiry_str)

        # Check if token has expired
        if datetime.utcnow().timestamp() > expiry:
            logging.info(
                f"Token expired for conversation {conversation_id[:8]}...")
            return False, ""

        # Verify signature to prevent tampering
        message = f"{conversation_id}:{expiry_str}"
        expected_signature = hmac.new(PDF_SECRET_KEY.encode(),
                                      message.encode(),
                                      hashlib.sha256).hexdigest()

        # Use constant-time comparison to prevent timing attacks
        if hmac.compare_digest(expected_signature, provided_signature):
            return True, conversation_id

        logging.warning(
            f"Invalid signature for conversation {conversation_id[:8]}...")
        return False, ""

    except Exception as e:
        logging.error(f"Token verification error: {e}")
        return False, ""


def generate_pdf_url(conversation_id: str, base_url: str = None) -> str:
    """
    Generate a complete URL for PDF access with signed token.
    Used in digest messages to provide secure PDF links.
    """
    if not base_url:
        base_url = os.environ.get("APP_BASE_URL",
                                  "https://your-app.herokuapp.com")

    token = generate_pdf_token(conversation_id)
    # URL-encode the token to handle special characters
    return f"{base_url}/pdf/transcript?token={quote(token)}"


# ===== PDF Endpoints =====


@app.get("/pdf/transcript")
async def get_transcript_pdf(token: str, session=Depends(get_session)):
    """
    Secure endpoint for PDF transcript generation.
    Generates PDF on-demand to ensure most current data.

    Usage: /pdf/transcript?token=SIGNED_TOKEN

    The token contains:
    - conversation_id
    - expiry timestamp
    - HMAC signature for security
    """
    from services.pdf_service import generate_conversation_pdf

    # Verify the token
    is_valid, conversation_id = verify_pdf_token(token)

    if not is_valid:
        logging.warning(
            f"Invalid PDF access attempt with token: {token[:20]}...")
        return PlainTextResponse(
            "This link has expired or is invalid. Please request a new digest from your contractor.",
            status_code=403)

    try:
        # Log successful access
        logging.info(
            f"Generating PDF for conversation {conversation_id[:8]}...")

        # Generate the PDF with current data
        pdf_bytes = await generate_conversation_pdf(session, conversation_id)

        # Return PDF as streaming response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                # inline = display in browser, attachment = force download
                "Content-Disposition":
                f"inline; filename=lead_transcript_{conversation_id[:8]}.pdf",
                # Cache for 5 minutes to avoid regenerating on refresh
                "Cache-Control": "private, max-age=300",
                # Security headers
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY"
            })

    except ValueError as e:
        # Conversation not found
        logging.error(f"Conversation not found: {conversation_id}")
        return PlainTextResponse(
            "This conversation could not be found. It may have been deleted.",
            status_code=404)

    except Exception as e:
        # Unexpected error
        logging.error(f"PDF generation failed for {conversation_id}: {e}",
                      exc_info=True)
        return PlainTextResponse(
            "An error occurred generating the PDF. Please try again later.",
            status_code=500)


@app.get("/pdf/test/{convo_id}")
async def test_pdf_generation(convo_id: str, session=Depends(get_session)):
    """
    TEST ENDPOINT: Generate PDF directly without token (for development only).
    Remove this endpoint in production!
    """
    if os.environ.get("ENVIRONMENT") == "production":
        return PlainTextResponse("Not available in production",
                                 status_code=404)

    from services.pdf_service import generate_conversation_pdf

    try:
        pdf_bytes = await generate_conversation_pdf(session, convo_id)
        return Response(content=pdf_bytes,
                        media_type="application/pdf",
                        headers={
                            "Content-Disposition":
                            f"inline; filename=test_{convo_id[:8]}.pdf"
                        })
    except Exception as e:
        return PlainTextResponse(f"Error: {str(e)}", status_code=500)


@app.post("/generate-pdf-link")
async def generate_pdf_link_endpoint(conversation_id: str = Form(...)):
    """
    Admin endpoint to generate a PDF link for a specific conversation.
    Useful for testing or manual link generation.
    """
    # Add authentication here if needed

    url = generate_pdf_url(conversation_id)
    return {
        "conversation_id": conversation_id,
        "pdf_url": url,
        "expires_in": "24 hours",
        "generated_at": datetime.utcnow().isoformat()
    }
