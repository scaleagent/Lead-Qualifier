import os
import json
import traceback
import asyncio
import re
import logging
from datetime import datetime

from fastapi import FastAPI, Depends, Form, Response
from fastapi.responses import PlainTextResponse, FileResponse
from sqlalchemy.future import select

from openai import OpenAI
from twilio.rest import Client
from apscheduler.schedulers.background import BackgroundScheduler

# --- Database & Repos imports ---
from repos.database import async_engine, AsyncSessionLocal
from repos.models import Base, Contractor, ConversationData
from repos.contractor_repo import ContractorRepo
from repos.conversation_repo import ConversationRepo
from repos.message_repo import MessageRepo
from repos.conversation_data_repo import ConversationDataRepo

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


# === Helpers ===
def send_message(to_number: str, message: str, is_whatsapp: bool = False):
    """
    UNIFIED: Send message via SMS or WhatsApp using same logic.
    Channel determined by is_whatsapp flag, not phone number format.
    """
    # Format recipient based on channel
    if is_whatsapp:
        tw_to = f"whatsapp:{to_number}"
        from_number = WA_SANDBOX_NUMBER
        channel = "WhatsApp"
    else:
        tw_to = to_number
        from_number = TWILIO_NUMBER
        channel = "SMS"

    try:
        msg = twilio_client.messages.create(body=message,
                                            from_=from_number,
                                            to=tw_to)
        print(
            f"📤 Sent {channel} to {to_number} | SID: {msg.sid} | Status: {msg.status}"
        )
    except Exception:
        print(f"❌ Failed to send {channel} to {to_number}")
        traceback.print_exc()


async def classify_message(message_text: str, history_string: str) -> str:
    """
    Classify incoming message as NEW, CONTINUATION, or UNSURE using OpenAI.
    """
    system_prompt = (
        "You are an AI assistant for a trades business. Classify the incoming message as exactly one word: NEW, CONTINUATION, or UNSURE.\n"
        "NEW: user is requesting a completely new job or mentions new location/job type\n"
        "CONTINUATION: user is giving more info on an existing, in-progress job\n"
        "UNSURE: unclear whether it's new or a continuation\n"
        "If they mention a new location, job type, or pivot to different work, return NEW.\n"
        "Respond with exactly one word: NEW, CONTINUATION, or UNSURE.")

    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": system_prompt
        }, {
            "role":
            "user",
            "content":
            f"Message:\n{message_text}\n\nHistory:\n{history_string}"
        }],
        temperature=0,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip().upper()


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
    customer_phone = From.split(":", 1)[1] if is_whatsapp else From

    print(f"🔍 PARSED:")
    print(f"   is_whatsapp: {is_whatsapp}")
    print(f"   customer_phone: {customer_phone}")

    # Initialize repositories
    contractor_repo = ContractorRepo(session)
    conv_repo = ConversationRepo(session)
    msg_repo = MessageRepo(session)
    data_repo = ConversationDataRepo(session)

    # 1) Handle contractor-initiated "reach out" commands
    contractor = await contractor_repo.get_by_phone(customer_phone)
    if contractor:
        print(
            f"✅ CONTRACTOR IDENTIFIED: {contractor.name} (ID: {contractor.id})"
        )
        # Match both UK and international phone numbers
        m = re.match(r'^\s*reach out to (\+\d{10,15})\s*$', Body,
                     re.IGNORECASE)
        if m:
            target_phone = m.group(1)
            print(f"📞 REACH OUT COMMAND to: {target_phone}")

            # Close any existing conversation for this customer
            old_convo = await conv_repo.get_active_conversation(
                contractor.id, target_phone)
            if old_convo:
                print(f"🔄 CLOSING existing conversation: {old_convo.id}")
                await conv_repo.close_conversation(old_convo.id)

            # Create new conversation
            convo = await conv_repo.create_conversation(
                contractor_id=contractor.id, customer_phone=target_phone)
            print(f"🆕 CREATED new conversation: {convo.id}")

            # Send introduction message
            intro = (
                f"Hi! I'm {contractor.name}'s assistant. "
                "To get started, please tell me the type of job you need.")

            await msg_repo.create_message(
                sender=To,  # Our system number
                receiver=target_phone,
                body=intro,
                direction="outbound",
                conversation_id=convo.id)
            send_message(target_phone, intro, is_whatsapp)
            print(f"📤 SENT intro message to customer: {target_phone}")

        return Response(status_code=204)

    # 2) Handle customer-initiated messages
    print(f"👤 CUSTOMER MESSAGE from: {customer_phone}")

    # Get the first available contractor (single-contractor setup)
    result = await session.execute(select(Contractor))
    contractor = result.scalars().first()
    if not contractor:
        print("⚠️ No contractor found; dropping message.")
        return Response(status_code=204)

    print(f"🏢 USING contractor: {contractor.name} (ID: {contractor.id})")

    # Get any active conversation for this customer
    old_convo = await conv_repo.get_active_conversation(
        contractor.id, customer_phone)
    if old_convo:
        print(
            f"💬 FOUND active conversation: {old_convo.id} (status: {old_convo.status})"
        )
    else:
        print(f"❌ NO active conversation found for customer: {customer_phone}")

    # 3) Handle CONFIRMING and COLLECTING_NOTES states with priority
    if old_convo and old_convo.status in ("CONFIRMING", "COLLECTING_NOTES"):
        print(f"🔄 HANDLING {old_convo.status} state")

        # Log the incoming message
        await msg_repo.create_message(sender=customer_phone,
                                      receiver=To,
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
                await msg_repo.create_message(sender=To,
                                              receiver=customer_phone,
                                              body=follow,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_message(customer_phone, follow, is_whatsapp)
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

                await data_repo.upsert(conversation_id=old_convo.id,
                                       contractor_id=contractor.id,
                                       customer_phone=customer_phone,
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

                await msg_repo.create_message(sender=To,
                                              receiver=customer_phone,
                                              body=summary,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_message(customer_phone, summary, is_whatsapp)

            return Response(status_code=204)

        # COLLECTING_NOTES state: collect additional information
        if old_convo.status == "COLLECTING_NOTES":
            if is_negative(Body):
                print("✅ COMPLETED - closing conversation")
                # Complete the conversation
                closing = "Great—thanks! I'll pass this along to your contractor. ✅"
                await msg_repo.create_message(sender=To,
                                              receiver=customer_phone,
                                              body=closing,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_message(customer_phone, closing, is_whatsapp)
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
                await msg_repo.create_message(sender=To,
                                              receiver=customer_phone,
                                              body=prompt,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_message(customer_phone, prompt, is_whatsapp)

            return Response(status_code=204)

    # 4) Message classification for new/continuation logic
    print(f"🧠 STARTING message classification...")

    recent_msgs = await msg_repo.get_recent_messages(customer=customer_phone,
                                                     contractor=To,
                                                     limit=10)

    print(f"🔍 CLASSIFICATION DEBUG:")
    print(f"   Querying with customer={customer_phone}, contractor={To}")
    print(f"   Found {len(recent_msgs)} recent messages")
    print(f"   Recent messages raw: {recent_msgs}")

    history = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                        for d, b in recent_msgs)
    print(f"   History for classification: {history!r}")

    classification = await classify_message(Body, history)
    print(f"🧠 CLASSIFICATION RESULT: {classification}")

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
            sender=To,
            receiver=customer_phone,
            body=prompt,
            direction="outbound",
            conversation_id=old_convo.id if old_convo else None)
        send_message(customer_phone, prompt, is_whatsapp)
        return Response(status_code=204)

    # Handle NEW classification
    if classification == "NEW":
        print("🆕 NEW classification - creating fresh conversation")
        if old_convo:
            print(f"   Closing old conversation: {old_convo.id}")
            await conv_repo.close_conversation(old_convo.id)

        convo = await conv_repo.create_conversation(
            contractor_id=contractor.id, customer_phone=customer_phone)
        print(f"   Created new conversation: {convo.id}")

        intro = f"Hi! I'm {contractor.name}'s assistant. To get started, please tell me the type of job you need."

        await msg_repo.create_message(sender=To,
                                      receiver=customer_phone,
                                      body=intro,
                                      direction="outbound",
                                      conversation_id=convo.id)
        send_message(customer_phone, intro, is_whatsapp)
        return Response(status_code=204)

    # Handle CONTINUATION classification
    if classification == "CONTINUATION":
        print("➡️ CONTINUATION classification")
        if not old_convo:
            print("   No active convo - creating new one")
            # Create new conversation if none exists
            convo = await conv_repo.create_conversation(
                contractor_id=contractor.id, customer_phone=customer_phone)
            intro = f"Hi! I'm {contractor.name}'s assistant. To get started, please tell me the type of job you need."

            await msg_repo.create_message(sender=To,
                                          receiver=customer_phone,
                                          body=intro,
                                          direction="outbound",
                                          conversation_id=convo.id)
            send_message(customer_phone, intro, is_whatsapp)
            return Response(status_code=204)
        else:
            print(f"   Using existing conversation: {old_convo.id}")
            convo = old_convo

    # 5) Continue with qualification process
    print(f"📋 CONTINUING qualification process...")

    # Log the incoming message
    await msg_repo.create_message(sender=customer_phone,
                                  receiver=To,
                                  body=Body,
                                  direction="inbound",
                                  conversation_id=convo.id)

    # Extract qualification data from full conversation
    full_msgs = await msg_repo.get_all_conversation_messages(convo.id)
    history = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                        for d, b in full_msgs)
    data = await extract_qualification_data(history)

    print(f"   Extracted data: {data}")

    # Store/update the qualification data
    await data_repo.upsert(conversation_id=convo.id,
                           contractor_id=contractor.id,
                           customer_phone=customer_phone,
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
        await msg_repo.create_message(sender=To,
                                      receiver=customer_phone,
                                      body=ask,
                                      direction="outbound",
                                      conversation_id=convo.id)
        send_message(customer_phone, ask, is_whatsapp)
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

        await msg_repo.create_message(sender=To,
                                      receiver=customer_phone,
                                      body=summary,
                                      direction="outbound",
                                      conversation_id=convo.id)
        send_message(customer_phone, summary, is_whatsapp)

        convo.status = "CONFIRMING"
        await session.commit()

    return Response(status_code=204)


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
print("📱 Supporting both SMS and WhatsApp channels")
print("🔧 Debug mode enabled with enhanced logging")
# print("⏰ Daily digest scheduled for 6 PM")  # Commented out
