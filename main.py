import os
import json
import traceback
import asyncio
import uuid
from datetime import datetime

from fastapi import FastAPI, Depends, Form
from fastapi.responses import PlainTextResponse, FileResponse

from openai import OpenAI
from twilio.rest import Client
from apscheduler.schedulers.background import BackgroundScheduler

# --- Database & Repos imports ---
from repos.database import async_engine, AsyncSessionLocal
from repos.models import Base
from repos.conversation_repo import ConversationRepo
from repos.message_repo import MessageRepo
from repos.conversation_data_repo import ConversationDataRepo

app = FastAPI()

# === External API Clients ===
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
twilio_client = Client(os.environ.get("TWILIO_ACCOUNT_SID", ""),
                       os.environ.get("TWILIO_AUTH_TOKEN", ""))
TWILIO_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
DIGEST_TEST_RECEIVER = os.environ.get("DIGEST_TEST_RECEIVER")

# === Qualification Schema ===
# We prompt for these five fields, two at a time.
REQUIRED_FIELDS = [
    "job_type",
    "property_type",
    "urgency",
    "address",
    "access",
]

# === AI / SMS Utility Functions ===


def send_sms(to_number: str, message: str):
    """
    Send SMS via Twilio and log errors.
    """
    try:
        msg = twilio_client.messages.create(body=message,
                                            from_=TWILIO_NUMBER,
                                            to=to_number)
        print(f"📤 Sent SMS SID: {msg.sid} Status: {msg.status}")
    except Exception:
        print(f"❌ Failed to send SMS to {to_number}")
        traceback.print_exc()


async def classify_message(message_text: str, history_string: str) -> str:
    """
    Classify whether a message starts a new job, continues an existing one, or is unclear.
    """
    messages = [{
        "role":
        "system",
        "content":
        ("You're an AI assistant for a trades business. Classify messages as:\n"
         "- NEW: a new job request\n"
         "- CONTINUATION: same job\n"
         "- UNSURE: unclear\n"
         "If new message mentions new location/job/type, classify as NEW.")
    }, {
        "role":
        "user",
        "content":
        f"Message:\n{message_text}\n\nHistory:\n{history_string}"
    }]
    response = openai.chat.completions.create(model="gpt-3.5-turbo",
                                              messages=messages,
                                              max_tokens=50)
    return response.choices[0].message.content.strip().upper()


async def extract_qualification_data(history_string: str) -> dict:
    """
    Extract structured qualification fields from conversation history.
    Always return all six keys, defaulting to empty string.
    """
    system_prompt = (
        "You are an assistant extracting exactly these fields from the conversation:\n"
        "  job_type, property_type, urgency, address, access, notes\n"
        "Respond ONLY with JSON, for example:\n"
        "{\n"
        '  "job_type": "Electrical repair",\n'
        '  "property_type": "House",\n'
        '  "urgency": "ASAP",\n'
        '  "address": "123 Main St",\n'
        '  "access": "Key under mat",\n'
        '  "notes": "Customer has dog on premises"\n'
        "}")
    messages = [{
        "role": "system",
        "content": system_prompt
    }, {
        "role": "user",
        "content": f"Conversation:\n{history_string}"
    }]
    response = openai.chat.completions.create(model="gpt-3.5-turbo",
                                              messages=messages,
                                              max_tokens=300)
    try:
        data = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        data = {}

    # Ensure all keys exist
    for key in REQUIRED_FIELDS + ["notes"]:
        data.setdefault(key, "")
    return data


def is_qualified(data: dict) -> bool:
    """
    Check if all required qualification fields are present.
    """
    return all(data.get(k) for k in REQUIRED_FIELDS)


async def generate_reply(chat_messages: list) -> str:
    """
    Generate a free-form AI assistant reply given chat messages.
    """
    response = openai.chat.completions.create(model="gpt-3.5-turbo",
                                              messages=chat_messages,
                                              max_tokens=150)
    return response.choices[0].message.content.strip()


def build_chat_messages(conversation: list) -> list:
    """
    Build OpenAI chat messages including a system prompt and history.
    """
    messages = [{
        "role":
        "system",
        "content":
        ("You are a helpful assistant for a trades business. Ask questions to qualify the job. "
         "Always finish your responses completely. Ask clear, unambiguous questions."
         )
    }]
    for direction, body in conversation:
        role = "user" if direction == "inbound" else "assistant"
        messages.append({"role": role, "content": body})
    return messages


# === Startup: create DB tables if missing ===
@app.on_event("startup")
async def on_startup():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# === Health-check endpoint ===
@app.get("/", response_class=PlainTextResponse)
def read_root():
    return "✅ SMS-Lead-Qual API is running."


# === Dependency: Async DB session per request ===
async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


# === SMS Webhook Handler ===
@app.post("/sms", response_class=PlainTextResponse)
async def sms_webhook(
        From: str = Form(...),
        To: str = Form(...),
        Body: str = Form(...),
        session=Depends(get_session),
):
    conv_repo = ConversationRepo(session)
    msg_repo = MessageRepo(session)
    data_repo = ConversationDataRepo(session)

    # 1) Log inbound SMS
    await msg_repo.create_message(sender=From,
                                  receiver=To,
                                  body=Body,
                                  direction="inbound")

    # 2) Handle “[CONTACTED xyz]” from contractor
    if Body.strip().upper().startswith("[CONTACTED"):
        job = Body.strip()[10:].strip(" ]").lower()
        await data_repo.mark_handed_over(job)
        await msg_repo.create_message(sender=To,
                                      receiver=From,
                                      body=f"Marked '{job}' as handed over.",
                                      direction="outbound")
        #return "OK"

    # 3) Fetch or start conversation
    convo = await conv_repo.get_active_conversation(contractor_phone=To,
                                                    customer_phone=From)
    if not convo:
        convo = await conv_repo.create_conversation(contractor_phone=To,
                                                    customer_phone=From)

    # 4) Get recent history for classification
    recent = await msg_repo.get_recent_messages(customer=From,
                                                contractor=To,
                                                limit=10)
    history_str = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                            for d, b in recent)

    # 5) Classify message
    classification = await classify_message(Body, history_str)
    if classification == "NEW":
        await conv_repo.close_conversation(convo.id)
        convo = await conv_repo.create_conversation(contractor_phone=To,
                                                    customer_phone=From)
    elif classification == "UNSURE":
        followup = "Hi! Is this about your previous job or a new one?"
        await msg_repo.create_message(sender=To,
                                      receiver=From,
                                      body=followup,
                                      direction="outbound")
        send_sms(From, followup)
        #return "OK"

    # 6) Full history for extraction
    full = await msg_repo.get_all_conversation_messages(convo.id)
    full_hist = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                          for d, b in full)

    # 7) Extract & upsert qualification data
    data = await extract_qualification_data(full_hist)
    qualified_flag = is_qualified(data)
    await data_repo.upsert(
        conversation_id=convo.id,
        contractor_phone=To,
        customer_phone=From,
        data_dict=data,
        qualified=qualified_flag,
        job_title=data.get("job_type")  # temporary
    )

    # 8) Two-fields-per-message prompt
    missing = [k for k in REQUIRED_FIELDS if not data.get(k)]
    if missing:
        next_two = missing[:2]
        # humanize field names
        labels = [f.replace("_", " ") for f in next_two]
        ask = f"Please provide your {labels[0]} and {labels[1]}."
        await msg_repo.create_message(sender=To,
                                      receiver=From,
                                      body=ask,
                                      direction="outbound")
        send_sms(From, ask)
        #return "OK"

    # 9) (Future) Confirmation and post‐qualification follow-up go here

    # 10) Default: free-form AI reply
    reply = await generate_reply(build_chat_messages(full))
    await msg_repo.create_message(sender=To,
                                  receiver=From,
                                  body=reply,
                                  direction="outbound")
    send_sms(From, reply)

    #return "OK"


# === PDF Generation Endpoint ===
@app.get("/pdf/{convo_id}")
def generate_pdf(convo_id: str):
    path = f"/tmp/{convo_id}.pdf"
    return FileResponse(path, media_type="application/pdf")


# === Daily Digest Task ===
async def run_daily_digest():
    async with AsyncSessionLocal() as session:
        data_repo = ConversationDataRepo(session)
        all_leads = await data_repo.get_all()

    per_contractor: dict[str, list] = {}
    for lead in all_leads:
        per_contractor.setdefault(lead.contractor_phone, []).append(lead)

    today = datetime.utcnow().strftime("%d/%m")
    for contractor, leads in per_contractor.items():
        lines = [f"📊 TODAY'S LEADS ({today})"]
        complete = [l for l in leads if l.qualified]
        if complete:
            lines.append("✅ Complete:")
            for l in complete:
                d = l.data_json
                pdf_url = f"https://{os.environ['REPLIT_DOMAIN']}/pdf/{l.conversation_id}"
                lines.append(
                    f"- {d.get('job_type','')} | {d.get('property_type','')} | {d.get('urgency','')} | {d.get('address','')}\n  View: {pdf_url}"
                )
        incomplete = [l for l in leads if not l.qualified]
        if incomplete:
            lines.append("⏸️ Incomplete:")
            for l in incomplete:
                d = l.data_json
                missing = [k for k in REQUIRED_FIELDS if not d.get(k)]
                last = l.last_updated.strftime("%d/%m %H:%M")
                lines.append(
                    f"- {d.get('job_type','')} ({l.customer_phone}), last update {last}\n"
                    f"  Missing: {', '.join(missing)}")
        body = "\n".join(lines)
        send_sms(contractor, body)


# === Scheduler setup ===
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: asyncio.create_task(run_daily_digest()),
                  "cron",
                  hour=18)
scheduler.start()
