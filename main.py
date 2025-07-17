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
    """
    messages = [{
        "role":
        "system",
        "content":
        ("Extract job details. Return JSON with job_type, urgency, timeline, "
         "property_type, name, notes, and a short job_title.")
    }, {
        "role": "user",
        "content": f"Conversation:\n{history_string}"
    }]
    response = openai.chat.completions.create(model="gpt-3.5-turbo",
                                              messages=messages,
                                              max_tokens=200)
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {}


def is_qualified(data: dict) -> bool:
    """
    Check if all required qualification fields are present.
    """
    return all(
        data.get(k)
        for k in ["job_type", "urgency", "timeline", "property_type"])


async def generate_reply(chat_messages: list) -> str:
    """
    Generate an AI assistant reply given chat messages.
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
    """
    Simple root endpoint so GET / returns 200 OK.
    """
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
    """
    Handles incoming SMS via Twilio webhook:
     1) Logs inbound message
     2) Processes “[CONTACTED ...]” commands
     3) Finds or creates a conversation
     4) Classifies new vs continuation
     5) Extracts qualification data
     6) Upserts conversation_data
     7) Generates & sends AI reply
    """
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

    # 4) Get recent history
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
        await msg_repo.create_message(
            sender=To,
            receiver=From,
            body="Hi! Is this about your previous job or a new one?",
            direction="outbound")
        #return "OK"

    # 6) Extract data
    full = await msg_repo.get_all_conversation_messages(convo.id)
    full_hist = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                          for d, b in full)
    data = await extract_qualification_data(full_hist)
    qualified = is_qualified(data)

    # 7) Upsert qualification data
    await data_repo.upsert(
        conversation_id=convo.id,
        contractor_phone=To,
        customer_phone=From,
        data_dict=data,
        qualified=qualified,
        job_title=data.get("job_title"),
    )

    # 8) Generate AI reply & respond
    reply = await generate_reply(build_chat_messages(full))
    await msg_repo.create_message(sender=To,
                                  receiver=From,
                                  body=reply,
                                  direction="outbound")
    send_sms(From, reply)

    return "OK"


# === PDF Generation Endpoint ===
@app.get("/pdf/{convo_id}")
def generate_pdf(convo_id: str):
    """
    Generate a PDF of the full conversation history.
    """
    path = f"/tmp/{convo_id}.pdf"
    return FileResponse(path, media_type="application/pdf")


# === Daily Digest Task (TODO) ===
async def run_daily_digest():
    """
    Gather all leads, group by contractor, format a SMS digest,
    and send via Twilio at 18:00 each day.
    """
    # 1) Spin up a short-lived DB session
    async with AsyncSessionLocal() as session:
        data_repo = ConversationDataRepo(session)
        all_leads = await data_repo.get_all()

    # 2) Group by contractor_phone
    per_contractor: dict[str, list] = {}
    for lead in all_leads:
        per_contractor.setdefault(lead.contractor_phone, []).append(lead)

    # 3) Build & send one digest per contractor
    today = datetime.utcnow().strftime("%d/%m")
    for contractor, leads in per_contractor.items():
        lines = [f"📊 TODAY'S LEADS ({today})"]

        # — Complete leads
        complete = [l for l in leads if l.qualified]
        if complete:
            lines.append("✅ Complete:")
            for l in complete:
                d = l.data_json
                # PDF link (needs REPLIT_DOMAIN env var set for your host)
                pdf_url = f"https://{os.environ['REPLIT_DOMAIN']}/pdf/{l.conversation_id}"
                lines.append(
                    f"- {d.get('name','')} ({l.customer_phone}): {d.get('job_type','')} | "
                    f"{d.get('timeline','')} | {d.get('property_type','')}\n  View: {pdf_url}"
                )

        # — Incomplete leads
        incomplete = [l for l in leads if not l.qualified]
        if incomplete:
            lines.append("⏸️ Incomplete:")
            for l in incomplete:
                d = l.data_json
                # find which fields are missing
                missing = [k for k in ["job_type","urgency","timeline","property_type"] if not d.get(k)]
                last = l.last_updated.strftime("%d/%m %H:%M")
                lines.append(
                    f"- {d.get('name','')} ({l.customer_phone}), last update {last}\n"
                    f"  Missing: {', '.join(missing)}"
                )

        # 4) Send the SMS
        body = "\n".join(lines)
        send_sms(contractor, body)


# === Scheduler setup ===
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: asyncio.create_task(run_daily_digest()),
                  "cron",
                  hour=18)
scheduler.start()
