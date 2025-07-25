import os
import json
import traceback
import asyncio
from datetime import datetime

from fastapi import FastAPI, Depends, Form, Response
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

# === Qualification Schema ===
REQUIRED_FIELDS = [
    "job_type",
    "property_type",
    "urgency",
    "address",
    "access",
]


def send_sms(to_number: str, message: str):
    try:
        msg = twilio_client.messages.create(body=message,
                                            from_=TWILIO_NUMBER,
                                            to=to_number)
        print(f"📤 Sent SMS SID: {msg.sid} Status: {msg.status}")
    except Exception:
        print(f"❌ Failed to send SMS to {to_number}")
        traceback.print_exc()


async def classify_message(message_text: str, history_string: str) -> str:
    messages = [
        {
            "role":
            "system",
            "content":
            ("You're an AI assistant for a trades business. Classify messages as:\n"
             "- NEW: a new job request\n"
             "- CONTINUATION: same job\n"
             "- UNSURE: unclear\n"
             "If the new message mentions a new location, job, or type, classify as NEW."
             ),
        },
        {
            "role": "user",
            "content":
            f"Message:\n{message_text}\n\nHistory:\n{history_string}",
        },
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=50,
    )
    return response.choices[0].message.content.strip().upper()


async def extract_qualification_data(history_string: str) -> dict:
    system_prompt = (
        "Extract exactly these fields from the conversation, based only on what the CUSTOMER explicitly said.\n"
        "Fields to extract: job_type, property_type, urgency, address, access, notes.\n"
        "- If the customer did NOT mention a field, set its value to an empty string.\n"
        "- Do NOT guess or infer anything not explicitly provided.\n"
        "- Put any other customer comments into 'notes'.\n"
        "Respond ONLY with a JSON object with these six keys.")
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Conversation:\n{history_string}"
        },
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=300,
    )
    try:
        data = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print("❗️ JSON parse error:", response.choices[0].message.content)
        data = {}
    for key in REQUIRED_FIELDS + ["notes"]:
        data.setdefault(key, "")
    return data


def is_qualified(data: dict) -> bool:
    return all(data.get(k) for k in REQUIRED_FIELDS)


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

    # 1) Handle “[CONTACTED xyz]” (no conversation_id needed)
    if Body.strip().upper().startswith("[CONTACTED"):
        job = Body.strip()[10:].strip(" ]").lower()
        await data_repo.mark_handed_over(job)
        note = f"Marked '{job}' as handed over."
        await msg_repo.create_message(To, From, note, "outbound")
        send_sms(From, note)
        return Response(status_code=204)

    # 2) Fetch or start a QUALIFYING conversation
    convo = await conv_repo.get_active_conversation(To, From)
    history_str = ""
    if not convo:
        # Classification only when there's truly NO active convo
        recent = await msg_repo.get_recent_messages(From, To, limit=10)
        history_str = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                                for d, b in recent)
        classification = await classify_message(Body, history_str)
        if classification == "UNSURE":
            follow = "Hi! Is this about your previous job or a new one?"
            await msg_repo.create_message(To, From, follow, "outbound")
            send_sms(From, follow)
            return Response(status_code=204)
        convo = await conv_repo.create_conversation(To, From)

    # 3) Log inbound SMS with conversation_id
    await msg_repo.create_message(From,
                                  To,
                                  Body,
                                  "inbound",
                                  conversation_id=convo.id)

    # 4) Pull & extract data from this conversation only
    full = await msg_repo.get_all_conversation_messages(convo.id)
    full_hist = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                          for d, b in full)
    data = await extract_qualification_data(full_hist)
    print("🔍 Extracted data:", data)
    qualified_flag = is_qualified(data)
    await data_repo.upsert(
        conversation_id=convo.id,
        contractor_phone=To,
        customer_phone=From,
        data_dict=data,
        qualified=qualified_flag,
        job_title=data.get("job_type"),
    )

    # 5) Two-fields-per-message prompt
    missing = [k for k in REQUIRED_FIELDS if not data[k]]
    print("🔎 Missing fields:", missing)
    if missing:
        next_two = missing[:2]
        labels = [f.replace("_", " ") for f in next_two]
        ask = (f"Please provide your {labels[0]}." if len(labels) == 1 else
               f"Please provide your {labels[0]} and {labels[1]}.")
        await msg_repo.create_message(To,
                                      From,
                                      ask,
                                      "outbound",
                                      conversation_id=convo.id)
        send_sms(From, ask)
        return Response(status_code=204)

    # 6) Post-qualification invite & close
    follow_up = (
        "Thanks! If there’s any other important info—parking, pets, special access—"
        "just reply here. I’ll pass it along to your electrician.")
    await msg_repo.create_message(To,
                                  From,
                                  follow_up,
                                  "outbound",
                                  conversation_id=convo.id)
    send_sms(From, follow_up)
    await conv_repo.close_conversation(convo.id)
    return Response(status_code=204)


@app.get("/pdf/{convo_id}")
def generate_pdf(convo_id: str):
    path = f"/tmp/{convo_id}.pdf"
    return FileResponse(path, media_type="application/pdf")


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
                pdf_url = f"https://{os.environ.get('REPLIT_DOMAIN')}/pdf/{l.conversation_id}"
                lines.append(
                    f"- {d.get('job_type','')} | {d.get('property_type','')} | "
                    f"{d.get('urgency','')} | {d.get('address','')}\n"
                    f"  View: {pdf_url}")
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


scheduler = BackgroundScheduler()
scheduler.add_job(lambda: asyncio.create_task(run_daily_digest()),
                  "cron",
                  hour=18)
scheduler.start()
