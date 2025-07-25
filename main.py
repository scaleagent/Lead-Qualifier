import os
import json
import traceback
import asyncio
import re
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

# === Helpers ===


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
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=50,
    )
    return resp.choices[0].message.content.strip().upper()


async def extract_qualification_data(history_string: str) -> dict:
    system_prompt = (
        "Extract exactly these fields from the conversation, based only on what the CUSTOMER explicitly said.\n"
        "Fields: job_type, property_type, urgency, address, access, notes.\n"
        "- If NOT mentioned, set value to empty string.\n"
        "- Do NOT guess or infer.\n"
        "- Put all other customer comments into 'notes'.\n"
        "Respond ONLY with a JSON object with exactly these six keys.")
    msgs = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Conversation:\n{history_string}"
        },
    ]
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=msgs,
        temperature=0,
        top_p=1,
        max_tokens=300,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        print("❗️ JSON parse error:", resp.choices[0].message.content)
        data = {}
    for key in REQUIRED_FIELDS + ["notes"]:
        data.setdefault(key, "")
    return data


async def apply_correction_data(current: dict, correction: str) -> dict:
    """
    Use the model to apply a customer correction to the existing JSON.
    """
    system_prompt = (
        "You are a JSON assistant. Given existing job data and a user's correction,"
        " return the updated JSON with the same six keys only.")
    user_prompt = (f"Existing data: {json.dumps(current)}\n"
                   f"Correction: {correction}\n"
                   "Respond ONLY with the full updated JSON.")
    msgs = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        },
    ]
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=msgs,
        temperature=0,
        top_p=1,
        max_tokens=300,
    )
    try:
        updated = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        print("❗️ Correction JSON parse error:",
              resp.choices[0].message.content)
        updated = current
    for key in REQUIRED_FIELDS + ["notes"]:
        updated.setdefault(key, current.get(key, ""))
    return updated


def is_affirmative(text: str) -> bool:
    return bool(
        re.match(r"^(yes|yep|yeah|correct|that is correct)\b",
                 text.strip().lower()))


def is_qualified(data: dict) -> bool:
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

    # 1) Handle "[CONTACTED ...]" if ever needed
    if Body.strip().upper().startswith("[CONTACTED"):
        # (we’ll scope this to contractors later)
        job = Body.strip()[10:].strip(" ]").lower()
        await data_repo.mark_handed_over(job)
        note = f"Marked '{job}' as handed over."
        await msg_repo.create_message(To, From, note, "outbound")
        send_sms(From, note)
        return Response(status_code=204)

    # 2) Find or start QUALIFYING conversation
    convo = await conv_repo.get_active_conversation(To, From)
    if not convo:
        # No active convo → classification for ambiguity
        recent = await msg_repo.get_recent_messages(From, To, limit=10)
        hist = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                         for d, b in recent)
        cls = await classify_message(Body, hist)
        if cls == "UNSURE":
            follow = "Hi! Is this about your previous job or a new one?"
            await msg_repo.create_message(To, From, follow, "outbound")
            send_sms(From, follow)
            return Response(status_code=204)
        convo = await conv_repo.create_conversation(To, From)

    # 3) Log inbound with conversation_id
    await msg_repo.create_message(From,
                                  To,
                                  Body,
                                  "inbound",
                                  conversation_id=convo.id)

    # 4) Fetch this conversation's history & extract data
    full = await msg_repo.get_all_conversation_messages(convo.id)
    hist = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                     for d, b in full)
    data = await extract_qualification_data(hist)
    print("🔍 Extracted data:", data)
    await data_repo.upsert(
        conversation_id=convo.id,
        contractor_phone=To,
        customer_phone=From,
        data_dict=data,
        qualified=is_qualified(data),
        job_title=data.get("job_type"),
    )

    # 5) Two-field prompts if still missing
    missing = [k for k in REQUIRED_FIELDS if not data[k]]
    print("🔎 Missing fields:", missing)
    if missing:
        nxt = missing[:2]
        labels = [f.replace("_", " ") for f in nxt]
        ask = (f"Please provide your {labels[0]}." if len(labels) == 1 else
               f"Please provide your {labels[0]} and {labels[1]}.")
        await msg_repo.create_message(To,
                                      From,
                                      ask,
                                      "outbound",
                                      conversation_id=convo.id)
        send_sms(From, ask)
        return Response(status_code=204)

    # 6) Confirmation loop
    if convo.status == "QUALIFYING":
        # First time all fields present: send summary
        lines = [
            f"• {f.replace('_', ' ').title()}: {data[f]}"
            for f in REQUIRED_FIELDS
        ]
        summary = "Here’s what I have so far:\n" + "\n".join(
            lines) + "\nIs that correct?"
        await msg_repo.create_message(To,
                                      From,
                                      summary,
                                      "outbound",
                                      conversation_id=convo.id)
        send_sms(From, summary)
        convo.status = "CONFIRMING"
        await session.commit()
        return Response(status_code=204)

    if convo.status == "CONFIRMING":
        # Handle customer response to summary
        if is_affirmative(Body):
            # Affirmed → post-qualification invite & close
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
        else:
            # Correction → apply and re-confirm
            updated = await apply_correction_data(data, Body)
            print("🔄 Corrected data:", updated)
            await data_repo.upsert(
                conversation_id=convo.id,
                contractor_phone=To,
                customer_phone=From,
                data_dict=updated,
                qualified=is_qualified(updated),
                job_title=updated.get("job_type"),
            )
            # Resend summary
            lines = [
                f"• {f.replace('_', ' ').title()}: {updated[f]}"
                for f in REQUIRED_FIELDS
            ]
            summary = "Got it! Here’s the updated info:\n" + "\n".join(
                lines) + "\nIs that correct?"
            await msg_repo.create_message(To,
                                          From,
                                          summary,
                                          "outbound",
                                          conversation_id=convo.id)
            send_sms(From, summary)
            # status remains CONFIRMING
            return Response(status_code=204)

    # (No free-form fallback any more)

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
