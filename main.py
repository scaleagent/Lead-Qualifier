import os
import json
import traceback
import asyncio
import re
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
def send_sms(to_number: str, message: str):
    """
    Send a message, remapping our internal or Twilio prefixes appropriately.
    """
    # If internal wa: prefix, convert to Twilio whatsapp:
    if to_number.startswith("wa:"):
        tw_to = "whatsapp:" + to_number[3:]
    else:
        tw_to = to_number
    # Choose sender
    from_number = WA_SANDBOX_NUMBER if tw_to.startswith("whatsapp:") else TWILIO_NUMBER
    try:
        msg = twilio_client.messages.create(
            body=message,
            from_=from_number,
            to=tw_to
        )
        print(f"📤 Sent from {from_number!r} to {tw_to!r} SID: {msg.sid} Status: {msg.status}")
    except Exception:
        print(f"❌ Failed to send to {tw_to!r}")
        traceback.print_exc()

async def classify_message(message_text: str, history_string: str) -> str:
    prompt = (
        "You are an AI assistant for a trades business. Classify the incoming message as exactly one word: NEW, CONTINUATION, or UNSURE."
        " If they mention a new location, job type, or pivot, return NEW."
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Message:\n{message_text}\n\nHistory:\n{history_string}"}
        ],
        temperature=0,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip().upper()

async def extract_qualification_data(history_string: str) -> dict:
    prompt = (
        "Extract exactly these fields from the customer's messages: job_type, property_type, urgency, address, access, notes."
        " If missing, set empty string. Put extras in notes. Return only JSON."
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Conversation:\n{history_string}"}
        ],
        temperature=0,
        max_tokens=300,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        print("❗️ JSON parse error:", resp.choices[0].message.content)
        data = {}
    for k in REQUIRED_FIELDS + ["notes"]:
        data.setdefault(k, "")
    return data

async def apply_correction_data(current: dict, correction: str) -> dict:
    prompt = (
        "Given existing JSON data and a user correction, return updated JSON with the same six keys only."
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Existing: {json.dumps(current)}\nCorrection: {correction}"}
        ],
        temperature=0,
        max_tokens=300,
    )
    try:
        updated = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        print("❗️ Correction parse error:", resp.choices[0].message.content)
        updated = current
    for k in REQUIRED_FIELDS + ["notes"]:
        updated.setdefault(k, current.get(k, ""))
    return updated

def is_affirmative(text: str) -> bool:
    return bool(re.match(r"^(yes|yep|yeah|correct)", text.strip().lower()))

def is_negative(text: str) -> bool:
    return bool(re.match(r"^(no|nope|all done|that'?s everything)", text.strip().lower()))

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
    From: str = Form(...), To: str = Form(...), Body: str = Form(...),
    session=Depends(get_session)
):
    From, To, Body = From.strip(), To.strip(), Body.strip()
    print(f"🔔 Incoming From={From}, To={To}, Body={Body!r}")
    # Channel detection
    is_whatsapp = From.startswith("whatsapp:")
    raw_from = From.split(":",1)[1] if is_whatsapp else From
    customer_db = f"wa:{raw_from}" if is_whatsapp else raw_from
    reply_to = f"whatsapp:{raw_from}" if is_whatsapp else raw_from

    contractor_repo = ContractorRepo(session)
    conv_repo = ConversationRepo(session)
    msg_repo = MessageRepo(session)
    data_repo = ConversationDataRepo(session)

    # 1) Contractor-initiated "reach out"
    contractor = await contractor_repo.get_by_phone(raw_from)
    if contractor:
        m = re.match(r'^reach out to (\+\d+)$', Body, re.IGNORECASE)
        if m:
            target = m.group(1)
            # close any existing
            old = await conv_repo.get_active_conversation(contractor.id, target)
            if old:
                await conv_repo.close_conversation(old.id)
            convo = await conv_repo.create_conversation(
                contractor_id=contractor.id,
                customer_phone=customer_db
            )
            intro = (
                f"Hi! I’m {contractor.name}’s assistant."
                " To get started, please tell me the type of job you need."
            )
            send_to = f"whatsapp:{target}" if is_whatsapp else target
            from_addr = WA_SANDBOX_NUMBER if is_whatsapp else TWILIO_NUMBER
            await msg_repo.create_message(
                sender=from_addr,
                receiver=send_to,
                body=intro,
                direction="outbound",
                conversation_id=convo.id
            )
            send_sms(send_to, intro)
        return Response(status_code=204)

    # 2) Customer-initiated flow
    res = await session.execute(select(Contractor))
    contractor = res.scalars().first()
    if not contractor:
        return Response(status_code=204)
    old_convo = await conv_repo.get_active_conversation(contractor.id, customer_db)

    # 3) CONFIRMING / COLLECTING_NOTES
    if old_convo and old_convo.status in ("CONFIRMING","COLLECTING_NOTES"):
        await msg_repo.create_message(
            sender=customer_db,
            receiver=TWILIO_NUMBER,
            body=Body,
            direction="inbound",
            conversation_id=old_convo.id
        )
        if old_convo.status == "CONFIRMING":
            if is_affirmative(Body):
                follow = "Thanks! Any other info? When done reply 'No'."
                await msg_repo.create_message(
                    sender=TWILIO_NUMBER,
                    receiver=reply_to,
                    body=follow,
                    direction="outbound",
                    conversation_id=old_convo.id
                )
                send_sms(reply_to, follow)
                old_convo.status = "COLLECTING_NOTES"
                await session.commit()
            else:
                full_msgs = await msg_repo.get_all_conversation_messages(old_convo.id)
                history = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}" for d,b in full_msgs)
                data = await extract_qualification_data(history)
                updated = await apply_correction_data(data, Body)
                await data_repo.upsert(
                    conversation_id=old_convo.id,
                    contractor_id=contractor.id,
                    customer_phone=customer_db,
                    data_dict=updated,
                    qualified=is_qualified(updated),
                    job_title=updated.get("job_type")
                )
                bullets = [f"• {f.replace('_',' ').title()}: {updated[f]}" for f in REQUIRED_FIELDS]
                summary = "Got it! Here’s the updated info:\n" + "\n".join(bullets) + "\nIs that correct?"
                await msg_repo.create_message(
                    sender=TWILIO_NUMBER,
                    receiver=reply_to,
                    body=summary,
                    direction="outbound",
                    conversation_id=old_convo.id
                )
                send_sms(reply_to, summary)
            return Response(status_code=204)
        if old_convo.status == "COLLECTING_NOTES":
            if is_negative(Body):
                closing = "Great—thanks! I’ll pass this along. ✅"
                await msg_repo.create_message(
                    sender=TWILIO_NUMBER,
                    receiver=reply_to,
                    body=closing,
                    direction="outbound",
                    conversation_id=old_convo.id
                )
                send_sms(reply_to, closing)
                await conv_repo.close_conversation(old_convo.id)
            else:
                cd: ConversationData = await session.get(ConversationData, old_convo.id)
                combined = (cd.data_json.get("notes", "") + "\n" + Body).strip()
                cd.data_json["notes"] = combined
                cd.last_updated = datetime.utcnow()
                await session.commit()
                prompt = "Anything else? If not, reply 'No'."
                await msg_repo.create_message(
                    sender=TWILIO_NUMBER,
                    receiver=reply_to,
                    body=prompt,
                    direction="outbound",
                    conversation_id=old_convo.id
                )
                send_sms(reply_to, prompt)
            return Response(status_code=204)

    # 4) Classification / pivot
    recent = await msg_repo.get_recent_messages(customer=customer_db, contractor=TWILIO_NUMBER, limit=10)
    history = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}" for d,b in recent)
    cls = await classify_message(Body, history)
    print(f"🧠 Classification: {cls}")
    if cls == "UNSURE":
        prompt = "Is this about your previous job or a new one?"
        send_sms(reply_to, prompt)
        return Response(status_code=204)
    if cls == "NEW":
        if old_convo:
            await conv_repo.close_conversation(old_convo.id)
        convo = await conv_repo.create_conversation(contractor.id, customer_db)
        intro = f"Hi! I’m {contractor.name}’s assistant. To get started, please tell me the type of job you need."
        from_addr = WA_SANDBOX_NUMBER if is_whatsapp else TWILIO_NUMBER
        await msg_repo.create_message(sender=from_addr, receiver=reply_to, body=intro, direction="outbound", conversation
