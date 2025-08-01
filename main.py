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
    Send a message choosing the right sender based on channel.
    """
    from_number = WA_SANDBOX_NUMBER if to_number.startswith(
        "whatsapp:") else TWILIO_NUMBER
    try:
        msg = twilio_client.messages.create(body=message,
                                            from_=from_number,
                                            to=to_number)
        print(
            f"📤 Sent from {from_number!r} to {to_number!r} SID: {msg.sid} Status: {msg.status}"
        )
    except Exception:
        print(f"❌ Failed to send to {to_number!r}")
        traceback.print_exc()


async def classify_message(message_text: str, history_string: str) -> str:
    system = (
        "You are an AI assistant for a trades business. Classify the incoming message as one word: NEW, CONTINUATION, or UNSURE. "
        "If they mention a new location, job type, or pivot, return NEW.")
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": system
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
    system = (
        "Extract exactly these fields from the customer's messages: job_type, property_type, urgency, address, access, notes. "
        "Set missing fields to empty strings. Include other comments in notes. Return only the JSON object."
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": system
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
        print("JSON parse error:", resp.choices[0].message.content)
        data = {}
    for k in REQUIRED_FIELDS + ["notes"]:
        data.setdefault(k, "")
    return data


async def apply_correction_data(current: dict, correction: str) -> dict:
    system = (
        "Given existing JSON data and a user correction, return updated JSON with the same keys only."
    )
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": system
        }, {
            "role":
            "user",
            "content":
            f"Existing: {json.dumps(current)}\nCorrection: {correction}"
        }],
        temperature=0,
        max_tokens=300,
    )
    try:
        updated = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        print("Correction parse error:", resp.choices[0].message.content)
        updated = current
    for k in REQUIRED_FIELDS + ["notes"]:
        updated.setdefault(k, current.get(k, ""))
    return updated


def is_affirmative(text: str) -> bool:
    return bool(re.match(r"^(yes|yep|yeah|correct)", text.strip().lower()))


def is_negative(text: str) -> bool:
    return bool(
        re.match(r"^(no|nope|all done|that'?s everything)",
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
async def sms_webhook(From: str = Form(...),
                      To: str = Form(...),
                      Body: str = Form(...),
                      session=Depends(get_session)):
    # Normalize
    From, To, Body = From.strip(), To.strip(), Body.strip()
    print(f"🔔 Incoming From={From}, To={To}, Body={Body!r}")

    # Channel detection
    is_whatsapp = From.startswith("whatsapp:")
    raw_from = From.split(":", 1)[1] if is_whatsapp else From
    customer_db = f"wa:{raw_from}" if is_whatsapp else From

    # Repository instances
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
            # close existing
            old = await conv_repo.get_active_conversation(
                contractor.id, target)
            if old:
                await conv_repo.close_conversation(old.id)
            # create new convo
            convo = await conv_repo.create_conversation(
                contractor_id=contractor.id, customer_phone=customer_db)
            intro = (
                f"Hi! I’m {contractor.name}’s assistant. "
                "To get started, please tell me the type of job you need.")
            reply_to = f"whatsapp:{target}" if is_whatsapp else target
            from_addr = WA_SANDBOX_NUMBER if is_whatsapp else TWILIO_NUMBER
            # log and send
            await msg_repo.create_message(sender=from_addr,
                                          receiver=reply_to,
                                          body=intro,
                                          direction="outbound",
                                          conversation_id=convo.id)
            send_sms(reply_to, intro)
        return Response(status_code=204)

    # 2) Customer-initiated flow
    res = await session.execute(select(Contractor))
    contractor = res.scalars().first()
    if not contractor:
        return Response(status_code=204)

    old_convo = await conv_repo.get_active_conversation(
        contractor.id, customer_db)

    # 3) CONFIRMING / COLLECTING_NOTES
    if old_convo and old_convo.status in ("CONFIRMING", "COLLECTING_NOTES"):
        # log inbound
        await msg_repo.create_message(sender=customer_db,
                                      receiver=TWILIO_NUMBER,
                                      body=Body,
                                      direction="inbound",
                                      conversation_id=old_convo.id)
        if old_convo.status == "CONFIRMING":
            if is_affirmative(Body):
                follow = "Thanks! Any other important info? When done reply 'No'."
                await msg_repo.create_message(sender=TWILIO_NUMBER,
                                              receiver=customer_db,
                                              body=follow,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_sms(customer_db, follow)
                old_convo.status = "COLLECTING_NOTES"
                await session.commit()
            else:
                full_msgs = await msg_repo.get_all_conversation_messages(
                    old_convo.id)
                history = "\n".join(
                    f"{ 'Customer' if d=='inbound' else 'AI'}: {b}"
                    for d, b in full_msgs)
                data = await extract_qualification_data(history)
                updated = await apply_correction_data(data, Body)
                await data_repo.upsert(conversation_id=old_convo.id,
                                       contractor_id=contractor.id,
                                       customer_phone=customer_db,
                                       data_dict=updated,
                                       qualified=is_qualified(updated),
                                       job_title=updated.get("job_type"))
                bullets = [
                    f"• {f.replace('_',' ').title()}: {updated[f]}"
                    for f in REQUIRED_FIELDS
                ]
                summary = "Got it! Here’s the updated info:\n" + "\n".join(
                    bullets) + "\nIs that correct?"
                await msg_repo.create_message(sender=TWILIO_NUMBER,
                                              receiver=customer_db,
                                              body=summary,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_sms(customer_db, summary)
            return Response(status_code=204)
        if old_convo.status == "COLLECTING_NOTES":
            if is_negative(Body):
                closing = "Great—thanks! I’ll pass this along to your contractor. ✅"
                await msg_repo.create_message(sender=TWILIO_NUMBER,
                                              receiver=customer_db,
                                              body=closing,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_sms(customer_db, closing)
                await conv_repo.close_conversation(old_convo.id)
            else:
                cd: ConversationData = await session.get(
                    ConversationData, old_convo.id)
                combined = (cd.data_json.get("notes", "") + "\n" +
                            Body).strip()
                cd.data_json["notes"] = combined
                cd.last_updated = datetime.utcnow()
                await session.commit()
                prompt = "Anything else to add? If not, reply 'No'."
                await msg_repo.create_message(sender=TWILIO_NUMBER,
                                              receiver=customer_db,
                                              body=prompt,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_sms(customer_db, prompt)
            return Response(status_code=204)

    # 4) Qualification loop / pivot detection
    recent = await msg_repo.get_recent_messages(customer=customer_db,
                                                contractor=TWILIO_NUMBER,
                                                limit=10)
    history = "\n".join(f"{ 'Customer' if d=='inbound' else 'AI'}: {b}"
                        for d, b in recent)
    cls = await classify_message(Body, history)
    print(f"🧠 Classification: {cls}")
    if cls == "UNSURE":
        prompt = "Is this about your previous job or a new one?"
        send_sms(customer_db, prompt)
        return Response(status_code=204)
    if cls == "NEW":
        if old_convo:
            await conv_repo.close_conversation(old_convo.id)
        convo = await conv_repo.create_conversation(contractor.id, customer_db)
        intro = f"Hi! I’m {contractor.name}’s assistant. To get started, please tell me the type of job you need."
        from_addr = WA_SANDBOX_NUMBER if is_whatsapp else TWILIO_NUMBER
        await msg_repo.create_message(sender=from_addr,
                                      receiver=customer_db,
                                      body=intro,
                                      direction="outbound",
                                      conversation_id=convo.id)
        send_sms(customer_db, intro)
        return Response(status_code=204)

    # 5) Continue qualification for CONTINUATION or resumed
    await msg_repo.create_message(sender=customer_db,
                                  receiver=TWILIO_NUMBER,
                                  body=Body,
                                  direction="inbound",
                                  conversation_id=convo.id)
    full_msgs = await msg_repo.get_all_conversation_messages(convo.id)
    history = "\n".join(f"{ 'Customer' if d=='inbound' else 'AI'}: {b}"
                        for d, b in full_msgs)
    data = await extract_qualification_data(history)
    await data_repo.upsert(conversation_id=convo.id,
                           contractor_id=contractor.id,
                           customer_phone=customer_db,
                           data_dict=data,
                           qualified=is_qualified(data),
                           job_title=data.get("job_type"))
    missing = [f for f in REQUIRED_FIELDS if not data[f]]
    if missing:
        if all(not data[f] for f in REQUIRED_FIELDS):
            ask = "Please provide your job type."
        else:
            nxt = missing[:2]
            lbls = [f.replace('_', ' ') for f in nxt]
            ask = (f"Please provide your {lbls[0]}." if len(lbls) == 1 else
                   f"Please provide your {lbls[0]} and {lbls[1]}.")
        await msg_repo.create_message(sender=TWILIO_NUMBER,
                                      receiver=customer_db,
                                      body=ask,
                                      direction="outbound",
                                      conversation_id=convo.id)
        send_sms(customer_db, ask)
        return Response(status_code=204)
    if convo.status == "QUALIFYING":
        bullets = [
            f"• {f.replace('_',' ').title()}: {data[f]}"
            for f in REQUIRED_FIELDS
        ]
        summary = "Here’s what I have so far:\n" + "\n".join(
            bullets) + "\nIs that correct?"
        await msg_repo.create_message(sender=TWILIO_NUMBER,
                                      receiver=customer_db,
                                      body=summary,
                                      direction="outbound",
                                      conversation_id=convo.id)
        send_sms(customer_db, summary)
        convo.status = "CONFIRMING"
        await session.commit()
        return Response(status_code=204)

    return Response(status_code=204)


@app.get("/pdf/{convo_id}")
def generate_pdf(convo_id: str):
    path = f"/tmp/{convo_id}.pdf"
    return FileResponse(path, media_type="application/pdf")


async def run_daily_digest():
    async with AsyncSessionLocal() as session:
        data_repo = ConversationDataRepo(session)
        all_leads = await data_repo.get_all()
    per: dict[int, list] = {}
    for lead in all_leads:
        per.setdefault(lead.contractor_id, []).append(lead)
    today = datetime.utcnow().strftime("%d/%m")
    for cid, leads in per.items():
        lines = [f"📊 TODAY'S LEADS ({today})"]
        complete = [l for l in leads if l.qualified]
        if complete:
            lines.append("✅ Complete:")
            for l in complete:
                d = l.data_json
                url = f"https://{os.environ.get('REPLIT_DOMAIN')}/pdf/{l.conversation_id}"
