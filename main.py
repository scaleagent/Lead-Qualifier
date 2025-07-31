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

# === Qualification Schema ===
REQUIRED_FIELDS = ["job_type", "property_type", "urgency", "address", "access"]

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
    """
    Strictly returns exactly one of: NEW, CONTINUATION, UNSURE
    """
    system = (
        "You are an AI assistant for a trades business. Classify the incoming SMS as exactly one of:\n"
        "NEW: user is requesting a completely new job\n"
        "CONTINUATION: user is giving more info on an existing, in-progress job\n"
        "UNSURE: unclear whether it's new or a continuation\n"
        "If they mention a new location, job type, or pivot, return NEW.\n"
        "Respond with exactly one word: NEW, CONTINUATION, or UNSURE.")
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system
            },
            {
                "role":
                "user",
                "content":
                f"Message:\n{message_text}\n\nHistory:\n{history_string}"
            },
        ],
        temperature=0,
        max_tokens=10,
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
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Conversation:\n{history_string}"
            },
        ],
        temperature=0,
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
    system_prompt = (
        "You are a JSON assistant. Given existing job data and a user's correction, "
        "return the updated JSON with the same six keys only.")
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role":
                "user",
                "content":
                f"Existing data: {json.dumps(current)}\nCorrection: {correction}\nRespond ONLY with the full updated JSON."
            },
        ],
        temperature=0,
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


def is_negative(text: str) -> bool:
    return bool(
        re.match(r"^(no|nope|that'?s (all|everything)|all done)\b",
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
    From, To, Body = From.strip(), To.strip(), Body.strip()
    print(f"🔔 Incoming SMS From={From}, To={To}, Body={Body!r}")

    # Repos
    contractor_repo = ContractorRepo(session)
    conv_repo = ConversationRepo(session)
    msg_repo = MessageRepo(session)
    data_repo = ConversationDataRepo(session)

    # 1) Contractor→AI “reach out”
    contractor = await contractor_repo.get_by_phone(From)
    if contractor:
        m = re.match(r'^\s*reach out to (\+44\d{9,})\s*$', Body, re.IGNORECASE)
        if m:
            customer = m.group(1)
            # close any stray open convos for that customer
            old = await conv_repo.get_active_conversation(
                contractor.id, customer)
            if old:
                await conv_repo.close_conversation(old.id)
            convo = await conv_repo.create_conversation(
                contractor_id=contractor.id, customer_phone=customer)
            intro = f"Hi! I’m {contractor.name}’s assistant. To get started, please tell me the type of job you need."
            # Log as AI→customer under TWILIO_NUMBER
            await msg_repo.create_message(sender=TWILIO_NUMBER,
                                          receiver=customer,
                                          body=intro,
                                          direction="outbound",
                                          conversation_id=convo.id)
            send_sms(customer, intro)
        return Response(status_code=204)

    # 2) Customer→AI flow
    # identify your contractor (single-Twilio setup)
    result = await session.execute(select(Contractor))
    contractor = result.scalars().first()
    if not contractor:
        print("⚠️ No contractor found; dropping SMS.")
        return Response(status_code=204)

    # fetch any active convo for this customer
    old_convo = await conv_repo.get_active_conversation(contractor.id, From)

    # 3) CONFIRMING or COLLECTING_NOTES priorities
    if old_convo and old_convo.status in ("CONFIRMING", "COLLECTING_NOTES"):
        # log the inbound
        await msg_repo.create_message(sender=From,
                                      receiver=TWILIO_NUMBER,
                                      body=Body,
                                      direction="inbound",
                                      conversation_id=old_convo.id)

        # CONFIRMING: accept or correct
        if old_convo.status == "CONFIRMING":
            if is_affirmative(Body):
                follow = (
                    "Thanks! If there’s any other important info—parking, pets, special access—"
                    "just reply here. When you’re done, reply “No”.")
                await msg_repo.create_message(sender=TWILIO_NUMBER,
                                              receiver=From,
                                              body=follow,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_sms(From, follow)
                old_convo.status = "COLLECTING_NOTES"
                await session.commit()
            else:
                # correction branch
                full = await msg_repo.get_all_conversation_messages(
                    old_convo.id)
                hist = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                                 for d, b in full)
                data = await extract_qualification_data(hist)
                updated = await apply_correction_data(data, Body)
                await data_repo.upsert(old_convo.id, contractor.id, From,
                                       updated, is_qualified(updated),
                                       updated.get("job_type"))
                bullets = [
                    f"• {f.replace('_',' ').title()}: {updated[f]}"
                    for f in REQUIRED_FIELDS
                ]
                summary = "Got it! Here’s the updated info:\n" + "\n".join(
                    bullets) + "\nIs that correct?"
                await msg_repo.create_message(sender=TWILIO_NUMBER,
                                              receiver=From,
                                              body=summary,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_sms(From, summary)
            return Response(status_code=204)

        # COLLECTING_NOTES: append until user says “No”
        if old_convo.status == "COLLECTING_NOTES":
            if is_negative(Body):
                closing = "Great—thanks! I’ll pass this along to your contractor. ✅"
                await msg_repo.create_message(sender=TWILIO_NUMBER,
                                              receiver=From,
                                              body=closing,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_sms(From, closing)
                await conv_repo.close_conversation(old_convo.id)
            else:
                cd: ConversationData = await session.get(
                    ConversationData, old_convo.id)
                combined = (cd.data_json.get("notes", "") + "\n" +
                            Body).strip()
                cd.data_json["notes"] = combined
                cd.last_updated = datetime.utcnow()
                await session.commit()
                prompt = "Anything else to add? If not, reply “No”."
                await msg_repo.create_message(sender=TWILIO_NUMBER,
                                              receiver=From,
                                              body=prompt,
                                              direction="outbound",
                                              conversation_id=old_convo.id)
                send_sms(From, prompt)
            return Response(status_code=204)

    # 4) QUALIFYING: assemble last 10 between customer↔AI
    recent = await msg_repo.get_recent_messages(customer=From,
                                                contractor=TWILIO_NUMBER,
                                                limit=10)
    history = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                        for d, b in recent)
    cls = await classify_message(Body, history)
    print(f"🧠 Classification: {cls}")

    # UNSURE → clarify with context
    if cls == "UNSURE":
        job = ""
        if old_convo:
            cd: ConversationData = await session.get(ConversationData,
                                                     old_convo.id)
            job = cd.job_title or cd.data_json.get("job_type", "")
        prompt = f"Is this about your previous “{job}” job or a new one?" if job else "Is this about your previous job or a new one?"
        await msg_repo.create_message(sender=TWILIO_NUMBER,
                                      receiver=From,
                                      body=prompt,
                                      direction="outbound",
                                      conversation_id=None)
        send_sms(From, prompt)
        return Response(status_code=204)

    # NEW → close any active and start fresh
    if cls == "NEW":
        if old_convo:
            await conv_repo.close_conversation(old_convo.id)
        convo = await conv_repo.create_conversation(
            contractor_id=contractor.id, customer_phone=From)
        intro = f"Hi! I’m {contractor.name}’s assistant. To get started, please tell me the type of job you need."
        await msg_repo.create_message(sender=TWILIO_NUMBER,
                                      receiver=From,
                                      body=intro,
                                      direction="outbound",
                                      conversation_id=convo.id)
        send_sms(From, intro)
        return Response(status_code=204)

    # CONTINUATION → fall back into existing
    if cls == "CONTINUATION":
        if not old_convo:
            convo = await conv_repo.create_conversation(
                contractor_id=contractor.id, customer_phone=From)
            intro = f"Hi! I’m {contractor.name}’s assistant. To get started, please tell me the type of job you need."
            await msg_repo.create_message(sender=TWILIO_NUMBER,
                                          receiver=From,
                                          body=intro,
                                          direction="outbound",
                                          conversation_id=convo.id)
            send_sms(From, intro)
            return Response(status_code=204)
        convo = old_convo

    # 5) Continue qualification loop
    await msg_repo.create_message(sender=From,
                                  receiver=TWILIO_NUMBER,
                                  body=Body,
                                  direction="inbound",
                                  conversation_id=convo.id)
    full = await msg_repo.get_all_conversation_messages(convo.id)
    history = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}"
                        for d, b in full)
    data = await extract_qualification_data(history)
    await data_repo.upsert(convo.id, contractor.id, From, data,
                           is_qualified(data), data.get("job_type"))

    missing = [k for k in REQUIRED_FIELDS if not data[k]]
    if missing:
        ask = "Please provide your job type." if all(
            data[k] == "" for k in REQUIRED_FIELDS) else (
                f"Please provide your {missing[0].replace('_',' ')}" +
                ("" if len(missing) == 1 else
                 f" and {missing[1].replace('_',' ')}"))
        await msg_repo.create_message(sender=TWILIO_NUMBER,
                                      receiver=From,
                                      body=ask,
                                      direction="outbound",
                                      conversation_id=convo.id)
        send_sms(From, ask)
        return Response(status_code=204)

    if convo.status == "QUALIFYING":
        bullets = [
            f"• {f.replace('_',' ').title()}: {data[f]}"
            for f in REQUIRED_FIELDS
        ]
        summary = "Here’s what I have so far:\n" + "\n".join(
            bullets) + "\nIs that correct?"
        await msg_repo.create_message(sender=TWILIO_NUMBER,
                                      receiver=From,
                                      body=summary,
                                      direction="outbound",
                                      conversation_id=convo.id)
        send_sms(From, summary)
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
        comp = [l for l in leads if l.qualified]
        if comp:
            lines.append("✅ Complete:")
            for l in comp:
                d = l.data_json
                url = f"https://{os.environ.get('REPLIT_DOMAIN')}/pdf/{l.conversation_id}"
                lines.append(
                    f"- {d.get('job_type','')} | {d.get('property_type','')} | {d.get('urgency','')} | {d.get('address','')}\n  View: {url}"
                )
        inc = [l for l in leads if not l.qualified]
        if inc:
            lines.append("⏸️ Incomplete:")
            for l in inc:
                d = l.data_json
                miss = [k for k in REQUIRED_FIELDS if not d.get(k)]
                last = l.last_updated.strftime("%d/%m %H:%M")
                lines.append(
                    f"- {d.get('job_type','')} ({l.customer_phone}), last update {last}\n  Missing: {', '.join(miss)}"
                )
        body = "\n".join(lines)
        send_sms(TWILIO_NUMBER, body)


scheduler = BackgroundScheduler()
scheduler.add_job(lambda: asyncio.create_task(run_daily_digest()),
                  'cron',
                  hour=18)
scheduler.start()
