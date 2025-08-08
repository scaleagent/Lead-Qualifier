import os
import asyncio
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hmac
import hashlib
from urllib.parse import quote

from repos.database import AsyncSessionLocal
from repos.contractor_repo import ContractorRepo
from repos.conversation_data_repo import ConversationDataRepo
from repos.conversation_repo import ConversationRepo
from repos.message_repo import MessageRepo
from repos.models import ConversationData

from utils.messaging import send_message

logger = logging.getLogger(__name__)

# Shared PDF signing config (must match main.py)
PDF_SECRET_KEY = os.environ.get("PDF_SECRET_KEY", "change-this-in-production")
APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://your-app.herokuapp.com")


def _generate_pdf_token(conversation_id: str,
                        expires_in_hours: int = 24) -> str:
    expiry = int(
        (datetime.utcnow() + timedelta(hours=expires_in_hours)).timestamp())
    message = f"{conversation_id}:{expiry}"
    signature = hmac.new(PDF_SECRET_KEY.encode(), message.encode(),
                         hashlib.sha256).hexdigest()
    return f"{message}:{signature}"


def generate_pdf_url(conversation_id: str) -> str:
    token = _generate_pdf_token(conversation_id)
    return f"{APP_BASE_URL}/pdf/transcript?token={quote(token)}"


def format_qualified_lead_sms(lead: ConversationData) -> str:
    """Format a qualified lead for SMS with all key details (kept short)"""
    data = lead.data_json or {}

    job_type = (data.get('job_type') or 'Unknown job')[:30]
    address = (data.get('address') or 'No address')[:40]
    urgency = (data.get('urgency') or 'Not specified')[:20]
    access = (data.get('access') or '')[:30]

    pdf_url = generate_pdf_url(lead.conversation_id)

    parts = [f"🏗 {lead.job_title or job_type}", f"📍 {address}", f"⏰ {urgency}"]
    if access:
        parts.append(f"🔑 {access}")
    parts.append(f"📄 {pdf_url}")

    text = "\n".join(parts)

    # Keep it under ~300 chars
    if len(text) > 300:
        address = address[:20] + "..."
        if access:
            access = access[:15] + "..."
        parts = [
            f"🏗 {lead.job_title or job_type}", f"📍 {address}", f"⏰ {urgency}"
        ]
        if access:
            parts.append(f"🔑 {access}")
        parts.append(f"📄 {pdf_url}")
        text = "\n".join(parts)

    return text


def format_ongoing_leads_sms(ongoing_leads: list[ConversationData]) -> str:
    """Format ongoing leads into a compact SMS"""
    if len(ongoing_leads) <= 3:
        titles = []
        for lead in ongoing_leads:
            title = (lead.job_title or "Untitled")[:25]
            titles.append(title)
        text = f"🔄 Ongoing: {' • '.join(titles)}"
    else:
        text = f"🔄 {len(ongoing_leads)} ongoing leads in progress"

    if len(text) > 160:
        text = f"🔄 {len(ongoing_leads)} ongoing leads"

    return text


async def run_daily_digest():
    """
    Scheduled: Generate and send daily digest of leads for each contractor
    when their local hour == digest_hour.
    """
    async with AsyncSessionLocal() as session:
        contractor_repo = ContractorRepo(session)
        data_repo = ConversationDataRepo(session)
        conv_repo = ConversationRepo(session)
        msg_repo = MessageRepo(session)

        contractors = await contractor_repo.get_all()
        logger.info(f"Found {len(contractors)} contractors for daily digest")

        for contractor in contractors:
            dest_phone = os.getenv("DIGEST_TEST_PHONE") or contractor.phone

            config = contractor.digest_config or {}
            digest_hour = config.get('digest_hour', 18)
            tz_name = config.get('timezone', 'Europe/London')

            # What time is it for this contractor?
            try:
                now_tz = datetime.now(ZoneInfo(tz_name))
            except Exception:
                logger.warning(
                    f"Invalid timezone '{tz_name}' for contractor {contractor.id}, defaulting to UTC"
                )
                now_tz = datetime.utcnow()

            # Only act at the configured hour
            if now_tz.hour != digest_hour:
                logger.debug(
                    f"Skipping contractor {contractor.id}, current hour {now_tz.hour} != digest_hour {digest_hour}"
                )
                continue

            logger.info(
                f"Processing digest for contractor_id={contractor.id} at {now_tz}"
            )

            repeat_flag = config.get('repeat_until_takeover', True)

            # Qualified & eligible across all contractors, then filter to this one
            qualified_leads = await data_repo.get_qualified_for_digest(
                repeat_flag)
            qualified_leads = [
                l for l in qualified_leads if l.contractor_id == contractor.id
            ]
            logger.info(
                f"  Found {len(qualified_leads)} qualified leads for digest")

            # Ongoing (collecting notes) for this contractor
            collecting_convos = await conv_repo.get_collecting_notes_for_contractor(
                contractor.id)
            ongoing_leads = []
            for convo in collecting_convos:
                cd = await session.get(ConversationData, convo.id)
                if cd and not cd.opt_out_of_digest:
                    ongoing_leads.append(cd)
            logger.info(
                f"  Found {len(ongoing_leads)} ongoing leads for digest")

            now_utc = datetime.utcnow()

            # Send each qualified lead as its own SMS
            for lead in qualified_leads:
                text = format_qualified_lead_sms(lead)
                send_message(dest_phone, text, is_whatsapp=False)
                await data_repo.mark_digest_sent(lead.conversation_id, now_utc)
                logger.debug(
                    f"    Sent qualified lead digest for convo {lead.conversation_id}"
                )

            # Send one SMS for all ongoing leads
            if ongoing_leads:
                text = format_ongoing_leads_sms(ongoing_leads)
                send_message(dest_phone, text, is_whatsapp=False)
                logger.debug(
                    f"    Sent ongoing leads digest for contractor {contractor.id}"
                )

    logger.info("Daily digest run completed")


# === NEW: Force send for a single contractor, on demand ===
async def run_digest_for_contractor(contractor_id: int,
                                    ignore_time: bool = True) -> dict:
    """
    On-demand trigger. Sends digest for one contractor.
    - If ignore_time=True (default), bypass the digest_hour check and send now.
    - Respects DIGEST_TEST_PHONE fallback routing.
    Returns a small summary dict with counts.
    """
    async with AsyncSessionLocal() as session:
        contractor_repo = ContractorRepo(session)
        data_repo = ConversationDataRepo(session)
        conv_repo = ConversationRepo(session)

        contractor = await contractor_repo.get_by_id(contractor_id)
        if not contractor:
            logger.warning(f"No contractor found with id={contractor_id}")
            return {
                "contractor_id": contractor_id,
                "sent_qualified": 0,
                "sent_ongoing": 0,
                "status": "not_found"
            }

        dest_phone = os.getenv("DIGEST_TEST_PHONE") or contractor.phone
        config = contractor.digest_config or {}
        digest_hour = config.get('digest_hour', 18)
        tz_name = config.get('timezone', 'Europe/London')
        repeat_flag = config.get('repeat_until_takeover', True)

        # Respect time if requested
        if not ignore_time:
            try:
                now_tz = datetime.now(ZoneInfo(tz_name))
            except Exception:
                logger.warning(
                    f"Invalid timezone '{tz_name}' for contractor {contractor.id}, defaulting to UTC"
                )
                now_tz = datetime.utcnow()
            if now_tz.hour != digest_hour:
                logger.info(
                    f"[force-digest] Hour mismatch for contractor {contractor.id}: now={now_tz.hour} vs digest_hour={digest_hour}. Nothing sent."
                )
                return {
                    "contractor_id": contractor_id,
                    "sent_qualified": 0,
                    "sent_ongoing": 0,
                    "status": "wrong_hour"
                }

        # Eligible qualified leads, filtered to this contractor
        qualified_leads = await data_repo.get_qualified_for_digest(repeat_flag)
        qualified_leads = [
            l for l in qualified_leads
            if l.contractor_id == contractor.id and not l.opt_out_of_digest
        ]

        # Ongoing
        collecting_convos = await conv_repo.get_collecting_notes_for_contractor(
            contractor.id)
        ongoing_leads = []
        for convo in collecting_convos:
            cd = await session.get(ConversationData, convo.id)
            if cd and not cd.opt_out_of_digest:
                ongoing_leads.append(cd)

        # Send
        sent_qualified = 0
        now_utc = datetime.utcnow()
        for lead in qualified_leads:
            text = format_qualified_lead_sms(lead)
            send_message(dest_phone, text, is_whatsapp=False)
            await data_repo.mark_digest_sent(lead.conversation_id, now_utc)
            sent_qualified += 1

        sent_ongoing = 0
        if ongoing_leads:
            text = format_ongoing_leads_sms(ongoing_leads)
            send_message(dest_phone, text, is_whatsapp=False)
            sent_ongoing = len(ongoing_leads)

        result = {
            "contractor_id": contractor_id,
            "sent_qualified": sent_qualified,
            "sent_ongoing": sent_ongoing,
            "status": "ok",
        }
        logger.info(f"[force-digest] {result}")
        return result


if __name__ == '__main__':
    asyncio.run(run_daily_digest())
