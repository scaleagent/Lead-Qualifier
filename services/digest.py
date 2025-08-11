# services/digest.py - WITH ENHANCED LOGGING

import os
import asyncio
import logging
import hashlib
import hmac
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from repos.database import AsyncSessionLocal
from repos.contractor_repo import ContractorRepo
from repos.conversation_data_repo import ConversationDataRepo
from repos.conversation_repo import ConversationRepo
from repos.message_repo import MessageRepo
from repos.models import ConversationData

from utils.messaging import send_message

# Set up detailed logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_pdf_url(conversation_id: str) -> str:
    """Generate PDF URL with signed token for security"""
    PDF_SECRET_KEY = os.environ.get("PDF_SECRET_KEY",
                                    "change-this-in-production")
    base_url = os.environ.get("APP_BASE_URL", "https://your-app.herokuapp.com")

    logger.debug(
        f"Generating PDF URL for conversation {conversation_id[:8]}...")

    # Generate time-limited token
    expiry = int((datetime.utcnow() + timedelta(hours=24)).timestamp())
    message = f"{conversation_id}:{expiry}"
    signature = hmac.new(PDF_SECRET_KEY.encode(), message.encode(),
                         hashlib.sha256).hexdigest()
    token = f"{message}:{signature}"

    url = f"{base_url}/pdf/transcript?token={token}"
    logger.debug(f"Generated PDF URL: {url[:50]}... (truncated)")
    return url


def format_qualified_lead_sms(lead: ConversationData) -> str:
    """Format a qualified lead for SMS with all key details"""
    logger.debug(
        f"Formatting qualified lead SMS for conversation {lead.conversation_id[:8]}..."
    )

    data = lead.data_json

    # Extract and truncate fields to fit SMS limits
    job_type = (data.get('job_type', 'Unknown job') or 'Unknown job')[:30]
    address = (data.get('address', 'No address') or 'No address')[:40]
    urgency = (data.get('urgency', 'Not specified') or 'Not specified')[:20]
    access = (data.get('access', '') or '')[:30]

    logger.debug(
        f"  Lead data - Job: {job_type}, Address: {address}, Urgency: {urgency}"
    )

    # Generate PDF URL
    pdf_url = generate_pdf_url(lead.conversation_id)

    # Build SMS message
    parts = [f"🏗 {lead.job_title or job_type}", f"📍 {address}", f"⏰ {urgency}"]

    if access:
        parts.append(f"🔑 {access}")

    parts.append(f"📄 {pdf_url}")

    text = "\n".join(parts)

    # Ensure SMS doesn't exceed length limits
    if len(text) > 300:
        logger.warning(f"SMS too long ({len(text)} chars), truncating...")
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

    logger.debug(f"  Final SMS length: {len(text)} chars")
    return text


def format_ongoing_leads_sms(ongoing_leads: list[ConversationData]) -> str:
    """Format ongoing leads into a compact SMS"""
    logger.debug(
        f"Formatting ongoing leads SMS for {len(ongoing_leads)} leads")

    if len(ongoing_leads) <= 3:
        titles = []
        for lead in ongoing_leads:
            title = (lead.job_title or "Untitled")[:25]
            titles.append(title)
            logger.debug(f"  Including lead: {title}")

        text = f"🔄 Ongoing: {' • '.join(titles)}"
    else:
        text = f"🔄 {len(ongoing_leads)} ongoing leads in progress"
        logger.debug(f"  Too many leads to list individually")

    if len(text) > 160:
        text = f"🔄 {len(ongoing_leads)} ongoing leads"
        logger.warning(f"Ongoing SMS truncated to fit length limit")

    return text


async def run_daily_digest():
    """
    Generate and send daily digest of leads to each contractor at their configured hour.
    """
    logger.info("=" * 60)
    logger.info("STARTING DAILY DIGEST RUN")
    logger.info(f"Current UTC time: {datetime.utcnow()}")
    logger.info("=" * 60)

    try:
        async with AsyncSessionLocal() as session:
            contractor_repo = ContractorRepo(session)
            data_repo = ConversationDataRepo(session)
            conv_repo = ConversationRepo(session)
            msg_repo = MessageRepo(session)

            contractors = await contractor_repo.get_all()
            logger.info(
                f"Found {len(contractors)} total contractors in database")

            # Track statistics
            stats = {
                'contractors_processed': 0,
                'contractors_skipped': 0,
                'qualified_leads_sent': 0,
                'ongoing_leads_sent': 0,
                'errors': 0
            }

            for contractor in contractors:
                logger.info(
                    f"\n--- Processing contractor: {contractor.name} (ID: {contractor.id}) ---"
                )

                # Determine destination phone
                dest_phone = os.getenv("DIGEST_TEST_PHONE") or contractor.phone
                if os.getenv("DIGEST_TEST_PHONE"):
                    logger.info(
                        f"📱 TEST MODE: Using test phone {dest_phone} instead of {contractor.phone}"
                    )
                else:
                    logger.info(
                        f"📱 Using contractor's actual phone: {dest_phone}")

                # Load and log digest config
                config = contractor.digest_config or {}
                digest_hour = config.get('digest_hour', 18)
                tz_name = config.get('timezone', 'Europe/London')
                repeat_flag = config.get('repeat_until_takeover', True)

                logger.info(
                    f"  Config: Hour={digest_hour}, Timezone={tz_name}, Repeat={repeat_flag}"
                )

                # Check timezone and hour
                try:
                    now_tz = datetime.now(ZoneInfo(tz_name))
                    logger.info(
                        f"  Current time in {tz_name}: {now_tz.strftime('%H:%M')} (hour={now_tz.hour})"
                    )
                except Exception as e:
                    logger.error(f"  ❌ Invalid timezone '{tz_name}': {e}")
                    logger.warning(f"  Falling back to UTC")
                    now_tz = datetime.utcnow()

                # Check if it's the right hour
                if now_tz.hour != digest_hour:
                    logger.info(
                        f"  ⏰ SKIPPING: Current hour {now_tz.hour} != configured hour {digest_hour}"
                    )
                    stats['contractors_skipped'] += 1
                    continue

                logger.info(f"  ✅ Hour matches! Processing digest...")
                stats['contractors_processed'] += 1

                # Fetch qualified leads
                try:
                    qualified_leads = await data_repo.get_qualified_for_digest(
                        contractor.id, repeat_flag)
                    logger.info(
                        f"  📋 Found {len(qualified_leads)} qualified leads")

                    if qualified_leads:
                        for i, lead in enumerate(qualified_leads, 1):
                            logger.debug(
                                f"    Lead {i}: {lead.job_title} (Conv: {lead.conversation_id[:8]}...)"
                            )
                except Exception as e:
                    logger.error(f"  ❌ Error fetching qualified leads: {e}")
                    qualified_leads = []

                # Fetch ongoing leads
                try:
                    collecting_convos = await conv_repo.get_collecting_notes_for_contractor(
                        contractor.id)
                    logger.info(
                        f"  🔄 Found {len(collecting_convos)} conversations in COLLECTING_NOTES"
                    )

                    ongoing_leads = []
                    for convo in collecting_convos:
                        cd = await session.get(ConversationData, convo.id)
                        if cd and not cd.opt_out_of_digest:
                            ongoing_leads.append(cd)
                            logger.debug(
                                f"    Including ongoing: {cd.job_title}")
                        elif cd and cd.opt_out_of_digest:
                            logger.debug(
                                f"    Skipping opted-out: {cd.job_title}")

                    logger.info(
                        f"  📝 {len(ongoing_leads)} ongoing leads after opt-out filter"
                    )
                except Exception as e:
                    logger.error(f"  ❌ Error fetching ongoing leads: {e}")
                    ongoing_leads = []

                # Check if there's anything to send
                if not qualified_leads and not ongoing_leads:
                    logger.info(
                        f"  📭 No leads to send for contractor {contractor.id}")
                    continue

                now_utc = datetime.utcnow()

                # Send qualified leads
                for lead in qualified_leads:
                    try:
                        text = format_qualified_lead_sms(lead)
                        logger.info(
                            f"  📤 Sending qualified lead: {lead.job_title[:30]}..."
                        )
                        logger.debug(
                            f"    SMS content ({len(text)} chars): {text[:100]}..."
                        )

                        send_message(dest_phone, text, is_whatsapp=False)

                        # Mark as sent
                        await data_repo.mark_digest_sent(
                            lead.conversation_id, now_utc)
                        logger.info(
                            f"    ✅ Sent and marked digest for conversation {lead.conversation_id[:8]}..."
                        )
                        stats['qualified_leads_sent'] += 1

                    except Exception as e:
                        logger.error(
                            f"    ❌ Failed to send lead {lead.conversation_id[:8]}: {e}"
                        )
                        stats['errors'] += 1

                # Send ongoing leads summary
                if ongoing_leads:
                    try:
                        text = format_ongoing_leads_sms(ongoing_leads)
                        logger.info(f"  📤 Sending ongoing leads summary...")
                        logger.debug(
                            f"    SMS content ({len(text)} chars): {text}")

                        send_message(dest_phone, text, is_whatsapp=False)
                        logger.info(f"    ✅ Sent ongoing leads summary")
                        stats['ongoing_leads_sent'] += 1

                    except Exception as e:
                        logger.error(
                            f"    ❌ Failed to send ongoing leads: {e}")
                        stats['errors'] += 1

            # Log final statistics
            logger.info("\n" + "=" * 60)
            logger.info("DAILY DIGEST RUN COMPLETED")
            logger.info(f"Statistics:")
            logger.info(
                f"  Contractors processed: {stats['contractors_processed']}")
            logger.info(
                f"  Contractors skipped: {stats['contractors_skipped']}")
            logger.info(
                f"  Qualified leads sent: {stats['qualified_leads_sent']}")
            logger.info(
                f"  Ongoing summaries sent: {stats['ongoing_leads_sent']}")
            logger.info(f"  Errors encountered: {stats['errors']}")
            logger.info("=" * 60)

    except Exception as e:
        logger.error(f"FATAL ERROR in daily digest: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # For manual testing
    logger.info("Running daily digest manually...")
    asyncio.run(run_daily_digest())
