import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from repos.database import AsyncSessionLocal
from repos.contractor_repo import ContractorRepo
from repos.conversation_data_repo import ConversationDataRepo
from repos.conversation_repo import ConversationRepo
from repos.message_repo import MessageRepo
from repos.models import ConversationData

# Import your send_message helper from main or a utils module
from utils.messaging import send_message

logger = logging.getLogger(__name__)


async def run_daily_digest():
    """
    Generate and send daily digest of leads to each contractor at their configured hour.
    """
    async with AsyncSessionLocal() as session:
        contractor_repo = ContractorRepo(session)
        data_repo = ConversationDataRepo(session)
        conv_repo = ConversationRepo(session)
        msg_repo = MessageRepo(session)

        contractors = await contractor_repo.get_all()
        logger.info(f"Found {len(contractors)} contractors for daily digest")

        for contractor in contractors:

            #Set destination phone as mine for testing
            dest_phone = os.getenv("DIGEST_TEST_PHONE") or contractor.phone

            # Load digest config
            config = contractor.digest_config or {}
            digest_hour = config.get('digest_hour', 18)
            tz_name = config.get('timezone', 'Europe/London')

            # Compute current hour in contractor's timezone
            try:
                now_tz = datetime.now(ZoneInfo(tz_name))
            except Exception:
                logger.warning(
                    f"Invalid timezone '{tz_name}' for contractor {contractor.id}, defaulting to UTC"
                )
                now_tz = datetime.utcnow()

            # Only run when current hour matches configured hour
            if now_tz.hour != digest_hour:
                logger.debug(
                    f"Skipping contractor {contractor.id}, current hour {now_tz.hour} != digest_hour {digest_hour}"
                )
                continue

            logger.info(
                f"Processing digest for contractor_id={contractor.id} at {now_tz}"
            )

            # Fetch leads eligible for digest
            repeat_flag = config.get('repeat_until_takeover', True)
            qualified_leads = await data_repo.get_qualified_for_digest(
                repeat_flag)
            logger.info(
                f"  Found {len(qualified_leads)} qualified leads for digest")

            # Fetch ongoing leads in COLLECTING_NOTES
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
            # Send a separate SMS for each qualified lead
            for lead in qualified_leads:
                pdf_url = f"https://your-domain.com/pdf/{lead.conversation_id}"
                text = (f"Closed Lead: {lead.job_title}\n"
                        f"Details: see details in PDF\n"
                        f"PDF Transcript: {pdf_url}")
                #send_message(contractor.phone, text, is_whatsapp=False) # un comment when out of testing
                send_message(dest_phone, text, is_whatsapp=False)

                await data_repo.mark_digest_sent(lead.conversation_id, now_utc)
                logger.debug(
                    f"    Sent qualified lead digest for convo {lead.conversation_id}"
                )

            # Send one SMS for all ongoing leads
            if ongoing_leads:
                titles = ' • '.join([l.job_title for l in ongoing_leads])
                text = f"Ongoing Leads: • {titles}"
                #send_message(contractor.phone, text, is_whatsapp=False) # un comment when out of testing
                send_message(dest_phone, text, is_whatsapp=False)

                logger.debug(
                    f"    Sent ongoing leads digest for contractor {contractor.id}"
                )

    logger.info("Daily digest run completed")


if __name__ == '__main__':
    asyncio.run(run_daily_digest())
