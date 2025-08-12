
# modules/digest/digest_service.py - Main digest service

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from repos.database import AsyncSessionLocal
from repos.contractor_repo import ContractorRepo
from repos.conversation_data_repo import ConversationDataRepo
from repos.conversation_repo import ConversationRepo
from repos.message_repo import MessageRepo
from repos.models import ConversationData
from utils.messaging import send_message

from .config import DIGEST_TEST_PHONE, DEFAULT_DIGEST_HOUR, DEFAULT_TIMEZONE, DEFAULT_REPEAT_UNTIL_TAKEOVER
from .message_formatter import MessageFormatter

logger = logging.getLogger(__name__)


class DigestService:
    """Self-contained daily digest service"""
    
    def __init__(self):
        self.formatter = MessageFormatter()
    
    async def run_daily_digest(self, force: bool = False, only_contractor_id: int | None = None):
        """
        Generate and send daily digest of leads to each contractor.
        If force=True, skip the hour check.
        If only_contractor_id is set, process only that contractor.
        """
        logger.info("=" * 60)
        logger.info("STARTING DAILY DIGEST RUN")
        logger.info(f"Current UTC time: {datetime.utcnow()}")
        logger.info(f"Overrides -> force={force}, only_contractor_id={only_contractor_id}")
        logger.info("=" * 60)

        try:
            async with AsyncSessionLocal() as session:
                contractor_repo = ContractorRepo(session)
                data_repo = ConversationDataRepo(session)
                conv_repo = ConversationRepo(session)
                msg_repo = MessageRepo(session)

                # Load only one contractor if requested
                if only_contractor_id is not None:
                    one = await contractor_repo.get_by_id(only_contractor_id)
                    contractors = [one] if one else []
                    if not contractors:
                        logger.warning(f"No contractor found with id={only_contractor_id}. Nothing to do.")
                else:
                    contractors = await contractor_repo.get_all()

                logger.info(f"Found {len(contractors)} contractor(s) to process")

                # Track statistics
                stats = {
                    'contractors_processed': 0,
                    'contractors_skipped': 0,
                    'qualified_leads_sent': 0,
                    'ongoing_leads_sent': 0,
                    'errors': 0
                }

                for contractor in contractors:
                    await self._process_contractor(contractor, data_repo, conv_repo, session, force, stats)

                # Final stats
                logger.info("\n" + "=" * 60)
                logger.info("DAILY DIGEST RUN COMPLETED")
                logger.info("Statistics:")
                logger.info(f"  Contractors processed: {stats['contractors_processed']}")
                logger.info(f"  Contractors skipped: {stats['contractors_skipped']}")
                logger.info(f"  Qualified leads sent: {stats['qualified_leads_sent']}")
                logger.info(f"  Ongoing summaries sent: {stats['ongoing_leads_sent']}")
                logger.info(f"  Errors encountered: {stats['errors']}")
                logger.info("=" * 60)

        except Exception as e:
            logger.error(f"FATAL ERROR in daily digest: {e}", exc_info=True)
            raise

    async def _process_contractor(self, contractor, data_repo, conv_repo, session, force, stats):
        """Process digest for a single contractor"""
        logger.info(f"\n--- Processing contractor: {contractor.name} (ID: {contractor.id}) ---")

        # Destination phone (test override respected)
        dest_phone = DIGEST_TEST_PHONE or contractor.phone
        if DIGEST_TEST_PHONE:
            logger.info(f"📱 TEST MODE: Using test phone {dest_phone} instead of {contractor.phone}")
        else:
            logger.info(f"📱 Using contractor's actual phone: {dest_phone}")

        # Digest config
        config = contractor.digest_config or {}
        digest_hour = config.get('digest_hour', DEFAULT_DIGEST_HOUR)
        tz_name = config.get('timezone', DEFAULT_TIMEZONE)
        repeat_flag = config.get('repeat_until_takeover', DEFAULT_REPEAT_UNTIL_TAKEOVER)

        logger.info(f"  Config: Hour={digest_hour}, Timezone={tz_name}, Repeat={repeat_flag}")

        # Local time for the contractor
        try:
            now_tz = datetime.now(ZoneInfo(tz_name))
            logger.info(f"  Current time in {tz_name}: {now_tz.strftime('%H:%M')} (hour={now_tz.hour})")
        except Exception as e:
            logger.error(f"  ❌ Invalid timezone '{tz_name}': {e}")
            logger.warning(f"  Falling back to UTC")
            now_tz = datetime.utcnow()

        # Hour gating (bypass if force=True)
        if not force:
            if now_tz.hour != digest_hour:
                logger.info(f"  ⏰ SKIPPING: Current hour {now_tz.hour} != configured hour {digest_hour}")
                stats['contractors_skipped'] += 1
                return
            else:
                logger.info(f"  ✅ Hour matches! Processing digest...")
        else:
            logger.info("  ⚡ FORCE OVERRIDE ENABLED: bypassing hour check")

        stats['contractors_processed'] += 1

        # Fetch qualified leads
        try:
            qualified_leads = await data_repo.get_qualified_for_digest(contractor.id, repeat_flag)
            logger.info(f"  📋 Found {len(qualified_leads)} qualified leads")
            if qualified_leads:
                for i, lead in enumerate(qualified_leads, 1):
                    logger.debug(f"    Lead {i}: {lead.job_title} (Conv: {lead.conversation_id[:8]}...)")
        except Exception as e:
            logger.error(f"  ❌ Error fetching qualified leads: {e}")
            qualified_leads = []

        # Fetch ongoing (collecting notes) leads
        try:
            collecting_convos = await conv_repo.get_collecting_notes_for_contractor(contractor.id)
            logger.info(f"  🔄 Found {len(collecting_convos)} conversations in COLLECTING_NOTES")

            ongoing_leads = []
            for convo in collecting_convos:
                cd = await session.get(ConversationData, convo.id)
                if cd and not cd.opt_out_of_digest:
                    ongoing_leads.append(cd)
                    logger.debug(f"    Including ongoing: {cd.job_title}")
                elif cd and cd.opt_out_of_digest:
                    logger.debug(f"    Skipping opted-out: {cd.job_title}")

            logger.info(f"  📝 {len(ongoing_leads)} ongoing leads after opt-out filter")
        except Exception as e:
            logger.error(f"  ❌ Error fetching ongoing leads: {e}")
            ongoing_leads = []

        # Nothing to send?
        if not qualified_leads and not ongoing_leads:
            logger.info(f"  📭 No leads to send for contractor {contractor.id}")
            return

        now_utc = datetime.utcnow()

        # Send qualified leads
        for lead in qualified_leads:
            try:
                text = self.formatter.format_qualified_lead_sms(lead)
                logger.info(f"  📤 Sending qualified lead: {lead.job_title[:30]}...")
                logger.debug(f"    SMS content ({len(text)} chars): {text[:100]}...")
                send_message(dest_phone, text, is_whatsapp=False)

                # Mark as sent
                await data_repo.mark_digest_sent(lead.conversation_id, now_utc)
                logger.info(f"    ✅ Sent and marked digest for conversation {lead.conversation_id[:8]}...")
                stats['qualified_leads_sent'] += 1
            except Exception as e:
                logger.error(f"    ❌ Failed to send lead {lead.conversation_id[:8]}: {e}")
                stats['errors'] += 1

        # Send ongoing summary
        if ongoing_leads:
            try:
                text = self.formatter.format_ongoing_leads_sms(ongoing_leads)
                logger.info(f"  📤 Sending ongoing leads summary...")
                logger.debug(f"    SMS content ({len(text)} chars): {text}")
                send_message(dest_phone, text, is_whatsapp=False)
                logger.info(f"    ✅ Sent ongoing leads summary")
                stats['ongoing_leads_sent'] += 1
            except Exception as e:
                logger.error(f"    ❌ Failed to send ongoing leads: {e}")
                stats['errors'] += 1


# Convenience function for backward compatibility
async def run_daily_digest(force: bool = False, only_contractor_id: int | None = None):
    """Convenience function to run digest service"""
    service = DigestService()
    await service.run_daily_digest(force, only_contractor_id)
