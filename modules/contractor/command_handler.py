
import re
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from repos.conversation_data_repo import ConversationDataRepo
from repos.conversation_repo import ConversationRepo
from repos.message_repo import MessageRepo
from utils.messaging import send_message

logger = logging.getLogger(__name__)


class CommandHandler:
    """Handles contractor-specific commands like takeover and reach out"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.data_repo = ConversationDataRepo(session)
        self.conv_repo = ConversationRepo(session)
        self.msg_repo = MessageRepo(session)
    
    def is_contractor_command(self, message: str) -> bool:
        """Check if message contains a contractor command"""
        message_lower = message.lower().strip()
        
        # Takeover patterns
        takeover_patterns = [
            r"stop\s+(?:daily\s+)?digest\s+for\s+.+",
            r"takeover\s+.+",
            r"take\s+over\s+.+",
            r"claim\s+.+"
        ]
        
        # Reach out patterns
        reach_out_patterns = [
            r"reach\s+out\s+to\s+\+?\d+",
            r"contact\s+\+?\d+"
        ]
        
        all_patterns = takeover_patterns + reach_out_patterns
        
        for pattern in all_patterns:
            if re.match(pattern, message_lower):
                return True
        
        return False
    
    async def handle_takeover_command(self, message: str, contractor) -> str | None:
        """Handle contractor takeover commands"""
        logger.info(f"🎯 CHECKING FOR TAKEOVER COMMAND")
        logger.info(f"  Contractor: {contractor.name} (ID: {contractor.id})")
        logger.info(f"  Message: '{message}'")

        patterns = [
            r"stop\s+(?:daily\s+)?digest\s+for\s+(.+)",
            r"takeover\s+(.+)",
            r"take\s+over\s+(.+)",
            r"claim\s+(.+)"
        ]

        for pattern_idx, pattern in enumerate(patterns):
            logger.debug(f"  Testing pattern {pattern_idx + 1}: {pattern}")
            match = re.match(pattern, message.lower().strip())

            if match:
                job_query = match.group(1).strip()
                logger.info(f"  ✅ TAKEOVER COMMAND DETECTED")
                logger.info(f"  Job query: '{job_query}'")

                # Find matching conversation
                lead = await self.data_repo.find_by_job_title_fuzzy(contractor.id, job_query)

                if lead:
                    logger.info(f"  ✅ FOUND MATCHING LEAD: {lead.job_title}")
                    await self.data_repo.mark_digest_opt_out(lead.conversation_id)

                    return (f"✅ Stopped daily digest for '{lead.job_title}'. "
                           f"You won't receive further updates about this lead.")
                else:
                    logger.warning(f"  ❌ NO MATCHING LEAD FOUND for query '{job_query}'")
                    return (f"❌ Couldn't find a lead matching '{job_query}'. "
                           f"Please check the job title and try again.")

        logger.info(f"  ℹ️ Not a takeover command - no patterns matched")
        return None
    
    async def handle_reach_out_command(self, message: str, contractor, is_whatsapp: bool) -> str | None:
        """Handle 'reach out to' command"""
        logger.info(f"📞 CHECKING FOR REACH OUT COMMAND")
        
        # Extract phone number from message
        phone_pattern = r"reach\s+out\s+to\s+(\+?\d+)"
        match = re.search(phone_pattern, message.lower())
        
        if not match:
            return None
            
        target_phone = match.group(1)
        logger.info(f"  Target phone: {target_phone}")
        
        # Normalize phone for database storage
        from modules.messaging.channel_manager import ChannelManager
        channel_manager = ChannelManager()
        target_phone_db = channel_manager.normalize_phone_for_db(target_phone, is_whatsapp)
        
        try:
            # Create new conversation
            convo = await self.conv_repo.create_conversation(
                contractor_id=contractor.id, 
                customer_phone=target_phone_db
            )
            logger.info(f"  🆕 Created new conversation: {convo.id}")

            # Send introduction message
            intro = (f"Hi! I'm {contractor.name}'s assistant. "
                    "To get started, please tell me the type of job you need.")

            # Get system phone for this contractor
            system_phone_db = contractor.phone_number  # Use the correct field name
            
            await self.msg_repo.create_message(
                sender=system_phone_db, 
                receiver=target_phone_db,
                body=intro, 
                direction="outbound", 
                conversation_id=convo.id
            )

            send_message(target_phone, intro, is_whatsapp)
            logger.info(f"  📤 Sent intro to customer: {target_phone}")

            return f"✅ Reached out to {target_phone}. Conversation started."
            
        except Exception as e:
            logger.error(f"❌ Failed to reach out to {target_phone}: {e}")
            return f"❌ Failed to reach out to {target_phone}. Please try again."
