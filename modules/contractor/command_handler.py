
# modules/contractor/command_handler.py

import re
import logging
from typing import Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from repos.conversation_data_repo import ConversationDataRepo
from utils.messaging import send_message
from .config import (
    TAKEOVER_PATTERNS, REACH_OUT_PATTERNS,
    TAKEOVER_SUCCESS_MSG, TAKEOVER_NOT_FOUND_MSG, TAKEOVER_MULTIPLE_MATCHES_MSG,
    REACH_OUT_SUCCESS_MSG, REACH_OUT_ERROR_MSG
)

logger = logging.getLogger(__name__)

class CommandHandler:
    """Handles contractor-specific commands like takeover and reach out."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.data_repo = ConversationDataRepo(session)
    
    async def handle_takeover_command(self, message_body: str, contractor) -> Optional[str]:
        """
        Handle takeover command from contractor.
        
        Args:
            message_body: The message content
            contractor: Contractor object
            
        Returns:
            Response message or None if not a takeover command
        """
        logger.info(f"Checking for takeover command in: '{message_body}'")
        
        job_title = self._extract_takeover_job_title(message_body)
        if not job_title:
            return None
            
        logger.info(f"Processing takeover for job: '{job_title}' (contractor: {contractor.name})")
        
        # Find matching conversation
        conversation_data = await self.data_repo.find_by_job_title_fuzzy(
            contractor.id, job_title
        )
        
        if not conversation_data:
            logger.warning(f"No lead found for takeover: '{job_title}'")
            return TAKEOVER_NOT_FOUND_MSG
        
        # Mark as opted out of digest (taken over)
        await self.data_repo.mark_digest_opt_out(conversation_data.conversation_id)
        
        logger.info(f"✅ Takeover completed for: {conversation_data.job_title}")
        return TAKEOVER_SUCCESS_MSG
    
    async def handle_reach_out_command(self, message_body: str, contractor, is_whatsapp: bool = False) -> Optional[str]:
        """
        Handle "reach out to" command from contractor.
        
        Args:
            message_body: The message content
            contractor: Contractor object
            is_whatsapp: Whether this is a WhatsApp message
            
        Returns:
            Response message or None if not a reach out command
        """
        logger.info(f"Checking for reach out command in: '{message_body}'")
        
        phone_and_message = self._extract_reach_out_details(message_body)
        if not phone_and_message:
            return None
            
        target_phone, custom_message = phone_and_message
        logger.info(f"Processing reach out to {target_phone} with message: '{custom_message[:50]}...'")
        
        try:
            # Send the custom message
            send_message(target_phone, custom_message, is_whatsapp)
            logger.info(f"✅ Reach out message sent to {target_phone}")
            return REACH_OUT_SUCCESS_MSG.format(phone=target_phone)
        except Exception as e:
            logger.error(f"❌ Failed to send reach out message: {e}")
            return REACH_OUT_ERROR_MSG
    
    def is_contractor_command(self, message_body: str) -> bool:
        """Check if message contains any contractor command."""
        return (
            self._extract_takeover_job_title(message_body) is not None or
            self._extract_reach_out_details(message_body) is not None
        )
    
    def _extract_takeover_job_title(self, message_body: str) -> Optional[str]:
        """Extract job title from takeover command."""
        for pattern in TAKEOVER_PATTERNS:
            match = re.search(pattern, message_body.strip(), re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_reach_out_details(self, message_body: str) -> Optional[Tuple[str, str]]:
        """Extract phone and message from reach out command."""
        for pattern in REACH_OUT_PATTERNS:
            match = re.search(pattern, message_body.strip(), re.IGNORECASE)
            if match:
                phone = match.group(1).strip()
                message = match.group(2).strip()
                return phone, message
        return None
