
# modules/messaging/webhook_handler.py

import logging
import re
from fastapi import Depends, Form, Response
from sqlalchemy.ext.asyncio import AsyncSession

from repos.contractor_repo import ContractorRepo
from repos.conversation_repo import ConversationRepo
from repos.message_repo import MessageRepo
from repos.conversation_data_repo import ConversationDataRepo
from utils.messaging import send_message

from .channel_manager import ChannelManager
from .message_classifier import MessageClassifier

logger = logging.getLogger(__name__)


class MessageWebhookHandler:
    """Handles incoming SMS/WhatsApp webhook messages"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.contractor_repo = ContractorRepo(session)
        self.conv_repo = ConversationRepo(session)
        self.msg_repo = MessageRepo(session)
        self.data_repo = ConversationDataRepo(session)
        self.channel_manager = ChannelManager()
        self.classifier = MessageClassifier()
    
    async def handle_webhook(self, from_number: str, to_number: str, body: str) -> Response:
        """Main webhook handler entry point"""
        logger.info("=" * 50)
        logger.info("🔔 NEW SMS/WHATSAPP WEBHOOK CALL")
        logger.info(f"From: {from_number}")
        logger.info(f"To: {to_number}")
        logger.info(f"Body: '{body}'")
        logger.info("=" * 50)
        
        # Clean and normalize inputs
        from_number, to_number, body = from_number.strip(), to_number.strip(), body.strip()
        
        # Detect channel and normalize phone numbers
        is_whatsapp, from_phone_clean = self.channel_manager.detect_channel(from_number)
        customer_phone_db = self.channel_manager.normalize_phone_for_db(from_phone_clean, is_whatsapp)
        system_phone_db = self.channel_manager.get_system_number_for_db(is_whatsapp)
        
        logger.info(f"  Customer DB format: {customer_phone_db}")
        logger.info(f"  System DB format: {system_phone_db}")
        
        # Check if sender is a contractor
        contractor = await self.contractor_repo.get_by_phone(from_phone_clean)
        
        if contractor:
            return await self.handle_contractor_message(
                contractor, body, from_phone_clean, is_whatsapp, customer_phone_db, system_phone_db
            )
        else:
            return await self.handle_customer_message(
                from_phone_clean, body, is_whatsapp, customer_phone_db, system_phone_db
            )
    
    async def handle_contractor_message(
        self, contractor, body: str, from_phone_clean: str, 
        is_whatsapp: bool, customer_phone_db: str, system_phone_db: str
    ) -> Response:
        """Handle messages from contractors (commands)"""
        logger.info(f"✅ CONTRACTOR IDENTIFIED: {contractor.name} (ID: {contractor.id})")
        
        response_phone = self.channel_manager.get_response_phone(from_phone_clean)
        
        # Check for takeover command
        takeover_response = await self.handle_takeover_command(body, contractor)
        if takeover_response:
            await self.log_and_send_response(
                takeover_response, response_phone, is_whatsapp,
                customer_phone_db, system_phone_db, body
            )
            return Response(status_code=204)
        
        # Check for reach out command
        is_reach_out, target_phone = self.classifier.is_reach_out_command(body)
        if is_reach_out:
            await self.handle_reach_out_command(
                contractor, target_phone, is_whatsapp, response_phone,
                customer_phone_db, system_phone_db
            )
            return Response(status_code=204)
        
        return Response(status_code=204)
    
    async def handle_customer_message(
        self, from_phone_clean: str, body: str, is_whatsapp: bool,
        customer_phone_db: str, system_phone_db: str
    ) -> Response:
        """Handle messages from customers"""
        logger.info(f"👤 CUSTOMER MESSAGE from: {customer_phone_db}")
        
        # Find the contractor that owns this assistant number
        contractor = await self.contractor_repo.get_by_assistant_phone(system_phone_db)
        if not contractor:
            logger.warning(f"⚠️ Unrecognized assistant number: {system_phone_db}; dropping message.")
            return Response(status_code=204)
        
        logger.info(f"🏢 Routing customer message to contractor: {contractor.name} (ID: {contractor.id})")
        
        # This is where we would continue with the customer qualification flow
        # For now, let's just log it and return success
        logger.info("📋 Customer qualification flow - to be implemented")
        
        return Response(status_code=204)
    
    async def handle_takeover_command(self, body: str, contractor) -> str | None:
        """Handle contractor takeover commands"""
        logger.info(f"🎯 CHECKING FOR TAKEOVER COMMAND")
        logger.info(f"  Contractor: {contractor.name} (ID: {contractor.id})")
        logger.info(f"  Message: '{body}'")
        
        patterns = [
            r"stop\s+(?:daily\s+)?digest\s+for\s+(.+)",
            r"takeover\s+(.+)",
            r"take\s+over\s+(.+)",
            r"claim\s+(.+)"
        ]
        
        for pattern_idx, pattern in enumerate(patterns):
            logger.debug(f"  Testing pattern {pattern_idx + 1}: {pattern}")
            match = re.match(pattern, body.lower().strip())
            
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
    
    async def handle_reach_out_command(
        self, contractor, target_phone: str, is_whatsapp: bool,
        response_phone: str, customer_phone_db: str, system_phone_db: str
    ) -> None:
        """Handle 'reach out to' command"""
        target_phone_db = self.channel_manager.normalize_phone_for_db(target_phone, is_whatsapp)
        logger.info(f"📞 REACH OUT COMMAND DETECTED")
        logger.info(f"  Target phone: {target_phone}")
        logger.info(f"  Target DB format: {target_phone_db}")
        
        # Create new conversation
        convo = await self.conv_repo.create_conversation(
            contractor_id=contractor.id, customer_phone=target_phone_db
        )
        logger.info(f"  🆕 Created new conversation: {convo.id}")
        
        # Send introduction message
        intro = (f"Hi! I'm {contractor.name}'s assistant. "
                "To get started, please tell me the type of job you need.")
        
        await self.msg_repo.create_message(
            sender=system_phone_db, receiver=target_phone_db,
            body=intro, direction="outbound", conversation_id=convo.id
        )
        
        send_message(target_phone, intro, is_whatsapp)
        logger.info(f"  📤 Sent intro to customer: {target_phone}")
        
        # Send confirmation to contractor
        confirmation = f"✅ Reached out to {target_phone}. Conversation started."
        send_message(response_phone, confirmation, is_whatsapp)
        logger.info(f"  📤 Sent confirmation to: {response_phone}")
    
    async def log_and_send_response(
        self, response: str, response_phone: str, is_whatsapp: bool,
        customer_phone_db: str, system_phone_db: str, original_body: str
    ) -> None:
        """Log command and response to database"""
        try:
            # Log the command
            await self.msg_repo.create_message(
                sender=customer_phone_db, receiver=system_phone_db,
                body=original_body, direction="inbound", conversation_id=None
            )
            
            # Send and log response
            send_message(response_phone, response, is_whatsapp)
            await self.msg_repo.create_message(
                sender=system_phone_db, receiver=customer_phone_db,
                body=response, direction="outbound", conversation_id=None
            )
            
            logger.info(f"✅ Command and response logged successfully")
        except Exception as e:
            logger.error(f"❌ Failed to log command/response: {e}")
