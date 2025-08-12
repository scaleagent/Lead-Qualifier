
# modules/qualification/qualification_service.py

import logging
from datetime import datetime
from typing import Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from openai import OpenAI

from repos.models import Conversation, ConversationData
from repos.message_repo import MessageRepo
from repos.conversation_data_repo import ConversationDataRepo
from utils.messaging import send_message
from .data_extractor import DataExtractor
from .flow_manager import FlowManager

logger = logging.getLogger(__name__)


class QualificationService:
    """Main service for handling lead qualification process."""
    
    def __init__(self, session: AsyncSession, openai_client: OpenAI):
        self.session = session
        self.data_extractor = DataExtractor(openai_client)
        self.flow_manager = FlowManager()
        self.msg_repo = MessageRepo(session)
        self.data_repo = ConversationDataRepo(session)
    
    async def process_qualification(
        self,
        conversation: Conversation,
        message_body: str,
        contractor,
        customer_phone_clean: str,
        customer_phone_db: str,
        system_phone_db: str,
        is_whatsapp: bool
    ) -> bool:
        """
        Main entry point for processing qualification messages.
        
        Returns:
            bool: True if message was handled, False if not
        """
        logger.info(f"Processing qualification for conversation {conversation.id}")
        
        # Handle CONFIRMING state
        if conversation.status == "CONFIRMING":
            return await self._handle_confirmation_state(
                conversation, message_body, contractor,
                customer_phone_clean, customer_phone_db, system_phone_db, is_whatsapp
            )
        
        # Handle COLLECTING_NOTES state
        if conversation.status == "COLLECTING_NOTES":
            return await self._handle_notes_collection_state(
                conversation, message_body,
                customer_phone_clean, customer_phone_db, system_phone_db, is_whatsapp
            )
        
        # Handle normal QUALIFYING state
        return await self._handle_qualifying_state(
            conversation, contractor,
            customer_phone_clean, customer_phone_db, system_phone_db, is_whatsapp
        )
    
    async def _handle_confirmation_state(
        self,
        conversation: Conversation,
        message_body: str,
        contractor,
        customer_phone_clean: str,
        customer_phone_db: str,
        system_phone_db: str,
        is_whatsapp: bool
    ) -> bool:
        """Handle customer confirmation or corrections."""
        logger.debug(f"Handling confirmation state for conversation {conversation.id}")
        
        if self.flow_manager.is_affirmative(message_body):
            logger.info("Customer confirmed qualification data - moving to notes collection")
            
            follow_up = (
                "Thanks! If there's any other important info—parking, pets, special access—"
                "just reply here. When you're done, reply 'No'."
            )
            
            await self._send_and_log_message(
                system_phone_db, customer_phone_db, follow_up,
                "outbound", conversation.id
            )
            send_message(customer_phone_clean, follow_up, is_whatsapp)
            
            conversation.status = "COLLECTING_NOTES"
            await self.session.commit()
            
        else:
            logger.info("Customer provided corrections - updating qualification data")
            
            # Get conversation history and apply corrections
            full_msgs = await self.msg_repo.get_all_conversation_messages(conversation.id)
            history = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}" for d, b in full_msgs)
            
            # Extract current data and apply corrections
            current_data = await self.data_extractor.extract_qualification_data(history)
            updated_data = await self.data_extractor.apply_correction_data(current_data, message_body)
            
            # Update stored data
            await self.data_repo.upsert(
                conversation_id=conversation.id,
                contractor_id=contractor.id,
                customer_phone=customer_phone_db,
                data_dict=updated_data,
                qualified=self.flow_manager.is_qualified(updated_data),
                job_title=updated_data.get("job_type")
            )
            
            # Show updated summary
            summary = self.flow_manager.generate_summary(updated_data)
            await self._send_and_log_message(
                system_phone_db, customer_phone_db, summary,
                "outbound", conversation.id
            )
            send_message(customer_phone_clean, summary, is_whatsapp)
        
        return True
    
    async def _handle_notes_collection_state(
        self,
        conversation: Conversation,
        message_body: str,
        customer_phone_clean: str,
        customer_phone_db: str,
        system_phone_db: str,
        is_whatsapp: bool
    ) -> bool:
        """Handle additional notes collection."""
        logger.debug(f"Handling notes collection state for conversation {conversation.id}")
        
        if self.flow_manager.is_negative(message_body):
            logger.info("Customer finished notes collection - completing conversation")
            
            closing = "Great—thanks! I'll pass this along to your contractor. ✅"
            await self._send_and_log_message(
                system_phone_db, customer_phone_db, closing,
                "outbound", conversation.id
            )
            send_message(customer_phone_clean, closing, is_whatsapp)
            
            # Mark conversation as complete
            conversation.status = "COMPLETE"
            await self.session.commit()
            
        else:
            logger.info("Adding customer message to notes")
            
            # Add to notes
            cd: ConversationData = await self.session.get(ConversationData, conversation.id)
            if cd:
                existing_notes = cd.data_json.get("notes", "")
                combined_notes = (existing_notes + "\n" + message_body).strip()
                cd.data_json["notes"] = combined_notes
                cd.last_updated = datetime.utcnow()
                await self.session.commit()
            
            prompt = "Anything else to add? If not, reply 'No'."
            await self._send_and_log_message(
                system_phone_db, customer_phone_db, prompt,
                "outbound", conversation.id
            )
            send_message(customer_phone_clean, prompt, is_whatsapp)
        
        return True
    
    async def _handle_qualifying_state(
        self,
        conversation: Conversation,
        contractor,
        customer_phone_clean: str,
        customer_phone_db: str,
        system_phone_db: str,
        is_whatsapp: bool
    ) -> bool:
        """Handle normal qualification data collection."""
        logger.debug(f"Handling qualifying state for conversation {conversation.id}")
        
        # Extract qualification data from full conversation
        full_msgs = await self.msg_repo.get_all_conversation_messages(conversation.id)
        history = "\n".join(f"{'Customer' if d=='inbound' else 'AI'}: {b}" for d, b in full_msgs)
        data = await self.data_extractor.extract_qualification_data(history)
        
        logger.debug(f"Extracted data: {data}")
        
        # Store/update the qualification data
        await self.data_repo.upsert(
            conversation_id=conversation.id,
            contractor_id=contractor.id,
            customer_phone=customer_phone_db,
            data_dict=data,
            qualified=self.flow_manager.is_qualified(data),
            job_title=data.get("job_type")
        )
        
        # Check if we need more information
        missing_fields = self.flow_manager.get_missing_fields(data)
        logger.debug(f"Missing fields: {missing_fields}")
        
        if missing_fields:
            # Ask for missing information
            prompt = self.flow_manager.generate_prompt(data)
            if prompt:
                logger.debug(f"Asking for missing info: {prompt}")
                await self._send_and_log_message(
                    system_phone_db, customer_phone_db, prompt,
                    "outbound", conversation.id
                )
                send_message(customer_phone_clean, prompt, is_whatsapp)
        else:
            # All fields collected - show summary for confirmation
            logger.info("All fields collected - showing summary for confirmation")
            summary = self.flow_manager.generate_summary(data)
            
            await self._send_and_log_message(
                system_phone_db, customer_phone_db, summary,
                "outbound", conversation.id
            )
            send_message(customer_phone_clean, summary, is_whatsapp)
            
            conversation.status = "CONFIRMING"
            await self.session.commit()
        
        return True
    
    async def _send_and_log_message(
        self,
        sender: str,
        receiver: str,
        body: str,
        direction: str,
        conversation_id: str
    ) -> None:
        """Helper to send and log messages."""
        await self.msg_repo.create_message(
            sender=sender,
            receiver=receiver,
            body=body,
            direction=direction,
            conversation_id=conversation_id
        )
