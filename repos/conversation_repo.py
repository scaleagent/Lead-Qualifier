# repos/conversation_repo.py

from datetime import datetime
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Conversation


class ConversationRepo:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_active_conversation(self, contractor_id: int,
                                      customer_phone: str):
        stmt = (select(Conversation).where(
            Conversation.contractor_id == contractor_id,
            Conversation.customer_phone == customer_phone,
            Conversation.closed_at.is_(None),
        ))
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_active_by_customer(self, customer_phone: str):
        stmt = (select(Conversation).where(
            Conversation.customer_phone == customer_phone,
            Conversation.closed_at.is_(None),
        ))
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def create_conversation(self, contractor_id: int,
                                  customer_phone: str):
        # First, close any existing active conversation for this phone pair
        existing = await self.get_active_conversation(contractor_id, customer_phone)
        if existing:
            existing.closed_at = datetime.utcnow()
            existing.updated_at = datetime.utcnow()
            
        # Create new conversation
        convo = Conversation(contractor_id=contractor_id,
                             customer_phone=customer_phone)
        self.session.add(convo)
        await self.session.commit()
        await self.session.refresh(convo)
        return convo

    async def close_conversation(self, conversation_id: str):
        convo = await self.session.get(Conversation, conversation_id)
        if convo:
            convo.closed_at = datetime.utcnow()
            convo.updated_at = datetime.utcnow()
            await self.session.commit()

    async def close_all_active_for_customer(self, customer_phone: str):
        """
        Close all active conversations for a customer phone number 
        (across all contractors). Useful when customer starts fresh.
        """
        stmt = select(Conversation).where(
            Conversation.customer_phone == customer_phone,
            Conversation.closed_at.is_(None)
        )
        result = await self.session.execute(stmt)
        active_conversations = result.scalars().all()
        
        for convo in active_conversations:
            convo.closed_at = datetime.utcnow()
            convo.updated_at = datetime.utcnow()
        
        if active_conversations:
            await self.session.commit()
        
        return len(active_conversations)

    # --- New method for fetching ongoing (collecting-notes) leads ---
    async def get_collecting_notes_for_contractor(
            self, contractor_id: int) -> list[Conversation]:
        """
        Return all conversations for a contractor that are in COLLECTING_NOTES status.
        """
        stmt = select(Conversation).where(
            Conversation.contractor_id == contractor_id,
            Conversation.status == "COLLECTING_NOTES")
        result = await self.session.execute(stmt)
        return result.scalars().all()
