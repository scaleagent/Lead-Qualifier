from datetime import datetime
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Conversation


class ConversationRepo:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_active_conversation(self, contractor_phone: str,
                                      customer_phone: str):
        """
        Return the open 'QUALIFYING' conversation for these two numbers, if any.
        """
        q = (select(Conversation).where(
            Conversation.contractor_phone == contractor_phone,
            Conversation.customer_phone == customer_phone,
            Conversation.status == "QUALIFYING",
        ))
        result = await self.session.execute(q)
        return result.scalars().first()

    async def create_conversation(self, contractor_phone: str,
                                  customer_phone: str):
        """
        Start a new conversation record.
        """
        convo = Conversation(
            contractor_phone=contractor_phone,
            customer_phone=customer_phone,
        )
        self.session.add(convo)
        await self.session.commit()
        await self.session.refresh(convo)
        return convo

    async def close_conversation(self, conversation_id: str):
        """
        Mark a conversation as COMPLETE.
        """
        convo = await self.session.get(Conversation, conversation_id)
        if convo:
            convo.status = "COMPLETE"
            convo.updated_at = datetime.utcnow()
            await self.session.commit()
