from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Message, Conversation


class MessageRepo:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_message(self,
                             sender: str,
                             receiver: str,
                             body: str,
                             direction: str,
                             conversation_id: str | None = None):
        """
        Insert a new SMS message into the log, tagged with conversation_id.
        """
        msg = Message(
            conversation_id=conversation_id,
            sender=sender,
            receiver=receiver,
            body=body,
            direction=direction,
        )
        self.session.add(msg)
        await self.session.commit()
        return msg

    async def get_recent_messages(self,
                                  customer: str,
                                  contractor: str,
                                  limit: int = 10):
        """
        Fetch the last `limit` messages between the two numbers, newest first.
        (Used only for classification when there's no active conversation.)
        """
        stmt = (select(
            Message.direction,
            Message.body).where(((Message.sender == customer)
                                 & (Message.receiver == contractor))
                                | ((Message.sender == contractor)
                                   & (Message.receiver == customer))).order_by(
                                       Message.timestamp.desc()).limit(limit))
        result = await self.session.execute(stmt)
        rows = result.all()
        return list(reversed(rows))

    async def get_all_conversation_messages(self, conversation_id: str):
        """
        Get the full history of a conversation, filtered by conversation_id.
        """
        stmt = (select(Message.direction, Message.body).where(
            Message.conversation_id == conversation_id).order_by(
                Message.timestamp.asc()))
        result = await self.session.execute(stmt)
        return result.all()
