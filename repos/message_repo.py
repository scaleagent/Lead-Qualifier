from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Message, Conversation


class MessageRepo:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_message(self, sender: str, receiver: str, body: str,
                             direction: str):
        """
        Insert a new SMS message into the log.
        """
        msg = Message(
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
        # return oldest→newest
        return list(reversed(rows))

    async def get_all_conversation_messages(self, conversation_id: str):
        """
        Get the full history of a conversation by joining on the Conversation table.
        """
        # First, fetch the convo to learn the two phone numbers
        convo = await self.session.get(Conversation, conversation_id)
        if not convo:
            return []

        stmt = (select(Message.direction, Message.body).where((
            (Message.sender == convo.customer_phone)
            & (Message.receiver == convo.contractor_phone)) | (
                (Message.sender == convo.contractor_phone)
                & (Message.receiver == convo.customer_phone))).order_by(
                    Message.timestamp.asc()))
        result = await self.session.execute(stmt)
        return result.all()
