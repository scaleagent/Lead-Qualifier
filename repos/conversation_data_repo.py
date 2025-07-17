from datetime import datetime
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from .models import ConversationData


class ConversationDataRepo:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(
        self,
        conversation_id: str,
        contractor_phone: str,
        customer_phone: str,
        data_dict: dict,
        qualified: bool,
        job_title: str,
    ):
        """
        Insert or update the qualification data for a conversation.
        """
        existing = await self.session.get(ConversationData, conversation_id)
        if existing:
            existing.data_json = data_dict
            existing.last_updated = datetime.utcnow()
            existing.qualified = qualified
            existing.job_title = job_title
        else:
            new = ConversationData(
                conversation_id=conversation_id,
                contractor_phone=contractor_phone,
                customer_phone=customer_phone,
                data_json=data_dict,
                last_updated=datetime.utcnow(),
                qualified=qualified,
                job_title=job_title,
            )
            self.session.add(new)
        await self.session.commit()

    async def mark_handed_over(self, job_title_lower: str):
        """
        Mark qualified=2 (handed over) for a given job title.
        """
        stmt = select(ConversationData).where(
            ConversationData.job_title.ilike(job_title_lower))
        result = await self.session.execute(stmt)
        row = result.scalars().first()
        if row:
            row.qualified = True  # or use an enum/status int if you prefer
            await self.session.commit()

    async def get_all_qualified(self):
        """
        Return all rows where `qualified == True`.
        """
        stmt = select(ConversationData).where(
            ConversationData.qualified == True)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_all(self) -> list[ConversationData]:
        """
        Return every ConversationData row (qualified and not).
        """
        stmt = select(ConversationData)
        result = await self.session.execute(stmt)
        return result.scalars().all()
