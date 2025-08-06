from datetime import datetime
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from .models import ConversationData


class ConversationDataRepo:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, conversation_id: str, contractor_id: int,
                     customer_phone: str, data_dict: dict, qualified: bool,
                     job_title: str):
        existing = await self.session.get(ConversationData, conversation_id)
        if existing:
            existing.data_json = data_dict
            existing.last_updated = datetime.utcnow()
            existing.qualified = qualified
            existing.job_title = job_title
        else:
            new = ConversationData(
                conversation_id=conversation_id,
                contractor_id=contractor_id,
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

    # --- New methods for daily digest feature ---

    async def mark_digest_opt_out(self, conversation_id: str) -> None:
        """
        Mark a conversation to be opt-out from future daily digests.
        """
        cd = await self.session.get(ConversationData, conversation_id)
        if cd:
            cd.opt_out_of_digest = True
            await self.session.commit()

    async def mark_digest_sent(self, conversation_id: str,
                               timestamp: datetime) -> None:
        """
        Record the timestamp when a digest was sent for a conversation.
        """
        cd = await self.session.get(ConversationData, conversation_id)
        if cd:
            cd.last_digest_sent = timestamp
            await self.session.commit()

    async def get_qualified_for_digest(
            self, repeat_until_takeover: bool) -> list[ConversationData]:
        """
        Fetch all qualified leads eligible for the daily digest:
          - qualified == True
          - opt_out_of_digest == False
          - AND (last_digest_sent IS NULL OR repeat_until_takeover == True)
        """
        stmt = select(ConversationData).where(
            ConversationData.qualified == True,
            ConversationData.opt_out_of_digest == False,
            sa.or_(ConversationData.last_digest_sent.is_(None),
                   sa.bindparam('repeat_flag', repeat_until_takeover)))
        result = await self.session.execute(
            stmt.params(repeat_flag=repeat_until_takeover))
        return result.scalars().all()
