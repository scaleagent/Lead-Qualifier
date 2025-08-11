#repos/conversation_data_repo.py

from datetime import datetime
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from .models import ConversationData
import sqlalchemy as sa
import logging
from .models import ConversationData

logger = logging.getLogger(__name__)


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
    Used when contractor takes over a lead.
    """
    logger.debug(
        f"Marking conversation {conversation_id[:8]}... as digest opt-out")

    cd = await self.session.get(ConversationData, conversation_id)
    if cd:
        previous_value = cd.opt_out_of_digest
        cd.opt_out_of_digest = True
        await self.session.commit()
        logger.info(
            f"✅ Marked {conversation_id[:8]}... as opted out (was: {previous_value})"
        )
    else:
        logger.warning(
            f"⚠️ Conversation {conversation_id[:8]}... not found for opt-out")


async def mark_digest_sent(self, conversation_id: str,
                           timestamp: datetime) -> None:
    """
    Record the timestamp when a digest was sent for a conversation.
    """
    logger.debug(
        f"Marking digest sent for {conversation_id[:8]}... at {timestamp}")

    cd = await self.session.get(ConversationData, conversation_id)
    if cd:
        cd.last_digest_sent = timestamp
        await self.session.commit()
        logger.debug(f"✅ Marked digest sent for {conversation_id[:8]}...")
    else:
        logger.warning(
            f"⚠️ Conversation {conversation_id[:8]}... not found for digest marking"
        )


async def get_qualified_for_digest(
        self, contractor_id: int,
        repeat_until_takeover: bool) -> list[ConversationData]:
    """
    Fetch all qualified leads eligible for the daily digest
    """
    logger.info(f"Fetching qualified leads for contractor {contractor_id}")
    logger.debug(f"  repeat_until_takeover: {repeat_until_takeover}")

    # Build base conditions
    conditions = [
        ConversationData.contractor_id == contractor_id,
        ConversationData.qualified == True,
        ConversationData.opt_out_of_digest == False
    ]

    # Add the repeat logic
    if repeat_until_takeover:
        logger.debug(f"  Including previously sent leads (repeat mode)")
        stmt = select(ConversationData).where(*conditions)
    else:
        logger.debug(f"  Excluding previously sent leads (send once mode)")
        conditions.append(ConversationData.last_digest_sent.is_(None))
        stmt = select(ConversationData).where(*conditions)

    result = await self.session.execute(stmt)
    leads = result.scalars().all()

    logger.info(f"  Found {len(leads)} qualified leads for digest")
    for idx, lead in enumerate(leads[:3], 1):
        logger.debug(
            f"    {idx}. {lead.job_title} (sent: {lead.last_digest_sent is not None})"
        )
    if len(leads) > 3:
        logger.debug(f"    ... and {len(leads) - 3} more")

    return leads


async def find_by_job_title_fuzzy(self, contractor_id: int,
                                  job_title: str) -> ConversationData | None:
    """
    Find a conversation by fuzzy matching job title for takeover command.
    """
    logger.info(
        f"Searching for job title matching '{job_title}' for contractor {contractor_id}"
    )

    stmt = select(ConversationData).where(
        ConversationData.contractor_id == contractor_id,
        ConversationData.job_title.ilike(f"%{job_title}%"))

    logger.debug(f"  SQL: Looking for job_title ILIKE '%{job_title}%'")

    result = await self.session.execute(stmt)
    matches = result.scalars().all()

    if not matches:
        logger.warning(f"  ❌ No matches found for '{job_title}'")
        return None

    if len(matches) == 1:
        logger.info(f"  ✅ Found exact match: {matches[0].job_title}")
        return matches[0]

    # Multiple matches - return the best one (shortest title that contains the query)
    logger.warning(f"  ⚠️ Found {len(matches)} matches for '{job_title}':")
    for idx, match in enumerate(matches[:5], 1):
        logger.debug(
            f"    {idx}. {match.job_title} ({match.conversation_id[:8]}...)")

    # Return the shortest matching title (likely most specific)
    best_match = min(matches, key=lambda x: len(x.job_title or ''))
    logger.info(f"  Using best match: {best_match.job_title}")

    return best_match
