# repos/contractor_repo.py
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Contractor


class ContractorRepo:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_phone(self, phone: str) -> Contractor | None:
        stmt = select(Contractor).where(Contractor.phone == phone)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_by_id(self, contractor_id: int) -> Contractor | None:
        """Get contractor by ID - needed for daily digest"""
        return await self.session.get(Contractor, contractor_id)

    async def get_all(self) -> list[Contractor]:
        """Get all contractors - for debugging and viewing stored data"""
        stmt = select(Contractor)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def create(self, name: str, phone: str, address: str = None):
        c = Contractor(name=name, phone=phone, address=address)
        self.session.add(c)
        await self.session.commit()
        await self.session.refresh(c)
        return c

    # --- New method for loading digest configuration ---
    async def get_digest_config(self, contractor_id: int) -> dict:
        """
        Load and return the digest_config JSON for the given contractor.
        """
        contractor = await self.get_by_id(contractor_id)
        if contractor and hasattr(contractor, 'digest_config'):
            return contractor.digest_config or {}
        return {}

    async def get_by_assistant_phone(
            self, assistant_phone: str) -> Contractor | None:
        stmt = select(Contractor).where(
            Contractor.assistant_phone == assistant_phone)
        result = await self.session.execute(stmt)
        return result.scalars().first()
