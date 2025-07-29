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

    async def create(self, name: str, phone: str, address: str = None):
        c = Contractor(name=name, phone=phone, address=address)
        self.session.add(c)
        await self.session.commit()
        await self.session.refresh(c)
        return c
