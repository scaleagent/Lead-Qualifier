
# modules/contractor/contractor_service.py

import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from repos.contractor_repo import ContractorRepo
from repos.models import Contractor

logger = logging.getLogger(__name__)

class ContractorService:
    """Main service for contractor operations and identification."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.contractor_repo = ContractorRepo(session)
    
    async def identify_contractor(self, from_phone: str) -> Optional[Contractor]:
        """
        Identify contractor by phone number.
        
        Args:
            from_phone: Phone number to look up
            
        Returns:
            Contractor object if found, None otherwise
        """
        logger.debug(f"Identifying contractor for phone: {from_phone}")
        
        contractor = await self.contractor_repo.get_by_phone(from_phone)
        if contractor:
            logger.info(f"✅ Identified contractor: {contractor.name} (ID: {contractor.id})")
        else:
            logger.debug(f"❌ No contractor found for phone: {from_phone}")
            
        return contractor
    
    async def get_contractor_by_phone(self, phone: str) -> Optional[Contractor]:
        """Get contractor by phone number."""
        return await self.contractor_repo.get_by_phone(phone)
    
    async def get_contractor_by_assistant_phone(self, phone: str) -> Optional[Contractor]:
        """Get contractor by assistant phone number."""
        return await self.contractor_repo.get_by_assistant_phone(phone)
