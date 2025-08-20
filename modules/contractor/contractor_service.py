
from sqlalchemy.ext.asyncio import AsyncSession
from repos.contractor_repo import ContractorRepo
import logging

logger = logging.getLogger(__name__)


class ContractorService:
    """Service for contractor operations including identification and profile management"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.contractor_repo = ContractorRepo(session)
    
    async def identify_contractor(self, phone_number: str):
        """
        Identify contractor by phone number.
        Checks if the phone number belongs to a contractor.
        """
        logger.debug(f"Identifying contractor for phone: {phone_number}")
        
        # Check if this phone belongs to a contractor
        contractor = await self.contractor_repo.get_by_phone(phone_number)
        
        if contractor:
            logger.info(f"✅ Identified contractor: {contractor.name} (ID: {contractor.id})")
            return contractor
        else:
            logger.debug(f"No contractor found for phone: {phone_number}")
            return None
    
    async def get_contractor_by_phone(self, phone_number: str):
        """Get contractor by phone number"""
        return await self.contractor_repo.get_by_phone(phone_number)
    
    async def get_contractor_by_assistant_phone(self, assistant_phone: str):
        """Get contractor by assistant phone number"""
        return await self.contractor_repo.get_by_assistant_phone(assistant_phone)
