
# api/contractors.py

import logging
from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from repos.database import AsyncSessionLocal
from repos.contractor_repo import ContractorRepo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/contractors", tags=["contractors"])


async def get_session():
    """Database session dependency"""
    async with AsyncSessionLocal() as session:
        yield session


@router.get("/", response_class=PlainTextResponse)
async def list_contractors(session=Depends(get_session)):
    """
    Debug endpoint to view stored contractor profiles.
    
    Returns a formatted list of all contractors in the system
    with their basic information for debugging purposes.
    """
    logger.info("Fetching all contractors for debug view")
    
    contractor_repo = ContractorRepo(session)
    contractors = await contractor_repo.get_all()

    if not contractors:
        return "❌ No contractors found in database"

    result = ["📋 STORED CONTRACTORS:"]
    for c in contractors:
        result.append(f"  ID: {c.id}")
        result.append(f"  Name: {c.name}")
        result.append(f"  Phone: {c.phone}")
        result.append(f"  Address: {c.address or 'Not set'}")
        result.append(f"  Created: {c.created_at}")
        result.append("  " + "-" * 30)

    logger.info(f"Found {len(contractors)} contractors")
    return "\n".join(result)
