
# api/webhooks.py

import logging
from fastapi import APIRouter, Depends, Form, Response
from sqlalchemy.ext.asyncio import AsyncSession

from repos.database import AsyncSessionLocal
from modules.messaging import MessageWebhookHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


async def get_session():
    """Database session dependency"""
    async with AsyncSessionLocal() as session:
        yield session


@router.post("/sms", response_class=Response)
async def sms_webhook(From: str = Form(...),
                      To: str = Form(...),
                      Body: str = Form(...),
                      session=Depends(get_session)):
    """
    SMS/WhatsApp webhook handler - receives messages from Twilio
    
    This endpoint handles all incoming SMS and WhatsApp messages,
    routing them through the modular messaging system.
    """
    logger.info(f"Received message from {From} to {To}")
    
    handler = MessageWebhookHandler(session)
    response_text = await handler.handle_webhook(From, To, Body)
    
    # Return 204 No Content for SMS webhooks (Twilio requirement)
    return Response(content=response_text, status_code=204)
