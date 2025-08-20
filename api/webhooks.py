
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
    logger.info("🚀 SMS WEBHOOK ENDPOINT HIT!")
    logger.info(f"From: {From}")
    logger.info(f"To: {To}")
    logger.info(f"Body: {Body}")
    
    try:
        handler = MessageWebhookHandler(session)
        await handler.handle_webhook(From, To, Body)
        
        logger.info("✅ Webhook handled successfully")
        # Return 204 No Content for SMS webhooks (Twilio requirement)
        return Response(status_code=204)
        
    except Exception as e:
        logger.error(f"❌ Webhook handler error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Still return 204 to Twilio to avoid retries
        return Response(status_code=204)
