
# api/health.py

import logging
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/", response_class=PlainTextResponse)
def health_check():
    """
    Root health check endpoint.
    
    Returns a simple status message to verify the API is running.
    Used by monitoring systems and load balancers.
    """
    logger.debug("Health check requested")
    return "✅ SMS-Lead-Qual API is running."


@router.get("/health", response_class=PlainTextResponse)
def detailed_health():
    """
    Detailed health check endpoint.
    
    Can be extended to include database connectivity,
    external service status, etc.
    """
    logger.debug("Detailed health check requested")
    return "✅ All systems operational"
