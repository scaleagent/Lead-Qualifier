
# modules/digest/api.py - API endpoints for digest functionality

import os
import logging
from fastapi import APIRouter, Depends, Form, Response
from fastapi.responses import PlainTextResponse
from urllib.parse import quote

from repos.database import AsyncSessionLocal
from .digest_service import DigestService
from .pdf_generator import PDFGenerator
from .config import TEST_MODE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/digest", tags=["digest"])


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


@router.get("/pdf/transcript")
async def get_transcript_pdf(token: str, session=Depends(get_session)):
    """Secure endpoint for PDF transcript generation"""
    # Verify the token
    is_valid, conversation_id = PDFGenerator.verify_pdf_token(token)

    if not is_valid:
        logger.warning(f"Invalid PDF access attempt with token: {token[:20]}...")
        return PlainTextResponse(
            "This link has expired or is invalid. Please request a new digest from your contractor.",
            status_code=403)

    try:
        # Log successful access
        logger.info(f"Generating PDF for conversation {conversation_id[:8]}...")

        # Generate the PDF with current data
        pdf_bytes = await PDFGenerator.generate_conversation_pdf(session, conversation_id)

        # Return PDF as streaming response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename=lead_transcript_{conversation_id[:8]}.pdf",
                "Cache-Control": "private, max-age=300",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY"
            })

    except ValueError as e:
        logger.error(f"Conversation not found: {conversation_id}")
        return PlainTextResponse(
            "This conversation could not be found. It may have been deleted.",
            status_code=404)

    except Exception as e:
        logger.error(f"PDF generation failed for {conversation_id}: {e}", exc_info=True)
        return PlainTextResponse(
            "An error occurred generating the PDF. Please try again later.",
            status_code=500)


@router.post("/trigger/{contractor_id}")
async def trigger_digest_manually(contractor_id: int):
    """Manually trigger digest for a specific contractor (for testing)"""
    if not TEST_MODE:
        return {"error": "Manual trigger only available in TEST_MODE"}

    logger.info(f"Manually triggering digest for contractor {contractor_id} (force override)")

    try:
        service = DigestService()
        await service.run_daily_digest(force=True, only_contractor_id=contractor_id)
        return {
            "status": "success",
            "message": f"Forced digest sent (or attempted) for contractor {contractor_id}",
            "note": "Check logs for details"
        }
    except Exception as e:
        logger.error(f"Failed to trigger digest: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/generate-pdf-link")
async def generate_pdf_link_endpoint(conversation_id: str = Form(...)):
    """Generate a PDF link for a specific conversation"""
    url = PDFGenerator.generate_pdf_url(conversation_id)
    return {
        "conversation_id": conversation_id,
        "pdf_url": url,
        "expires_in": "24 hours",
        "generated_at": f"{os.urandom(8).hex()}"
    }
