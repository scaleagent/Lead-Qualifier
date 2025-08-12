
# Digest Module

A self-contained, modular daily digest system for sending lead summaries to contractors via SMS.

## Features

- ✅ **Complete isolation** - Can be worked on independently
- ✅ **PDF generation** - Secure, token-based PDF transcripts
- ✅ **SMS formatting** - Optimized message formatting for mobile
- ✅ **Configurable scheduling** - Per-contractor timezone and hour settings
- ✅ **Test mode** - Override phone numbers for testing

## Structure

```
modules/digest/
├── __init__.py              # Module initialization
├── config.py                # Configuration constants
├── digest_service.py        # Main digest logic
├── pdf_generator.py         # PDF generation and security
├── message_formatter.py     # SMS message formatting
├── api.py                   # FastAPI endpoints
└── README.md               # This file
```

## Key Components

### DigestService
Main service class that orchestrates the digest process:
- Fetches contractor configurations
- Handles timezone-based scheduling
- Sends qualified and ongoing lead summaries

### PDFGenerator  
Handles secure PDF generation:
- Token-based security with HMAC signatures
- Time-limited access (24 hours by default)
- Comprehensive lead transcripts

### MessageFormatter
Optimizes SMS content:
- Fits within SMS length limits
- Includes key lead information
- Provides PDF links

## Usage

### Adding to FastAPI App
```python
from modules.digest.api import router as digest_router
app.include_router(digest_router)
```

### Running Digest Manually
```python
from modules.digest.digest_service import DigestService

service = DigestService()
await service.run_daily_digest(force=True, only_contractor_id=1)
```

## API Endpoints

- `GET /digest/pdf/transcript?token=xxx` - Secure PDF access
- `POST /digest/trigger/{contractor_id}` - Manual digest trigger (TEST_MODE only)
- `POST /digest/generate-pdf-link` - Generate PDF link

## Environment Variables

- `PDF_SECRET_KEY` - HMAC key for PDF security
- `APP_BASE_URL` - Base URL for PDF links  
- `DIGEST_TEST_PHONE` - Override phone for testing
- `TEST_MODE` - Enable test features

## Dependencies

This module depends on:
- `repos/*` - Database repositories
- `utils.messaging` - SMS sending
- `reportlab` - PDF generation
- Standard Python libraries

## Testing

Set `TEST_MODE=true` and `DIGEST_TEST_PHONE=+1234567890` to test without affecting real contractors.
