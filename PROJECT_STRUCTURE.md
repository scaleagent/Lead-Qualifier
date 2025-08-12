
# Project Structure Guide

This document explains the organization of the SMS Lead Qualification Bot codebase in simple terms.

## Overview

The project is organized into **modules** (feature areas), **repos** (database access), **api** (web endpoints), and **utils** (shared tools). This modular structure means you can work on one feature without needing to understand the entire codebase.

## Directory Structure

```
├── api/                    # Web endpoints (FastAPI routes)
├── modules/                # Feature modules (business logic)
├── repos/                  # Database access layer
├── utils/                  # Shared utilities
├── main.py                 # App startup and configuration
└── PROJECT_STRUCTURE.md    # This documentation
```

## Detailed Breakdown

### `/api/` - Web Endpoints
**What it does:** Handles incoming web requests (SMS webhooks, health checks, etc.)

- `webhooks.py` - Receives SMS/WhatsApp messages from Twilio
- `contractors.py` - API endpoints for contractor management
- `health.py` - System status endpoints for monitoring
- `README.md` - API layer documentation

**When to modify:** When adding new web endpoints or changing HTTP request handling.

### `/modules/` - Feature Modules
**What it does:** Contains the main business logic, organized by feature area.

#### `/modules/messaging/`
Handles all SMS/WhatsApp message processing.
- `webhook_handler.py` - Main message processing logic
- `channel_manager.py` - Detects SMS vs WhatsApp, normalizes phone numbers
- `message_classifier.py` - Determines if message is new conversation or continuation
- `config.py` - Messaging-specific settings

#### `/modules/qualification/`
Manages the lead qualification process (collecting job details from customers).
- `qualification_service.py` - Main qualification logic
- `data_extractor.py` - Uses AI to extract job details from messages
- `flow_manager.py` - Manages the question/answer flow
- `config.py` - Qualification settings (required fields, AI prompts)
- `README.md` - Qualification module documentation

#### `/modules/contractor/`
Handles contractor-specific features (commands like "takeover", "reach out to").
- `contractor_service.py` - Main contractor operations
- `command_handler.py` - Processes contractor commands
- `config.py` - Contractor command patterns and messages
- `README.md` - Contractor module documentation

#### `/modules/digest/`
Manages daily digest emails/messages to contractors.
- `digest_service.py` - Main digest generation logic
- `message_formatter.py` - Formats digest messages
- `pdf_generator.py` - Creates PDF reports
- `api.py` - Digest-specific endpoints
- `config.py` - Digest settings
- `README.md` - Digest module documentation

### `/repos/` - Database Access
**What it does:** Handles all database operations (saving, retrieving, updating data).

- `models.py` - Database table definitions
- `database.py` - Database connection setup
- `conversation_repo.py` - Conversation data operations
- `conversation_data_repo.py` - Lead qualification data operations
- `contractor_repo.py` - Contractor data operations
- `message_repo.py` - Message history operations

**When to modify:** When changing how data is stored or retrieved from the database.

### `/utils/` - Shared Utilities
**What it does:** Common functions used across multiple modules.

- `messaging.py` - Functions for sending SMS/WhatsApp messages

**When to modify:** When adding utility functions that multiple modules need.

### Root Files
- `main.py` - Application startup, FastAPI setup, scheduler configuration
- `requirements.txt` - Python package dependencies
- `pyproject.toml` - Python project configuration
- `.replit` - Replit environment configuration

## How to Work with This Structure

### Adding a New Feature
1. Create a new folder in `/modules/` (e.g., `/modules/scheduling/`)
2. Add your business logic files there
3. Create any needed database operations in `/repos/`
4. Add web endpoints in `/api/` if needed
5. Update this documentation

### Working on Existing Features
- **SMS/WhatsApp issues:** Look in `/modules/messaging/`
- **Lead qualification problems:** Look in `/modules/qualification/`
- **Contractor commands:** Look in `/modules/contractor/`
- **Daily digest:** Look in `/modules/digest/`
- **Database issues:** Look in `/repos/`
- **Web endpoint issues:** Look in `/api/`

### Finding Specific Functionality
- **Message processing:** `modules/messaging/webhook_handler.py`
- **AI data extraction:** `modules/qualification/data_extractor.py`
- **Contractor commands:** `modules/contractor/command_handler.py`
- **Database queries:** Files in `/repos/` folder
- **Configuration:** `config.py` files in each module

## Benefits of This Structure

1. **Feature Isolation:** Each module is self-contained
2. **Easy Testing:** Test individual modules without affecting others
3. **Team Collaboration:** Different developers can work on different modules
4. **Clear Responsibility:** Easy to know where to make changes
5. **Scalability:** Simple to add new features as new modules

## Safe Files to Delete

These files/folders can be safely removed:
- `scripts/` - Contains old test data scripts
- `alembic/` - Database migration tool (not currently used)

## Important Notes

- Each module has its own `config.py` for settings
- Each module has a `README.md` explaining its purpose
- Database operations are kept separate from business logic
- Web endpoints are kept separate from processing logic
- Shared utilities go in `/utils/`

This structure follows the principle that **similar functionality should be grouped together** and **different concerns should be separated**.
