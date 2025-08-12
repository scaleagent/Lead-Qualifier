
# Contractor Module

This module handles all contractor-related functionality including command processing, contractor identification, and contractor-specific business logic.

## Components

### ContractorService
Main service for contractor operations including identification and management.

**Key Methods:**
- `identify_contractor()` - Determine which contractor a message is associated with
- `get_contractor_by_phone()` - Find contractor by phone number

### CommandHandler
Processes contractor-specific commands like "takeover" and "reach out to".

**Key Methods:**
- `handle_takeover_command()` - Process takeover commands from contractors
- `handle_reach_out_command()` - Process "reach out to" commands
- `is_contractor_command()` - Determine if message contains a contractor command

## Usage

```python
from modules.contractor import ContractorService, CommandHandler

# Initialize services
contractor_service = ContractorService(session)
command_handler = CommandHandler(session)

# Process contractor message
contractor = await contractor_service.identify_contractor(phone_number)
if contractor:
    response = await command_handler.handle_takeover_command(message, contractor)
```

## Command Examples

- **Takeover**: "takeover kitchen renovation" - Marks a lead as taken over by contractor
- **Reach out**: "reach out to +447742001014 about bathroom work" - Initiates contact with potential lead
