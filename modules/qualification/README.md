
# Qualification Module

This module handles the lead qualification process for incoming customer messages. It extracts relevant information from conversations and manages the qualification flow state.

## Components

### QualificationService
Main orchestrator for the qualification process. Handles the overall flow of qualifying a lead from initial contact to completion.

**Key Methods:**
- `process_qualification()` - Main entry point for processing qualification messages
- `handle_confirmation()` - Process customer confirmations of extracted data
- `handle_notes_collection()` - Manage additional notes collection phase

### DataExtractor
Uses OpenAI to extract structured qualification data from conversation history.

**Key Methods:**
- `extract_qualification_data()` - Extract job details from conversation
- `apply_correction_data()` - Apply customer corrections to existing data

### FlowManager
Manages conversation state transitions and determines next steps in the qualification flow.

**Key Methods:**
- `is_qualified()` - Check if all required fields are present
- `get_missing_fields()` - Identify which fields still need to be collected
- `generate_prompt()` - Create appropriate prompts for missing information

## Usage

```python
from modules.qualification import QualificationService

# Initialize service
qualification_service = QualificationService(session, openai_client)

# Process a qualification message
response = await qualification_service.process_qualification(
    conversation=conversation,
    message_body=message_body,
    contractor=contractor
)
```

## Configuration

Required fields for qualification are defined in `config.py`:
- job_type
- property_type  
- urgency
- address
- access

Additional notes are collected in a separate field but are not required for qualification completion.
