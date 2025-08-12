
# modules/qualification/config.py

# Required fields that must be collected for a lead to be considered qualified
REQUIRED_FIELDS = ["job_type", "property_type", "urgency", "address", "access"]

# OpenAI model configuration
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0
OPENAI_MAX_TOKENS = 300

# Response patterns for customer confirmation
AFFIRMATIVE_PATTERNS = [
    r"^(yes|yep|yeah|correct|that is correct)\b"
]

NEGATIVE_PATTERNS = [
    r"^(no|nope|all done|that'?s (all|everything))\b"
]
