
# modules/qualification/config.py

# OpenAI model configuration
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.4  # Increased for more variability
OPENAI_MAX_TOKENS = 300

# Response patterns for customer confirmation
AFFIRMATIVE_PATTERNS = [
    r"^(yes|yep|yeah|correct|that is correct)\b"
]

NEGATIVE_PATTERNS = [
    r"^(no|nope|all done|that'?s (all|everything))\b"
]

# Note: REQUIRED_FIELDS is now dynamic based on contractor profiles
# See modules/qualification/contractor_profiles.py for profile definitions
