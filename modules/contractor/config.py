
# modules/contractor/config.py

# Contractor command patterns
TAKEOVER_PATTERNS = [
    r"^takeover\s+(.+)",
    r"^take\s+over\s+(.+)", 
    r"^taking\s+over\s+(.+)"
]

REACH_OUT_PATTERNS = [
    r"reach\s+out\s+to\s+(\+\d+)\s+(.+)",
    r"contact\s+(\+\d+)\s+(.+)",
    r"message\s+(\+\d+)\s+(.+)"
]

# Response messages
TAKEOVER_SUCCESS_MSG = "✅ I've marked that lead as taken over. They won't receive any more daily digest messages."
TAKEOVER_NOT_FOUND_MSG = "❌ I couldn't find a lead matching that description. Please be more specific."
TAKEOVER_MULTIPLE_MATCHES_MSG = "❌ Multiple leads match that description. Please be more specific."

REACH_OUT_SUCCESS_MSG = "✅ I've sent your message to {phone}."
REACH_OUT_ERROR_MSG = "❌ There was an error sending your message. Please try again."
