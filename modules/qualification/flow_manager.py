
# modules/qualification/flow_manager.py

import re
import logging
from typing import Dict, List
from .config import REQUIRED_FIELDS, AFFIRMATIVE_PATTERNS, NEGATIVE_PATTERNS

logger = logging.getLogger(__name__)


class FlowManager:
    """Manages the qualification flow state and determines next steps."""
    
    def is_qualified(self, data: Dict[str, str]) -> bool:
        """Check if all required fields are filled."""
        qualified = all(data.get(k) for k in REQUIRED_FIELDS)
        logger.debug(f"Qualification check: {qualified} (filled: {sum(1 for k in REQUIRED_FIELDS if data.get(k))} of {len(REQUIRED_FIELDS)})")
        return qualified
    
    def get_missing_fields(self, data: Dict[str, str]) -> List[str]:
        """Get list of missing required fields."""
        missing = [k for k in REQUIRED_FIELDS if not data.get(k)]
        logger.debug(f"Missing fields: {missing}")
        return missing
    
    def generate_prompt(self, data: Dict[str, str]) -> str:
        """Generate appropriate prompt based on current qualification state."""
        missing = self.get_missing_fields(data)
        
        if not missing:
            logger.debug("All fields collected - should show summary")
            return None
        
        # If no fields collected yet, ask for job type
        if all(data[k] == "" for k in REQUIRED_FIELDS):
            return "Please provide your job type."
        
        # Ask for up to 2 missing fields at once
        next_fields = missing[:2]
        labels = [f.replace("_", " ") for f in next_fields]
        
        if len(labels) == 1:
            prompt = f"Please provide your {labels[0]}."
        else:
            prompt = f"Please provide your {labels[0]} and {labels[1]}."
        
        logger.debug(f"Generated prompt for missing fields: {next_fields}")
        return prompt
    
    def generate_summary(self, data: Dict[str, str]) -> str:
        """Generate summary of collected qualification data."""
        bullets = [
            f"• {f.replace('_', ' ').title()}: {data[f]}"
            for f in REQUIRED_FIELDS
        ]
        summary = "Here's what I have so far:\n" + "\n".join(bullets) + "\nIs that correct?"
        logger.debug("Generated qualification summary")
        return summary
    
    def is_affirmative(self, text: str) -> bool:
        """Check if text indicates affirmative response."""
        text_lower = text.strip().lower()
        for pattern in AFFIRMATIVE_PATTERNS:
            if re.match(pattern, text_lower):
                logger.debug(f"Detected affirmative response: '{text}'")
                return True
        return False
    
    def is_negative(self, text: str) -> bool:
        """Check if text indicates negative/completion response."""
        text_lower = text.strip().lower()
        for pattern in NEGATIVE_PATTERNS:
            if re.match(pattern, text_lower):
                logger.debug(f"Detected negative response: '{text}'")
                return True
        return False
