
# modules/qualification/data_extractor.py

import json
import logging
from typing import Dict
from openai import OpenAI
from .config import OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS

logger = logging.getLogger(__name__)


class DataExtractor:
    """Handles extraction of qualification data from conversation history using OpenAI."""
    
    def __init__(self, openai_client: OpenAI):
        self.openai = openai_client
    
    async def extract_qualification_data(self, history_string: str) -> Dict[str, str]:
        """
        Extract qualification data from conversation history using OpenAI.
        
        Args:
            history_string: Formatted conversation history
            
        Returns:
            Dictionary containing extracted qualification data
        """
        logger.debug(f"Extracting qualification data from {len(history_string)} chars of history")
        
        system_prompt = (
            "Extract exactly these fields from the customer's messages: job_type, property_type, urgency, address, access, notes.\n"
            "- If NOT mentioned by the customer, set value to empty string.\n"
            "- Do NOT guess or infer information.\n"
            "- Put all other customer comments into 'notes'.\n"
            "Respond ONLY with a JSON object with exactly these six keys."
        )

        try:
            resp = self.openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Conversation:\n{history_string}"}
                ],
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS,
            )

            data = json.loads(resp.choices[0].message.content)
            logger.debug(f"Successfully extracted data: {list(data.keys())}")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"OpenAI response: {resp.choices[0].message.content}")
            data = {}
        except Exception as e:
            logger.error(f"Error extracting qualification data: {e}")
            data = {}

        # Ensure notes field exists (other fields will be handled by qualification service)
        data.setdefault("notes", "")

        return data

    async def apply_correction_data(self, current: Dict[str, str], correction: str) -> Dict[str, str]:
        """
        Apply user corrections to existing qualification data using OpenAI.
        
        Args:
            current: Current qualification data dictionary
            correction: Customer's correction message
            
        Returns:
            Updated qualification data dictionary
        """
        logger.debug(f"Applying correction: '{correction}' to existing data")
        
        system_prompt = (
            "You are a JSON assistant. Given existing job data and a user correction, "
            "return the updated JSON with the same six keys only."
        )

        try:
            resp = self.openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Existing data: {json.dumps(current)}\nCorrection: {correction}\nRespond ONLY with the full updated JSON."
                    }
                ],
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS,
            )

            updated = json.loads(resp.choices[0].message.content)
            logger.debug(f"Successfully applied correction")
            
        except json.JSONDecodeError as e:
            logger.error(f"Correction JSON parse error: {e}")
            logger.error(f"OpenAI response: {resp.choices[0].message.content}")
            updated = current
        except Exception as e:
            logger.error(f"Error applying correction: {e}")
            updated = current

        # Ensure notes field exists and preserve all existing keys
        updated.setdefault("notes", current.get("notes", ""))
        for key in current:
            updated.setdefault(key, current[key])

        return updated
