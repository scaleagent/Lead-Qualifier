# modules/qualification/qualification_service.py

import json
import logging
from typing import Dict, List, Optional
from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from .contractor_profiles import (
    get_contractor_profile,
    get_active_categories_for_profile,
    get_required_categories_for_profile,
    get_category_config,
    MASTER_DATA_CATEGORIES
)

logger = logging.getLogger(__name__)

class QualificationService:
    """Service for qualifying leads based on contractor profiles."""

    def __init__(self, session: AsyncSession, openai_client):
        self.session = session
        self.openai_client = openai_client
        self.data_extractor = DataExtractor()
        self.flow_manager = FlowManager()

    def get_profile_specific_prompt(self, contractor_profile: str) -> str:
        """Generate AI assistant prompt based on contractor profile."""
        profile_config = get_contractor_profile(contractor_profile)
        active_categories = get_active_categories_for_profile(contractor_profile)

        # Build list of information to collect
        info_to_collect = []
        for category in active_categories:
            category_config = get_category_config(category)
            if category_config:
                info_to_collect.append(f"- {category_config['description']}")

        prompt = f"""You are a professional assistant working for a {profile_config['name']}.

Your role is to gather comprehensive information from potential customers to help your contractor provide accurate quotes and service. You are experienced in {profile_config['name'].lower()} work and understand what information is most valuable.

PERSONALITY: {profile_config['personality']}

INFORMATION TO COLLECT:
{chr(10).join(info_to_collect)}

GUIDELINES:
- Be conversational and professional, like an experienced assistant
- Ask follow-up questions to get complete information
- Only give advice if it helps you gather more information
- Do not provide safety warnings or technical advice
- Focus on understanding the customer's needs and situation
- Ask one clear question at a time
- Be helpful and personable while staying focused on information gathering

Remember: Your goal is to collect comprehensive information so your contractor can provide the best possible service and accurate pricing."""

        return prompt

    async def extract_qualification_data(self, conversation_history: str, contractor_profile: str) -> Dict:
        """Extract qualification data from conversation using profile-specific categories."""
        active_categories = get_active_categories_for_profile(contractor_profile)

        # Build extraction prompt with only active categories
        categories_description = {}
        for category in active_categories:
            config = get_category_config(category)
            if config:
                categories_description[category] = config['description']

        system_prompt = f"""Extract information from the customer's messages into these specific categories:

{json.dumps(categories_description, indent=2)}

Rules:
- Only extract information explicitly mentioned by the customer
- If not mentioned, set value to empty string
- Put any additional customer comments into 'notes'
- Respond ONLY with a JSON object containing these exact keys: {', '.join(active_categories + ['notes'])}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Conversation:\n{conversation_history}"}
                ],
                temperature=0.3,  # Slight variability for more natural responses
                max_tokens=500
            )

            extracted_data = json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to extract qualification data: {e}")
            extracted_data = {}

        # Ensure all active categories are present
        for category in active_categories + ['notes']:
            extracted_data.setdefault(category, "")

        return extracted_data

    def is_qualified(self, data: Dict, contractor_profile: str) -> bool:
        """Check if lead is qualified based on profile requirements."""
        required_categories = get_required_categories_for_profile(contractor_profile)
        return all(data.get(category) for category in required_categories)

    async def generate_response(self, message: str, conversation_history: str, contractor_profile: str, current_data: Dict) -> str:
        """Generate AI response using profile-specific personality and focus."""
        system_prompt = self.get_profile_specific_prompt(contractor_profile)

        # Add context about what information we already have
        active_categories = get_active_categories_for_profile(contractor_profile)
        collected_info = []
        missing_info = []

        for category in active_categories:
            config = get_category_config(category)
            if config and current_data.get(category): # Use current_data here
                collected_info.append(f"✓ {config['description']}")
            elif config:
                missing_info.append(f"? {config['description']}")

        context = f"""
CURRENT CONVERSATION STATUS:
Information collected: {', '.join(collected_info) if collected_info else 'None yet'}
Still needed: {', '.join(missing_info) if missing_info else 'All required info collected'}

CONVERSATION HISTORY:
{conversation_history}

LATEST MESSAGE: {message}

Respond naturally and professionally, focusing on gathering the missing information."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.4,  # Adding variability for more personable responses
                max_tokens=200
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate AI response: {e}")
            return "Thanks for that information. Could you tell me more about what work you need done?"