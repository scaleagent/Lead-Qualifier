
# modules/messaging/message_classifier.py

import logging
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


class MessageClassifier:
    """Handles message classification using OpenAI"""
    
    @staticmethod
    async def classify_message(message_text: str, history_string: str) -> str:
        """
        Classify incoming message as NEW, CONTINUATION, or UNSURE using OpenAI.
        """
        logger.info(f"🧠 STARTING message classification...")
        logger.debug(f"   Message: '{message_text}'")
        logger.debug(f"   History: {history_string!r}")
        
        # Extract the last AI message if it exists
        last_ai_message = ""
        if history_string:
            lines = history_string.strip().split('\n')
            for line in reversed(lines):
                if line.startswith("AI:"):
                    last_ai_message = line.replace("AI:", "").strip()
                    break

        system_prompt = (
            "You are a message classifier for a contractor's AI assistant. Your job is to determine if an incoming message is part of an existing conversation or starting a new one.\n\n"
            "Classification rules:\n"
            "1. CONTINUATION - Use when:\n"
            "   - The message answers or responds to a question from the conversation history\n"
            "   - The message provides information related to the ongoing topic\n"
            "   - The message acknowledges the previous message (e.g., 'thanks', 'yes', 'sure') AND provides relevant information\n"
            "   - The message corrects or adds to previously discussed information\n"
            "2. NEW - Use when:\n"
            "   - The message asks about a completely different job/service\n"
            "   - The message mentions a new address/location not previously discussed\n"
            "   - The message explicitly states starting over or new request\n"
            "   - The message is unrelated to any previous questions or topics\n"
            "3. UNSURE - Use when:\n"
            "   - The message is ambiguous or could go either way\n"
            "   - Simple greetings without context\n"
            "   - Very short responses that don't clearly relate to previous messages\n\n"
            "IMPORTANT: Focus on whether the message logically follows from or responds to the conversation history.\n"
            "Respond with exactly one word: NEW, CONTINUATION, or UNSURE."
        )

        user_prompt = f"""Analyze this message classification task:

CONVERSATION HISTORY:
{history_string if history_string else "[No previous messages]"}

LAST QUESTION/STATEMENT FROM AI:
{last_ai_message if last_ai_message else "[No previous AI message]"}

NEW INCOMING MESSAGE FROM CUSTOMER:
"{message_text}"

ANALYSIS CHECKLIST:
1. Does the new message answer or respond to the last AI question? 
2. Does the new message provide information that was requested?
3. Is the new message about the same topic/job as the conversation history?
4. Does the new message explicitly mention a different job or location?

Based on the analysis, classify as: NEW, CONTINUATION, or UNSURE"""

        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": user_prompt
                }],
                temperature=0,
                max_tokens=10,
            )

            result = resp.choices[0].message.content.strip().upper()

            # Failsafe: ensure we only return valid classifications
            if result not in ["NEW", "CONTINUATION", "UNSURE"]:
                logger.warning(f"⚠️ Invalid classification result: {result}, defaulting to UNSURE")
                return "UNSURE"

            logger.info(f"🧠 CLASSIFICATION RESULT: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Classification error: {e}")
            return "UNSURE"
    
    @staticmethod
    def is_reach_out_command(message: str) -> tuple[bool, str | None]:
        """Check if message is a 'reach out to' command and extract phone number"""
        import re
        
        match = re.match(r'^\s*reach out to (\+\d{10,15})\s*$', message, re.IGNORECASE)
        if match:
            return True, match.group(1)
        return False, None
    
    @staticmethod
    def is_after_reach_out(last_messages: list) -> bool:
        """Check if this is the first response after a reach out command"""
        if (len(last_messages) == 1 and 
            last_messages[0][0] == "outbound" and 
            "To get started, please tell me the type of job" in last_messages[0][1]):
            return True
        return False
