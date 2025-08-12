
# modules/messaging/channel_manager.py

import logging
from .config import TWILIO_NUMBER, WA_SANDBOX_NUMBER

logger = logging.getLogger(__name__)


class ChannelManager:
    """Handles channel detection and phone number normalization"""
    
    @staticmethod
    def detect_channel(from_number: str) -> tuple[bool, str]:
        """
        Detect if message is from WhatsApp and return clean phone number
        
        Returns:
            tuple: (is_whatsapp, clean_phone_number)
        """
        is_whatsapp = from_number.startswith("whatsapp:")
        clean_phone = from_number.split(":", 1)[1] if is_whatsapp else from_number
        
        logger.info(f"📱 CHANNEL DETECTION:")
        logger.info(f"  Is WhatsApp: {is_whatsapp}")
        logger.info(f"  Clean phone: {clean_phone}")
        
        return is_whatsapp, clean_phone
    
    @staticmethod
    def normalize_phone_for_db(phone: str, is_whatsapp: bool) -> str:
        """
        Normalize phone numbers for database storage with channel separation:
        - WhatsApp: wa:+447742001014
        - SMS: +447742001014
        """
        # Remove any existing whatsapp: prefix first
        clean_phone = phone.split(":", 1)[1] if phone.startswith("whatsapp:") else phone

        if is_whatsapp:
            return f"wa:{clean_phone}"
        else:
            return clean_phone

    @staticmethod
    def get_system_number_for_db(is_whatsapp: bool) -> str:
        """Get the system's number in the correct format for database storage"""
        if is_whatsapp:
            # Remove whatsapp: prefix and add wa: prefix
            clean_number = WA_SANDBOX_NUMBER.split(
                ":", 1)[1] if WA_SANDBOX_NUMBER.startswith(
                    "whatsapp:") else WA_SANDBOX_NUMBER
            return f"wa:{clean_number}"
        else:
            return TWILIO_NUMBER
    
    @staticmethod
    def get_response_phone(original_phone: str) -> str:
        """
        Determine where to send response - either test phone or original
        """
        from .config import TEST_MODE, DIGEST_TEST_PHONE
        
        if TEST_MODE and DIGEST_TEST_PHONE:
            logger.info(f"📱 TEST MODE: Responses will go to {DIGEST_TEST_PHONE} instead of {original_phone}")
            return DIGEST_TEST_PHONE
        return original_phone
