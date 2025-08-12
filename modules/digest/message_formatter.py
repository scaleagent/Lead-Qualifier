
# modules/digest/message_formatter.py - SMS message formatting for digest

from .pdf_generator import PDFGenerator


class MessageFormatter:
    """Format digest messages for SMS delivery"""
    
    @staticmethod
    def format_qualified_lead_sms(lead) -> str:
        """Format a qualified lead for SMS with all key details"""
        data = lead.data_json

        # Extract and truncate fields to fit SMS limits
        job_type = (data.get('job_type', 'Unknown job') or 'Unknown job')[:30]
        address = (data.get('address', 'No address') or 'No address')[:40]
        urgency = (data.get('urgency', 'Not specified') or 'Not specified')[:20]
        access = (data.get('access', '') or '')[:30]

        # Generate PDF URL
        pdf_url = PDFGenerator.generate_pdf_url(lead.conversation_id)

        # Build SMS message
        parts = [f"🏗 {lead.job_title or job_type}", f"📍 {address}", f"⏰ {urgency}"]

        if access:
            parts.append(f"🔑 {access}")

        parts.append(f"📄 {pdf_url}")

        text = "\n".join(parts)

        # Ensure SMS doesn't exceed length limits
        if len(text) > 300:
            address = address[:20] + "..."
            if access:
                access = access[:15] + "..."

            parts = [f"🏗 {lead.job_title or job_type}", f"📍 {address}", f"⏰ {urgency}"]
            if access:
                parts.append(f"🔑 {access}")
            parts.append(f"📄 {pdf_url}")

            text = "\n".join(parts)

        return text

    @staticmethod
    def format_ongoing_leads_sms(ongoing_leads) -> str:
        """Format ongoing leads into a compact SMS"""
        if len(ongoing_leads) <= 3:
            titles = []
            for lead in ongoing_leads:
                title = (lead.job_title or "Untitled")[:25]
                titles.append(title)

            text = f"🔄 Ongoing: {' • '.join(titles)}"
        else:
            text = f"🔄 {len(ongoing_leads)} ongoing leads in progress"

        if len(text) > 160:
            text = f"🔄 {len(ongoing_leads)} ongoing leads"

        return text
