#services/pdf_service.py

import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER


async def generate_conversation_pdf(session, conversation_id: str) -> bytes:
    """
    Generate PDF transcript as bytes (for streaming response).
    Returns PDF bytes ready to be sent as response.
    """
    from repos.conversation_repo import ConversationRepo
    from repos.message_repo import MessageRepo
    from repos.conversation_data_repo import ConversationDataRepo
    from repos.contractor_repo import ContractorRepo
    from repos.models import Conversation, ConversationData

    # Initialize repos
    conv_repo = ConversationRepo(session)
    msg_repo = MessageRepo(session)
    data_repo = ConversationDataRepo(session)
    contractor_repo = ContractorRepo(session)

    # Fetch all data
    conversation = await session.get(Conversation, conversation_id)
    if not conversation:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation_data = await session.get(ConversationData, conversation_id)
    contractor = await contractor_repo.get_by_id(conversation.contractor_id)
    messages = await msg_repo.get_all_conversation_messages(conversation_id)

    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer,
                            pagesize=A4,
                            rightMargin=72,
                            leftMargin=72,
                            topMargin=72,
                            bottomMargin=18)

    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('CustomTitle',
                                 parent=styles['Title'],
                                 fontSize=20,
                                 textColor=colors.HexColor('#1a1a1a'),
                                 spaceAfter=30)

    heading_style = ParagraphStyle('SectionHeading',
                                   parent=styles['Heading2'],
                                   fontSize=14,
                                   textColor=colors.HexColor('#333333'),
                                   spaceBefore=20,
                                   spaceAfter=10,
                                   borderColor=colors.HexColor('#cccccc'),
                                   borderWidth=0,
                                   borderPadding=0)

    # Title
    job_title = conversation_data.job_title if conversation_data else "Lead"
    story.append(Paragraph(f"Lead Transcript: {job_title}", title_style))
    story.append(Spacer(1, 12))

    # Lead Summary Section
    story.append(Paragraph("Lead Summary", heading_style))

    if conversation_data and conversation_data.data_json:
        data = conversation_data.data_json
        summary_data = []

        # Build summary table
        if data.get('job_type'):
            summary_data.append(['Job Type:', data.get('job_type', '')])
        if data.get('property_type'):
            summary_data.append(
                ['Property Type:',
                 data.get('property_type', '')])
        if data.get('urgency'):
            summary_data.append(['Urgency:', data.get('urgency', '')])
        if data.get('address'):
            summary_data.append(['Address:', data.get('address', '')])
        if data.get('access'):
            summary_data.append(['Access:', data.get('access', '')])
        if data.get('notes'):
            # Wrap long notes
            notes = data.get('notes', '')
            if len(notes) > 100:
                notes = notes[:100] + '...'
            summary_data.append(['Notes:', notes])

        if summary_data:
            summary_table = Table(summary_data, colWidths=[2 * inch, 4 * inch])
            summary_table.setStyle(
                TableStyle([
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
            story.append(summary_table)

    story.append(Spacer(1, 20))

    # Conversation Details
    story.append(Paragraph("Conversation Details", heading_style))

    details_data = [
        ['Customer Phone:',
         conversation.customer_phone.replace('wa:', '')],
        [
            'Channel:',
            'WhatsApp' if 'wa:' in conversation.customer_phone else 'SMS'
        ], ['Status:', conversation.status],
        ['Started:',
         conversation.created_at.strftime('%d %b %Y at %H:%M')],
        [
            'Qualified:', 'Yes'
            if conversation_data and conversation_data.qualified else 'No'
        ]
    ]

    details_table = Table(details_data, colWidths=[2 * inch, 4 * inch])
    details_table.setStyle(
        TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
    story.append(details_table)

    story.append(Spacer(1, 20))

    # Full Transcript
    story.append(Paragraph("Full Conversation Transcript", heading_style))
    story.append(Spacer(1, 10))

    # Message styles
    customer_style = ParagraphStyle('CustomerMessage',
                                    parent=styles['Normal'],
                                    fontSize=10,
                                    leftIndent=0,
                                    textColor=colors.HexColor('#000080'))

    ai_style = ParagraphStyle('AIMessage',
                              parent=styles['Normal'],
                              fontSize=10,
                              leftIndent=20,
                              textColor=colors.HexColor('#006400'))

    # Add messages
    for direction, body in messages:
        if direction == 'inbound':
            story.append(Paragraph(f"<b>Customer:</b> {body}", customer_style))
        else:
            story.append(
                Paragraph(f"<b>{contractor.name}'s Assistant:</b> {body}",
                          ai_style))
        story.append(Spacer(1, 8))

    # Footer
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle('Footer',
                                  parent=styles['Normal'],
                                  fontSize=8,
                                  textColor=colors.gray,
                                  alignment=TA_CENTER)
    story.append(
        Paragraph(
            f"Generated on {datetime.utcnow().strftime('%d %b %Y at %H:%M UTC')} | "
            f"Conversation ID: {conversation_id[:8]}", footer_style))

    # Build PDF
    doc.build(story)

    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
