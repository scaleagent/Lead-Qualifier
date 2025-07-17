import uuid
from datetime import datetime
from sqlalchemy import (Column, String, DateTime, Integer, JSON, Boolean, Text,
                        ForeignKey)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    contractor_phone = Column(String, nullable=False)
    customer_phone = Column(String, nullable=False)
    status = Column(String, default="QUALIFYING", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime,
                        default=datetime.utcnow,
                        onupdate=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    sender = Column(String, nullable=False)
    receiver = Column(String, nullable=False)
    body = Column(Text, nullable=False)
    direction = Column(String, nullable=False)  # "inbound" / "outbound"
    timestamp = Column(DateTime, default=datetime.utcnow)


class ConversationData(Base):
    __tablename__ = "conversation_data"
    conversation_id = Column(
        String,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    contractor_phone = Column(String, nullable=False)
    customer_phone = Column(String, nullable=False)
    data_json = Column(JSON, nullable=False)
    last_updated = Column(DateTime,
                          default=datetime.utcnow,
                          onupdate=datetime.utcnow)
    qualified = Column(Boolean, default=False)
    job_title = Column(String, nullable=True)
