#repos/models.py

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    JSON,
    Boolean,
    Text,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Contractor(Base):
    __tablename__ = "contractors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    phone = Column(String, unique=True, nullable=False)
    address = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime,
                        default=datetime.utcnow,
                        onupdate=datetime.utcnow)
    digest_config =Column(JSON, nullable=False, default=dict)

    # backref from Conversation:
    conversations = relationship("Conversation", back_populates="contractor")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    contractor_id = Column(Integer,
                           ForeignKey("contractors.id"),
                           nullable=False)
    customer_phone = Column(String, nullable=False)
    status = Column(String, default="QUALIFYING", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime,
                        default=datetime.utcnow,
                        onupdate=datetime.utcnow)

    contractor = relationship("Contractor", back_populates="conversations")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String,
                             ForeignKey("conversations.id",
                                        ondelete="CASCADE"),
                             nullable=True)
    sender = Column(String, nullable=False)
    receiver = Column(String, nullable=False)
    body = Column(Text, nullable=False)
    direction = Column(String, nullable=False)  # "inbound"/"outbound"
    timestamp = Column(DateTime, default=datetime.utcnow)


class ConversationData(Base):
    __tablename__ = "conversation_data"

    conversation_id = Column(String,
                             ForeignKey("conversations.id",
                                        ondelete="CASCADE"),
                             primary_key=True)
    contractor_id = Column(Integer,
                           ForeignKey("contractors.id"),
                           nullable=False)
    customer_phone = Column(String, nullable=False)
    data_json = Column(JSON, nullable=False)
    last_updated = Column(DateTime,
                          default=datetime.utcnow,
                          onupdate=datetime.utcnow)
    qualified = Column(Boolean, default=False)
    job_title = Column(String, nullable=True)
