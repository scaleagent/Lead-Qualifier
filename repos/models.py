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
from sqlalchemy.ext.mutable import MutableDict  # ← added for JSON mutability

Base = declarative_base()


class Contractor(Base):
    __tablename__ = "contractors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    phone = Column(String, unique=True, nullable=False)
    assistant_phone = Column(String, unique=True, nullable=False)
    address = Column(String, nullable=True)
    contractor_profile = Column(String, nullable=False, default="general_contractor")
    digest_config = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime,
                        default=datetime.utcnow,
                        onupdate=datetime.utcnow)

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
    closed_at = Column(DateTime(timezone=True), nullable=True)
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
    # Use MutableDict so nested JSON changes are tracked by SQLAlchemy
    data_json = Column(MutableDict.as_mutable(JSON), nullable=False)
    last_updated = Column(DateTime,
                          default=datetime.utcnow,
                          onupdate=datetime.utcnow)
    qualified = Column(Boolean, default=False)
    job_title = Column(String, nullable=True)
    opt_out_of_digest = Column(Boolean, default=False, nullable=False)
    last_digest_sent = Column(DateTime, nullable=True)
