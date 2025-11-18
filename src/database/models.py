"""Database models for Zeus Splunk Query LLM."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Float, JSON
from sqlalchemy.orm import relationship

from src.database.database import Base


class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    queries = relationship("Query", back_populates="user", cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="creator", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class Query(Base):
    """Query history model."""

    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    instruction = Column(Text, nullable=False)
    input_text = Column(Text, default="")
    generated_query = Column(Text, nullable=False)
    is_clarification = Column(Boolean, default=False)
    alternatives = Column(Text, nullable=True)  # JSON string of alternative queries
    temperature = Column(Float, default=0.1)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User", back_populates="queries")
    feedback = relationship("Feedback", back_populates="query", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Query(id={self.id}, user_id={self.user_id}, created_at='{self.created_at}')>"


class Feedback(Base):
    """Feedback model for query ratings and corrections."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=False, unique=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    rating = Column(String(10), nullable=False)  # 'good' or 'bad'
    corrected_query = Column(Text, nullable=True)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    query = relationship("Query", back_populates="feedback")
    user = relationship("User", back_populates="feedback")

    def __repr__(self):
        return f"<Feedback(id={self.id}, query_id={self.query_id}, rating='{self.rating}')>"


class TrainingJob(Base):
    """Training job model for tracking model retraining jobs."""

    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(String(20), nullable=False, index=True)  # pending, running, completed, failed
    config = Column(JSON, nullable=True)  # Training configuration as JSON
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    model_output_path = Column(String(500), nullable=True)
    metrics = Column(JSON, nullable=True)  # Training metrics as JSON
    error_message = Column(Text, nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    creator = relationship("User", back_populates="training_jobs")

    def __repr__(self):
        return f"<TrainingJob(id={self.id}, status='{self.status}', created_by={self.created_by})>"
