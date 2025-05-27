"""
Database Models for Document Processor Agent
SQLAlchemy ORM models for storing documents, processing results, and metadata
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json
import uuid

from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, Boolean, 
    JSON, ForeignKey, Index, UniqueConstraint, CheckConstraint,
    create_engine, MetaData
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.types import TypeDecorator, VARCHAR

# Create base class
Base = declarative_base()

# Custom UUID type that works with both PostgreSQL and SQLite
class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(36), storing as stringified hex values.
    """
    impl = VARCHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(VARCHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value


class TimestampMixin:
    """Mixin for adding timestamp fields"""
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)


class Document(Base, TimestampMixin):
    """Main document table storing uploaded documents and metadata"""
    __tablename__ = 'documents'
    
    # Primary key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    
    # File information
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=True)  # Storage path
    file_size = Column(Integer, nullable=False)  # Size in bytes
    file_type = Column(String(50), nullable=False)  # File extension
    mime_type = Column(String(100), nullable=True)
    
    # Processing status
    status = Column(String(50), nullable=False, default='uploaded')  # uploaded, processing, completed, failed
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Content metadata
    title = Column(String(500), nullable=True)
    author = Column(String(255), nullable=True)
    subject = Column(String(500), nullable=True)
    language = Column(String(10), nullable=True, default='en')
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    character_count = Column(Integer, nullable=True)
    
    # Content fingerprint for deduplication
    content_hash = Column(String(64), nullable=True, unique=True)
    
    # Processing configuration used
    processing_config = Column(JSON, nullable=True)
    
    # Error information
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    processing_results = relationship("ProcessingResult", back_populates="document", cascade="all, delete-orphan")
    queries = relationship("QueryResult", back_populates="document")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_filename', 'filename'),
        Index('idx_documents_status', 'status'),
        Index('idx_documents_content_hash', 'content_hash'),
        Index('idx_documents_created_at', 'created_at'),
        Index('idx_documents_file_type', 'file_type'),
    )
    
    def __repr__(self):
        return f"<Document(id='{self.id}', filename='{self.filename}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'file_size_mb': round(self.file_size / (1024 * 1024), 2),
            'file_type': self.file_type,
            'mime_type': self.mime_type,
            'status': self.status,
            'title': self.title,
            'author': self.author,
            'subject': self.subject,
            'language': self.language,
            'page_count': self.page_count,
            'word_count': self.word_count,
            'character_count': self.character_count,
            'content_hash': self.content_hash,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'processing_started_at': self.processing_started_at.isoformat() if self.processing_started_at else None,
            'processing_completed_at': self.processing_completed_at.isoformat() if self.processing_completed_at else None,
            'error_message': self.error_message
        }


class DocumentChunk(Base, TimestampMixin):
    """Document chunks for vector storage and retrieval"""
    __tablename__ = 'document_chunks'
    
    # Primary key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to document
    document_id = Column(GUID(), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    
    # Chunk information
    chunk_index = Column(Integer, nullable=False)  # Order within document
    chunk_text = Column(Text, nullable=False)
    chunk_length = Column(Integer, nullable=False)
    
    # Position in original document
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    page_number = Column(Integer, nullable=True)
    section = Column(String(255), nullable=True)
    
    # Vector embedding information
    embedding_model = Column(String(100), nullable=True)
    embedding_dimension = Column(Integer, nullable=True)
    vector_id = Column(String(255), nullable=True)  # ID in vector database
    
    # Chunk metadata
    chunk_metadata = Column(JSON, nullable=True)
    
    # Processing information
    chunk_strategy = Column(String(50), nullable=True)  # recursive, sentence, etc.
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    # Indexes
    __table_args__ = (
        Index('idx_chunks_document_id', 'document_id'),
        Index('idx_chunks_chunk_index', 'document_id', 'chunk_index'),
        Index('idx_chunks_vector_id', 'vector_id'),
        UniqueConstraint('document_id', 'chunk_index', name='uq_document_chunk_index'),
    )
    
    def __repr__(self):
        return f"<DocumentChunk(id='{self.id}', document_id='{self.document_id}', index={self.chunk_index})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'document_id': str(self.document_id),
            'chunk_index': self.chunk_index,
            'chunk_text': self.chunk_text,
            'chunk_length': self.chunk_length,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'page_number': self.page_number,
            'section': self.section,
            'embedding_model': self.embedding_model,
            'embedding_dimension': self.embedding_dimension,
            'vector_id': self.vector_id,
            'chunk_metadata': self.chunk_metadata,
            'chunk_strategy': self.chunk_strategy,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ProcessingResult(Base, TimestampMixin):
    """Store results from various processing agents"""
    __tablename__ = 'processing_results'
    
    # Primary key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to document
    document_id = Column(GUID(), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    
    # Processing information
    agent_name = Column(String(100), nullable=False)  # document_loader, parser, etc.
    agent_version = Column(String(50), nullable=True)
    processing_stage = Column(String(50), nullable=False)  # load, parse, embed, validate, etc.
    
    # Results
    success = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=True)
    execution_time = Column(Float, nullable=True)  # Processing time in seconds
    
    # Result data
    result_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Processing metadata
    processing_metadata = Column(JSON, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="processing_results")
    
    # Indexes
    __table_args__ = (
        Index('idx_processing_results_document_id', 'document_id'),
        Index('idx_processing_results_agent', 'agent_name'),
        Index('idx_processing_results_stage', 'processing_stage'),
        Index('idx_processing_results_success', 'success'),
    )
    
    def __repr__(self):
        return f"<ProcessingResult(id='{self.id}', agent='{self.agent_name}', success={self.success})>"


class QueryResult(Base, TimestampMixin):
    """Store Q&A query results and user interactions"""
    __tablename__ = 'query_results'
    
    # Primary key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    
    # Session information
    session_id = Column(String(255), nullable=True)  # For tracking conversations
    user_id = Column(String(255), nullable=True)  # User identifier
    
    # Query information
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=True)  # question, summary, analysis, etc.
    
    # Response information
    answer_text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    response_time = Column(Float, nullable=True)  # Response time in seconds
    
    # Context information
    context_documents_used = Column(Integer, nullable=True)
    max_context_length = Column(Integer, nullable=True)
    
    # Source documents
    document_id = Column(GUID(), ForeignKey('documents.id', ondelete='SET NULL'), nullable=True)
    source_chunks = Column(JSON, nullable=True)  # List of chunk IDs used
    
    # AI model information
    model_used = Column(String(100), nullable=True)
    model_parameters = Column(JSON, nullable=True)
    
    # User feedback
    user_rating = Column(Integer, nullable=True)  # 1-5 rating
    user_feedback = Column(Text, nullable=True)
    is_helpful = Column(Boolean, nullable=True)
    
    # Validation results
    validation_status = Column(String(50), nullable=True)  # PASS, FAIL, NEEDS_REVIEW
    validation_confidence = Column(Float, nullable=True)
    validation_results = Column(JSON, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="queries")
    
    # Indexes
    __table_args__ = (
        Index('idx_query_results_session_id', 'session_id'),
        Index('idx_query_results_user_id', 'user_id'),
        Index('idx_query_results_document_id', 'document_id'),
        Index('idx_query_results_created_at', 'created_at'),
        Index('idx_query_results_confidence', 'confidence'),
    )
    
    def __repr__(self):
        return f"<QueryResult(id='{self.id}', query='{self.query_text[:50]}...', confidence={self.confidence})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': str(self.id),
            'session_id': self.session_id,
            'user_id': self.user_id,
            'query_text': self.query_text,
            'query_type': self.query_type,
            'answer_text': self.answer_text,
            'confidence': self.confidence,
            'response_time': self.response_time,
            'context_documents_used': self.context_documents_used,
            'document_id': str(self.document_id) if self.document_id else None,
            'model_used': self.model_used,
            'user_rating': self.user_rating,
            'user_feedback': self.user_feedback,
            'is_helpful': self.is_helpful,
            'validation_status': self.validation_status,
            'validation_confidence': self.validation_confidence,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ValidationResult(Base, TimestampMixin):
    """Store detailed validation results"""
    __tablename__ = 'validation_results'
    
    # Primary key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to query result
    query_result_id = Column(GUID(), ForeignKey('query_results.id', ondelete='CASCADE'), nullable=False)
    
    # Validation information
    validation_strategy = Column(String(50), nullable=False)  # completeness, accuracy, etc.
    validation_level = Column(String(50), nullable=False)  # quick, standard, comprehensive
    
    # Results
    passed = Column(Boolean, nullable=False)
    score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    execution_time = Column(Float, nullable=True)
    
    # Feedback and suggestions
    feedback = Column(Text, nullable=True)
    suggestions = Column(JSON, nullable=True)  # List of improvement suggestions
    
    # Detailed results
    detailed_results = Column(JSON, nullable=True)
    
    # Human review
    requires_human_review = Column(Boolean, nullable=False, default=False)
    human_review_completed = Column(Boolean, nullable=False, default=False)
    human_reviewer_id = Column(String(255), nullable=True)
    human_review_notes = Column(Text, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_validation_results_query_id', 'query_result_id'),
        Index('idx_validation_results_strategy', 'validation_strategy'),
        Index('idx_validation_results_passed', 'passed'),
        Index('idx_validation_results_human_review', 'requires_human_review'),
    )
    
    def __repr__(self):
        return f"<ValidationResult(id='{self.id}', strategy='{self.validation_strategy}', passed={self.passed})>"


class SystemMetrics(Base, TimestampMixin):
    """Store system performance and usage metrics"""
    __tablename__ = 'system_metrics'
    
    # Primary key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    
    # Metric information
    metric_name = Column(String(100), nullable=False)
    metric_category = Column(String(50), nullable=False)  # performance, usage, error, etc.
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)  # seconds, bytes, count, etc.
    
    # Context information
    context_data = Column(JSON, nullable=True)
    
    # Time period
    period_start = Column(DateTime(timezone=True), nullable=True)
    period_end = Column(DateTime(timezone=True), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_system_metrics_name', 'metric_name'),
        Index('idx_system_metrics_category', 'metric_category'),
        Index('idx_system_metrics_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<SystemMetrics(name='{self.metric_name}', value={self.metric_value}, unit='{self.metric_unit}')>"


class UserSession(Base, TimestampMixin):
    """Track user sessions and interactions"""
    __tablename__ = 'user_sessions'
    
    # Primary key
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    
    # Session information
    session_id = Column(String(255), nullable=False, unique=True)
    user_id = Column(String(255), nullable=True)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    
    # Session activity
    started_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    last_activity_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)
    
    # Session data
    session_data = Column(JSON, nullable=True)
    
    # Activity counters
    documents_uploaded = Column(Integer, nullable=False, default=0)
    queries_made = Column(Integer, nullable=False, default=0)
    validations_requested = Column(Integer, nullable=False, default=0)
    
    # Indexes
    __table_args__ = (
        Index('idx_user_sessions_session_id', 'session_id'),
        Index('idx_user_sessions_user_id', 'user_id'),
        Index('idx_user_sessions_started_at', 'started_at'),
    )
    
    def __repr__(self):
        return f"<UserSession(id='{self.id}', session_id='{self.session_id}', user_id='{self.user_id}')>"


# Database utility functions
def create_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)


def drop_tables(engine):
    """Drop all tables from the database"""
    Base.metadata.drop_all(engine)


def get_table_names():
    """Get list of all table names"""
    return [table.name for table in Base.metadata.tables.values()]


# Database constraints and checks
def add_constraints():
    """Add additional constraints that aren't supported in table definitions"""
    # This would be called during database initialization
    # Add any complex constraints here
    pass


# Example usage and testing
if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create in-memory SQLite database for testing
    engine = create_engine("sqlite:///:memory:", echo=True)
    
    # Create all tables
    create_tables(engine)
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Test document creation
    doc = Document(
        filename="test_document.pdf",
        original_filename="test_document.pdf",
        file_size=1024000,
        file_type=".pdf",
        mime_type="application/pdf",
        status="completed",
        title="Test Document",
        page_count=10,
        word_count=5000,
        character_count=25000,
        content_hash="abcd1234567890"
    )
    
    session.add(doc)
    session.commit()
    
    print(f"Created document: {doc}")
    print(f"Document dict: {doc.to_dict()}")
    
    # Test query
    docs = session.query(Document).all()
    print(f"Found {len(docs)} documents")
    
    session.close()