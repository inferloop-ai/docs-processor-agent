"""
Database Connection Manager
Handles database connections, sessions, and common operations
"""

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, List, Optional, Any, AsyncGenerator, Generator
import os
from datetime import datetime, timezone
import uuid

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from .models import (
    Base, Document, DocumentChunk, ProcessingResult, 
    QueryResult, ValidationResult, SystemMetrics, UserSession
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database connection and session manager
    
    Supports both synchronous and asynchronous operations
    Handles connection pooling, session management, and common database operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Database configuration
        self.database_url = self.config.get('url', os.getenv('DATABASE_URL', 'sqlite:///./documents.db'))
        self.pool_size = self.config.get('pool_size', 10)
        self.max_overflow = self.config.get('max_overflow', 20)
        self.pool_timeout = self.config.get('pool_timeout', 30)
        self.pool_recycle = self.config.get('pool_recycle', 3600)
        self.echo = self.config.get('echo', False)
        
        # Engine and session makers
        self.engine: Optional[Engine] = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        
        # Connection status
        self.is_initialized = False
        self.is_connected = False
        
        logger.info(f"DatabaseManager initialized with URL: {self._mask_database_url()}")
    
    def _mask_database_url(self) -> str:
        """Mask sensitive parts of database URL for logging"""
        if '://' not in self.database_url:
            return self.database_url
        
        protocol, rest = self.database_url.split('://', 1)
        if '@' in rest:
            credentials, host_db = rest.split('@', 1)
            return f"{protocol}://***:***@{host_db}"
        return self.database_url
    
    async def initialize(self):
        """Initialize database connections and create tables"""
        try:
            # Create synchronous engine
            if self.database_url.startswith('sqlite'):
                # SQLite-specific configuration
                self.engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    poolclass=StaticPool,
                    connect_args={"check_same_thread": False}
                )
            else:
                # PostgreSQL and other databases
                self.engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    pool_timeout=self.pool_timeout,
                    pool_recycle=self.pool_recycle
                )
            
            # Create async engine
            async_url = self._convert_to_async_url(self.database_url)
            if async_url:
                if self.database_url.startswith('sqlite'):
                    self.async_engine = create_async_engine(
                        async_url,
                        echo=self.echo,
                        poolclass=StaticPool,
                        connect_args={"check_same_thread": False}
                    )
                else:
                    self.async_engine = create_async_engine(
                        async_url,
                        echo=self.echo,
                        pool_size=self.pool_size,
                        max_overflow=self.max_overflow,
                        pool_timeout=self.pool_timeout,
                        pool_recycle=self.pool_recycle
                    )
            
            # Create session makers
            self.SessionLocal = sessionmaker(bind=self.engine)
            if self.async_engine:
                self.AsyncSessionLocal = async_sessionmaker(bind=self.async_engine)
            
            # Test connection
            await self.test_connection()
            
            # Create tables
            await self.create_tables()
            
            self.is_initialized = True
            self.is_connected = True
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def _convert_to_async_url(self, sync_url: str) -> Optional[str]:
        """Convert synchronous database URL to asynchronous version"""
        if sync_url.startswith('sqlite:'):
            return sync_url.replace('sqlite:', 'sqlite+aiosqlite:', 1)
        elif sync_url.startswith('postgresql:'):
            return sync_url.replace('postgresql:', 'postgresql+asyncpg:', 1)
        elif sync_url.startswith('mysql:'):
            return sync_url.replace('mysql:', 'mysql+aiomysql:', 1)
        else:
            logger.warning(f"Unknown database type for async conversion: {sync_url}")
            return None
    
    async def test_connection(self):
        """Test database connection"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            raise
    
    async def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise
    
    async def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop tables: {str(e)}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a synchronous database session"""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an asynchronous database session"""
        if not self.AsyncSessionLocal:
            raise RuntimeError("Async database not initialized. Call initialize() first.")
        
        session = self.AsyncSessionLocal()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    # Document operations
    async def create_document(self, document_data: Dict[str, Any]) -> Document:
        """Create a new document record"""
        try:
            with self.get_session() as session:
                # Check for duplicate content hash
                if document_data.get('content_hash'):
                    existing = session.query(Document).filter_by(
                        content_hash=document_data['content_hash']
                    ).first()
                    if existing:
                        logger.warning(f"Document with hash {document_data['content_hash']} already exists")
                        return existing
                
                document = Document(**document_data)
                session.add(document)
                session.flush()  # Get the ID
                
                logger.info(f"Created document: {document.id}")
                return document
                
        except IntegrityError as e:
            logger.error(f"Integrity error creating document: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error creating document: {str(e)}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID"""
        try:
            with self.get_session() as session:
                document = session.query(Document).filter_by(id=document_id).first()
                return document
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {str(e)}")
            return None
    
    async def get_documents(self, limit: int = 20, offset: int = 0, 
                          status: Optional[str] = None) -> List[Document]:
        """Get multiple documents with pagination"""
        try:
            with self.get_session() as session:
                query = session.query(Document)
                
                if status:
                    query = query.filter_by(status=status)
                
                documents = query.order_by(Document.created_at.desc())\
                              .offset(offset)\
                              .limit(limit)\
                              .all()
                
                return documents
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            return []
    
    async def get_document_count(self, status: Optional[str] = None) -> int:
        """Get total count of documents"""
        try:
            with self.get_session() as session:
                query = session.query(Document)
                
                if status:
                    query = query.filter_by(status=status)
                
                return query.count()
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0
    
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document record"""
        try:
            with self.get_session() as session:
                document = session.query(Document).filter_by(id=document_id).first()
                
                if not document:
                    return False
                
                for key, value in updates.items():
                    if hasattr(document, key):
                        setattr(document, key, value)
                
                document.updated_at = datetime.now(timezone.utc)
                
                logger.info(f"Updated document: {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {str(e)}")
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all related records"""
        try:
            with self.get_session() as session:
                document = session.query(Document).filter_by(id=document_id).first()
                
                if not document:
                    return False
                
                session.delete(document)
                
                logger.info(f"Deleted document: {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    # Document chunk operations
    async def create_document_chunks(self, chunks_data: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Create multiple document chunks"""
        try:
            with self.get_session() as session:
                chunks = []
                for chunk_data in chunks_data:
                    chunk = DocumentChunk(**chunk_data)
                    session.add(chunk)
                    chunks.append(chunk)
                
                session.flush()  # Get the IDs
                
                logger.info(f"Created {len(chunks)} document chunks")
                return chunks
                
        except Exception as e:
            logger.error(f"Error creating document chunks: {str(e)}")
            raise
    
    async def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document"""
        try:
            with self.get_session() as session:
                chunks = session.query(DocumentChunk)\
                               .filter_by(document_id=document_id)\
                               .order_by(DocumentChunk.chunk_index)\
                               .all()
                return chunks
        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {str(e)}")
            return []
    
    # Processing result operations
    async def create_processing_result(self, result_data: Dict[str, Any]) -> ProcessingResult:
        """Create a processing result record"""
        try:
            with self.get_session() as session:
                result = ProcessingResult(**result_data)
                session.add(result)
                session.flush()
                
                logger.info(f"Created processing result: {result.id}")
                return result
                
        except Exception as e:
            logger.error(f"Error creating processing result: {str(e)}")
            raise
    
    async def get_processing_results(self, document_id: str, 
                                   agent_name: Optional[str] = None) -> List[ProcessingResult]:
        """Get processing results for a document"""
        try:
            with self.get_session() as session:
                query = session.query(ProcessingResult).filter_by(document_id=document_id)
                
                if agent_name:
                    query = query.filter_by(agent_name=agent_name)
                
                results = query.order_by(ProcessingResult.created_at.desc()).all()
                return results
        except Exception as e:
            logger.error(f"Error getting processing results: {str(e)}")
            return []
    
    # Query result operations
    async def create_query_result(self, query_data: Dict[str, Any]) -> QueryResult:
        """Create a query result record"""
        try:
            with self.get_session() as session:
                query_result = QueryResult(**query_data)
                session.add(query_result)
                session.flush()
                
                logger.info(f"Created query result: {query_result.id}")
                return query_result
                
        except Exception as e:
            logger.error(f"Error creating query result: {str(e)}")
            raise
    
    async def get_query_results(self, session_id: Optional[str] = None,
                               user_id: Optional[str] = None,
                               limit: int = 50) -> List[QueryResult]:
        """Get query results with optional filtering"""
        try:
            with self.get_session() as session:
                query = session.query(QueryResult)
                
                if session_id:
                    query = query.filter_by(session_id=session_id)
                
                if user_id:
                    query = query.filter_by(user_id=user_id)
                
                results = query.order_by(QueryResult.created_at.desc())\
                              .limit(limit)\
                              .all()
                return results
        except Exception as e:
            logger.error(f"Error getting query results: {str(e)}")
            return []
    
    # System metrics operations
    async def record_metric(self, metric_name: str, value: float, 
                          category: str = "performance", 
                          unit: Optional[str] = None,
                          context_data: Optional[Dict] = None):
        """Record a system metric"""
        try:
            with self.get_session() as session:
                metric = SystemMetrics(
                    metric_name=metric_name,
                    metric_category=category,
                    metric_value=value,
                    metric_unit=unit,
                    context_data=context_data
                )
                session.add(metric)
                
                logger.debug(f"Recorded metric: {metric_name} = {value} {unit or ''}")
                
        except Exception as e:
            logger.error(f"Error recording metric: {str(e)}")
    
    async def get_metrics(self, metric_name: Optional[str] = None,
                         category: Optional[str] = None,
                         limit: int = 100) -> List[SystemMetrics]:
        """Get system metrics"""
        try:
            with self.get_session() as session:
                query = session.query(SystemMetrics)
                
                if metric_name:
                    query = query.filter_by(metric_name=metric_name)
                
                if category:
                    query = query.filter_by(metric_category=category)
                
                metrics = query.order_by(SystemMetrics.created_at.desc())\
                              .limit(limit)\
                              .all()
                return metrics
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return []
    
    # Session management
    async def create_user_session(self, session_id: str, user_id: Optional[str] = None,
                                 user_agent: Optional[str] = None,
                                 ip_address: Optional[str] = None) -> UserSession:
        """Create a user session record"""
        try:
            with self.get_session() as session:
                user_session = UserSession(
                    session_id=session_id,
                    user_id=user_id,
                    user_agent=user_agent,
                    ip_address=ip_address
                )
                session.add(user_session)
                session.flush()
                
                logger.info(f"Created user session: {session_id}")
                return user_session
                
        except Exception as e:
            logger.error(f"Error creating user session: {str(e)}")
            raise
    
    async def update_session_activity(self, session_id: str):
        """Update session last activity timestamp"""
        try:
            with self.get_session() as session:
                user_session = session.query(UserSession).filter_by(session_id=session_id).first()
                
                if user_session:
                    user_session.last_activity_at = datetime.now(timezone.utc)
                    
        except Exception as e:
            logger.error(f"Error updating session activity: {str(e)}")
    
    # Utility methods
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.get_session() as session:
                stats = {
                    'total_documents': session.query(Document).count(),
                    'processing_documents': session.query(Document).filter_by(status='processing').count(),
                    'completed_documents': session.query(Document).filter_by(status='completed').count(),
                    'failed_documents': session.query(Document).filter_by(status='failed').count(),
                    'total_chunks': session.query(DocumentChunk).count(),
                    'total_queries': session.query(QueryResult).count(),
                    'total_sessions': session.query(UserSession).count()
                }
                
                return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}
    
    async def cleanup_old_records(self, days: int = 30):
        """Clean up old records to maintain database size"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            with self.get_session() as session:
                # Clean up old metrics
                old_metrics = session.query(SystemMetrics)\
                                   .filter(SystemMetrics.created_at < cutoff_date)\
                                   .count()
                
                session.query(SystemMetrics)\
                       .filter(SystemMetrics.created_at < cutoff_date)\
                       .delete()
                
                # Clean up old sessions
                old_sessions = session.query(UserSession)\
                                    .filter(UserSession.ended_at < cutoff_date)\
                                    .count()
                
                session.query(UserSession)\
                       .filter(UserSession.ended_at < cutoff_date)\
                       .delete()
                
                logger.info(f"Cleaned up {old_metrics} old metrics and {old_sessions} old sessions")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    async def close(self):
        """Close database connections"""
        try:
            if self.engine:
                self.engine.dispose()
            
            if self.async_engine:
                await self.async_engine.dispose()
            
            self.is_connected = False
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")
    
    def __del__(self):
        """Cleanup on object destruction"""
        if self.is_connected:
            try:
                if self.engine:
                    self.engine.dispose()
            except:
                pass