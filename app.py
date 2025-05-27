"""
Document Processor Agent - Main FastAPI Application
Advanced AI-powered document processing system with RAG and human-in-the-loop validation
"""

import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import tempfile
import shutil

# FastAPI and related imports
from fastapi import (
    FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks,
    WebSocket, WebSocketDisconnect, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Pydantic models
from pydantic import BaseModel, Field

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our agents and components
from agents.document_loader import DocumentLoader
from agents.parser import DocumentParser
from agents.embedder import EmbeddingAgent
from agents.metadata_writer import MetadataWriter
from agents.rag_enricher import RAGEnricher
from agents.qa_agent import QAAgent
from agents.validator import ValidatorAgent
from langgraph_flows.document_graph import DocumentProcessingGraph
from sqlstore.database import DatabaseManager
from vectorstore.chroma_config import ChromaConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_FILE = "config.yaml"
ENV_FILE = ".env"

# Load configuration
def load_config() -> Dict[str, Any]:
    """Load application configuration"""
    config = {
        'server': {
            'host': os.getenv('HOST', '0.0.0.0'),
            'port': int(os.getenv('PORT', '8000')),
            'reload': os.getenv('RELOAD', 'false').lower() == 'true',
            'workers': int(os.getenv('WORKERS', '1'))
        },
        'database': {
            'url': os.getenv('DATABASE_URL', 'sqlite:///./documents.db'),
            'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
            'echo': os.getenv('DB_ECHO', 'false').lower() == 'true'
        },
        'vector_store': {
            'provider': os.getenv('VECTOR_STORE_PROVIDER', 'chroma'),
            'persist_directory': os.getenv('CHROMA_PERSIST_DIR', './chroma_db'),
            'collection_name': os.getenv('CHROMA_COLLECTION', 'documents')
        },
        'llm': {
            'primary_provider': os.getenv('LLM_PROVIDER', 'openai'),
            'models': {
                'openai': {
                    'model': os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview'),
                    'api_key': os.getenv('OPENAI_API_KEY'),
                    'temperature': float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
                },
                'anthropic': {
                    'model': os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
                    'api_key': os.getenv('ANTHROPIC_API_KEY'),
                    'temperature': float(os.getenv('ANTHROPIC_TEMPERATURE', '0.1'))
                }
            }
        },
        'embeddings': {
            'embedding_provider': os.getenv('EMBEDDING_PROVIDER', 'openai'),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            'chunk_size': int(os.getenv('CHUNK_SIZE', '1000')),
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '200')),
            'embedding_dimensions': int(os.getenv('EMBEDDING_DIMENSIONS', '1536'))
        },
        'agents': {
            'document_loader': {
                'supported_formats': ['.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.pptx', '.xlsx'],
                'max_file_size_mb': int(os.getenv('MAX_FILE_SIZE_MB', '100')),
                'temp_dir': os.getenv('TEMP_DIR', './temp_uploads')
            },
            'validator': {
                'confidence_threshold': float(os.getenv('VALIDATION_THRESHOLD', '0.7')),
                'human_review_threshold': float(os.getenv('HUMAN_REVIEW_THRESHOLD', '0.5'))
            }
        },
        'web_search': {
            'provider': os.getenv('SEARCH_PROVIDER', 'tavily'),
            'api_key': os.getenv('SEARCH_API_KEY', '')
        }
    }
    
    return config

# Global variables
app_config = load_config()
db_manager: Optional[DatabaseManager] = None
vector_store: Optional[ChromaConfig] = None
processing_graph: Optional[DocumentProcessingGraph] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Document Processor Agent...")
    
    # Initialize global components
    global db_manager, vector_store, processing_graph
    
    try:
        # Initialize database
        db_manager = DatabaseManager(app_config['database'])
        await db_manager.initialize()
        
        # Initialize vector store
        vector_store = ChromaConfig(app_config['vector_store'])
        
        # Initialize processing graph
        processing_graph = DocumentProcessingGraph(app_config)
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('uploads', exist_ok=True)
        os.makedirs(app_config['agents']['document_loader']['temp_dir'], exist_ok=True)
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document Processor Agent...")
    
    try:
        if db_manager:
            await db_manager.close()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

# Create FastAPI application
app = FastAPI(
    title="Document Processor Agent",
    description="AI-powered document processing system with RAG and human-in-the-loop validation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user (implement authentication as needed)"""
    # For now, return a default user
    # In production, implement proper JWT validation
    return {"user_id": "default_user", "permissions": ["read", "write"]}

# Pydantic models
class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    file_size: int
    status: str
    message: str

class QueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    max_results: int = Field(default=5, ge=1, le=20)
    include_context: bool = True
    response_format: str = Field(default="detailed", pattern="^(brief|detailed|structured)$")

class QueryResponse(BaseModel):
    success: bool
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    processing_time: float
    context_used: int
    model_used: str

class ValidationRequest(BaseModel):
    query_id: str
    validation_type: str = Field(default="standard", pattern="^(quick|standard|comprehensive)$")
    human_review: bool = False

class SystemStats(BaseModel):
    total_documents: int
    processing_documents: int
    completed_documents: int
    failed_documents: int
    total_queries: int
    average_confidence: float
    system_status: str

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        if db_manager:
            stats = await db_manager.get_database_stats()
            db_healthy = True
        else:
            db_healthy = False
        
        # Check vector store
        vector_healthy = True
        try:
            if vector_store:
                collections = vector_store.list_collections()
        except:
            vector_healthy = False
        
        status = "healthy" if db_healthy and vector_healthy else "degraded"
        
        return {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "vector_store": "healthy" if vector_healthy else "unhealthy",
                "api": "healthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/api/stats", response_model=SystemStats)
async def get_system_stats(user=Depends(get_current_user)):
    """Get system statistics"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        stats = await db_manager.get_database_stats()
        
        return SystemStats(
            total_documents=stats.get('total_documents', 0),
            processing_documents=stats.get('processing_documents', 0),
            completed_documents=stats.get('completed_documents', 0),
            failed_documents=stats.get('failed_documents', 0),
            total_queries=stats.get('total_queries', 0),
            average_confidence=0.75,  # Calculate from actual data
            system_status="operational"
        )
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system statistics")

# Document management endpoints
@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    """Upload and process a document"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_extension = Path(file.filename).suffix.lower()
        supported_formats = app_config['agents']['document_loader']['supported_formats']
        
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: {', '.join(supported_formats)}"
            )
        
        # Check file size
        max_size = app_config['agents']['document_loader']['max_file_size_mb'] * 1024 * 1024
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {max_size // 1024 // 1024}MB"
            )
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Save file temporarily
        temp_dir = app_config['agents']['document_loader']['temp_dir']
        safe_filename = f"{document_id}_{file.filename}"
        file_path = Path(temp_dir) / safe_filename
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Create document record
        document_data = {
            'id': document_id,
            'filename': safe_filename,
            'original_filename': file.filename,
            'file_path': str(file_path),
            'file_size': file_size,
            'file_type': file_extension,
            'mime_type': file.content_type,
            'status': 'uploaded',
            'created_at': datetime.now(timezone.utc)
        }
        
        if db_manager:
            await db_manager.create_document(document_data)
        
        # Schedule background processing
        background_tasks.add_task(
            process_document_background,
            document_id,
            str(file_path),
            file.filename
        )
        
        logger.info(f"Document uploaded: {document_id} ({file.filename})")
        
        return DocumentUploadResponse(
            success=True,
            document_id=document_id,
            filename=file.filename,
            file_size=file_size,
            status="uploaded",
            message="Document uploaded successfully and queued for processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload document")

async def process_document_background(document_id: str, file_path: str, filename: str):
    """Background task to process uploaded document"""
    try:
        logger.info(f"Starting background processing for document: {document_id}")
        
        # Broadcast processing start
        await manager.broadcast(json.dumps({
            "type": "document_processing_started",
            "document_id": document_id,
            "status": "processing"
        }))
        
        if processing_graph:
            # Process document using LangGraph workflow
            result = await processing_graph.process_document({
                "document_id": document_id,
                "file_path": file_path,
                "filename": filename,
                "processing_config": app_config
            })
            
            # Broadcast completion
            await manager.broadcast(json.dumps({
                "type": "document_processing_completed",
                "document_id": document_id,
                "status": result["status"],
                "success": result["success"]
            }))
            
            logger.info(f"Document processing completed: {document_id} - {result['status']}")
        
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except:
            pass
            
    except Exception as e:
        logger.error(f"Background processing error for {document_id}: {str(e)}")
        
        # Update document status
        if db_manager:
            await db_manager.update_document(document_id, {
                'status': 'failed',
                'error_message': str(e)
            })
        
        # Broadcast error
        await manager.broadcast(json.dumps({
            "type": "document_processing_failed",
            "document_id": document_id,
            "error": str(e)
        }))

@app.get("/api/documents")
async def get_documents(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    user=Depends(get_current_user)
):
    """Get documents with pagination"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        offset = (page - 1) * page_size
        documents = await db_manager.get_documents(
            limit=page_size,
            offset=offset,
            status=status
        )
        
        total_count = await db_manager.get_document_count(status=status)
        
        return {
            "documents": [doc.to_dict() for doc in documents],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": (total_count + page_size - 1) // page_size
            }
        }
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get documents")

@app.get("/api/documents/{document_id}")
async def get_document(document_id: str, user=Depends(get_current_user)):
    """Get specific document details"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        document = await db_manager.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document chunks
        chunks = await db_manager.get_document_chunks(document_id)
        
        # Get processing results
        processing_results = await db_manager.get_processing_results(document_id)
        
        return {
            "document": document.to_dict(),
            "chunks": [chunk.to_dict() for chunk in chunks],
            "processing_results": [result.__dict__ for result in processing_results]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get document")

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, user=Depends(get_current_user)):
    """Delete a document"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        success = await db_manager.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # TODO: Also delete from vector store
        
        return {"success": True, "message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

# Query endpoints
@app.post("/api/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    user=Depends(get_current_user)
):
    """Query documents using natural language"""
    try:
        start_time = datetime.now()
        
        # Initialize QA agent
        qa_agent = QAAgent(app_config)
        
        # Get relevant context from vector store
        context_documents = []
        if vector_store:
            # Query vector store for relevant chunks
            results = vector_store.query_documents(
                query_texts=[request.question],
                n_results=request.max_results
            )
            
            # Format context documents
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    context_documents.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': results['distances'][0][i] if results['distances'] else 0.0
                    })
        
        # Process query
        result = await qa_agent.process({
            'query': request.question,
            'context_documents': context_documents,
            'response_format': request.response_format,
            'include_context': request.include_context
        })
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store query result
        if db_manager:
            query_data = {
                'query_text': request.question,
                'answer_text': result.data['answer'],
                'confidence': result.confidence,
                'response_time': processing_time,
                'context_documents_used': len(context_documents),
                'model_used': app_config['llm']['primary_provider'],
                'created_at': datetime.now(timezone.utc)
            }
            await db_manager.create_query_result(query_data)
        
        return QueryResponse(
            success=True,
            answer=result.data['answer'],
            confidence=result.confidence,
            sources=context_documents[:3],  # Return top 3 sources
            processing_time=processing_time,
            context_used=len(context_documents),
            model_used=app_config['llm']['primary_provider']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process query")

@app.get("/api/queries")
async def get_query_history(
    page: int = 1,
    page_size: int = 50,
    user=Depends(get_current_user)
):
    """Get query history"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        queries = await db_manager.get_query_results(limit=page_size)
        
        return {
            "queries": [query.to_dict() for query in queries],
            "pagination": {
                "page": page,
                "page_size": page_size
            }
        }
    except Exception as e:
        logger.error(f"Error getting query history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get query history")

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for now - could implement more sophisticated messaging
            await manager.send_personal_message(f"Message received: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Admin endpoints
@app.post("/api/admin/reindex")
async def reindex_documents(user=Depends(get_current_user)):
    """Reindex all documents in vector store"""
    try:
        # This would reprocess all documents and rebuild the vector index
        # Implementation depends on specific requirements
        return {"success": True, "message": "Reindexing started"}
    except Exception as e:
        logger.error(f"Reindexing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start reindexing")

@app.get("/api/admin/system-info")
async def get_system_info(user=Depends(get_current_user)):
    """Get detailed system information"""
    try:
        return {
            "version": "1.0.0",
            "python_version": sys.version,
            "config": {
                # Return non-sensitive config
                "llm_provider": app_config['llm']['primary_provider'],
                "vector_store": app_config['vector_store']['provider'],
                "embedding_provider": app_config['embeddings']['embedding_provider']
            },
            "stats": await db_manager.get_database_stats() if db_manager else {}
        }
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system information")

# Serve static files for the UI
if Path("web_ui/dist").exists():
    app.mount("/static", StaticFiles(directory="web_ui/dist"), name="static")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Document Processor Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "message": "The requested resource was not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

# CLI interface
def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Processor Agent")
    parser.add_argument("--host", default=app_config['server']['host'], help="Host to bind to")
    parser.add_argument("--port", type=int, default=app_config['server']['port'], help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=app_config['server']['workers'], help="Number of workers")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Run the server
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()

    # --- 
    """
Document Processor Agent - Main FastAPI Application
Advanced AI-powered document processing system with RAG and human-in-the-loop validation
"""

import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import tempfile
import shutil

# FastAPI and related imports
from fastapi import (
    FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks,
    WebSocket, WebSocketDisconnect, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Pydantic models
from pydantic import BaseModel, Field

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our agents and components
from agents.document_loader import DocumentLoader
from agents.parser import DocumentParser
from agents.embedder import EmbeddingAgent
from agents.metadata_writer import MetadataWriter
from agents.rag_enricher import RAGEnricher
from agents.qa_agent import QAAgent
from agents.validator import ValidatorAgent
from langgraph_flows.document_graph import DocumentProcessingGraph
from sqlstore.database import DatabaseManager
from vectorstore.chroma_config import ChromaConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_FILE = "config.yaml"
ENV_FILE = ".env"

# Load configuration
def load_config() -> Dict[str, Any]:
    """Load application configuration"""
    config = {
        'server': {
            'host': os.getenv('HOST', '0.0.0.0'),
            'port': int(os.getenv('PORT', '8000')),
            'reload': os.getenv('RELOAD', 'false').lower() == 'true',
            'workers': int(os.getenv('WORKERS', '1'))
        },
        'database': {
            'url': os.getenv('DATABASE_URL', 'sqlite:///./documents.db'),
            'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
            'echo': os.getenv('DB_ECHO', 'false').lower() == 'true'
        },
        'vector_store': {
            'provider': os.getenv('VECTOR_STORE_PROVIDER', 'chroma'),
            'persist_directory': os.getenv('CHROMA_PERSIST_DIR', './chroma_db'),
            'collection_name': os.getenv('CHROMA_COLLECTION', 'documents')
        },
        'llm': {
            'primary_provider': os.getenv('LLM_PROVIDER', 'openai'),
            'models': {
                'openai': {
                    'model': os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview'),
                    'api_key': os.getenv('OPENAI_API_KEY'),
                    'temperature': float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
                },
                'anthropic': {
                    'model': os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
                    'api_key': os.getenv('ANTHROPIC_API_KEY'),
                    'temperature': float(os.getenv('ANTHROPIC_TEMPERATURE', '0.1'))
                }
            }
        },
        'embeddings': {
            'embedding_provider': os.getenv('EMBEDDING_PROVIDER', 'openai'),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            'chunk_size': int(os.getenv('CHUNK_SIZE', '1000')),
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '200')),
            'embedding_dimensions': int(os.getenv('EMBEDDING_DIMENSIONS', '1536'))
        },
        'agents': {
            'document_loader': {
                'supported_formats': ['.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.pptx', '.xlsx'],
                'max_file_size_mb': int(os.getenv('MAX_FILE_SIZE_MB', '100')),
                'temp_dir': os.getenv('TEMP_DIR', './temp_uploads')
            },
            'validator': {
                'confidence_threshold': float(os.getenv('VALIDATION_THRESHOLD', '0.7')),
                'human_review_threshold': float(os.getenv('HUMAN_REVIEW_THRESHOLD', '0.5'))
            }
        },
        'web_search': {
            'provider': os.getenv('SEARCH_PROVIDER', 'tavily'),
            'api_key': os.getenv('SEARCH_API_KEY', '')
        }
    }
    
    return config

# Global variables
app_config = load_config()
db_manager: Optional[DatabaseManager] = None
vector_store: Optional[ChromaConfig] = None
processing_graph: Optional[DocumentProcessingGraph] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Document Processor Agent...")
    
    # Initialize global components
    global db_manager, vector_store, processing_graph
    
    try:
        # Initialize database
        db_manager = DatabaseManager(app_config['database'])
        await db_manager.initialize()
        
        # Initialize vector store
        vector_store = ChromaConfig(app_config['vector_store'])
        
        # Initialize processing graph
        processing_graph = DocumentProcessingGraph(app_config)
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('uploads', exist_ok=True)
        os.makedirs(app_config['agents']['document_loader']['temp_dir'], exist_ok=True)
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document Processor Agent...")
    
    try:
        if db_manager:
            await db_manager.close()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

# Create FastAPI application
app = FastAPI(
    title="Document Processor Agent",
    description="AI-powered document processing system with RAG and human-in-the-loop validation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user (implement authentication as needed)"""
    # For now, return a default user
    # In production, implement proper JWT validation
    return {"user_id": "default_user", "permissions": ["read", "write"]}

# Pydantic models
class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    file_size: int
    status: str
    message: str

class QueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    max_results: int = Field(default=5, ge=1, le=20)
    include_context: bool = True
    response_format: str = Field(default="detailed", pattern="^(brief|detailed|structured)$")

class QueryResponse(BaseModel):
    success: bool
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    processing_time: float
    context_used: int
    model_used: str

class ValidationRequest(BaseModel):
    query_id: str
    validation_type: str = Field(default="standard", pattern="^(quick|standard|comprehensive)$")
    human_review: bool = False

class SystemStats(BaseModel):
    total_documents: int
    processing_documents: int
    completed_documents: int
    failed_documents: int
    total_queries: int
    average_confidence: float
    system_status: str

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        if db_manager:
            stats = await db_manager.get_database_stats()
            db_healthy = True
        else:
            db_healthy = False
        
        # Check vector store
        vector_healthy = True
        try:
            if vector_store:
                collections = vector_store.list_collections()
        except:
            vector_healthy = False
        
        status = "healthy" if db_healthy and vector_healthy else "degraded"
        
        return {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "vector_store": "healthy" if vector_healthy else "unhealthy",
                "api": "healthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/api/stats", response_model=SystemStats)
async def get_system_stats(user=Depends(get_current_user)):
    """Get system statistics"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        stats = await db_manager.get_database_stats()
        
        return SystemStats(
            total_documents=stats.get('total_documents', 0),
            processing_documents=stats.get('processing_documents', 0),
            completed_documents=stats.get('completed_documents', 0),
            failed_documents=stats.get('failed_documents', 0),
            total_queries=stats.get('total_queries', 0),
            average_confidence=0.75,  # Calculate from actual data
            system_status="operational"
        )
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system statistics")

# Document management endpoints
@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    """Upload and process a document"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_extension = Path(file.filename).suffix.lower()
        supported_formats = app_config['agents']['document_loader']['supported_formats']
        
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: {', '.join(supported_formats)}"
            )
        
        # Check file size
        max_size = app_config['agents']['document_loader']['max_file_size_mb'] * 1024 * 1024
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {max_size // 1024 // 1024}MB"
            )
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Save file temporarily
        temp_dir = app_config['agents']['document_loader']['temp_dir']
        safe_filename = f"{document_id}_{file.filename}"
        file_path = Path(temp_dir) / safe_filename
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Create document record
        document_data = {
            'id': document_id,
            'filename': safe_filename,
            'original_filename': file.filename,
            'file_path': str(file_path),
            'file_size': file_size,
            'file_type': file_extension,
            'mime_type': file.content_type,
            'status': 'uploaded',
            'created_at': datetime.now(timezone.utc)
        }
        
        if db_manager:
            await db_manager.create_document(document_data)
        
        # Schedule background processing
        background_tasks.add_task(
            process_document_background,
            document_id,
            str(file_path),
            file.filename
        )
        
        logger.info(f"Document uploaded: {document_id} ({file.filename})")
        
        return DocumentUploadResponse(
            success=True,
            document_id=document_id,
            filename=file.filename,
            file_size=file_size,
            status="uploaded",
            message="Document uploaded successfully and queued for processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload document")

async def process_document_background(document_id: str, file_path: str, filename: str):
    """Background task to process uploaded document"""
    try:
        logger.info(f"Starting background processing for document: {document_id}")
        
        # Broadcast processing start
        await manager.broadcast(json.dumps({
            "type": "document_processing_started",
            "document_id": document_id,
            "status": "processing"
        }))
        
        if processing_graph:
            # Process document using LangGraph workflow
            result = await processing_graph.process_document({
                "document_id": document_id,
                "file_path": file_path,
                "filename": filename,
                "processing_config": app_config
            })
            
            # Broadcast completion
            await manager.broadcast(json.dumps({
                "type": "document_processing_completed",
                "document_id": document_id,
                "status": result["status"],
                "success": result["success"]
            }))
            
            logger.info(f"Document processing completed: {document_id} - {result['status']}")
        
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except:
            pass
            
    except Exception as e:
        logger.error(f"Background processing error for {document_id}: {str(e)}")
        
        # Update document status
        if db_manager:
            await db_manager.update_document(document_id, {
                'status': 'failed',
                'error_message': str(e)
            })
        
        # Broadcast error
        await manager.broadcast(json.dumps({
            "type": "document_processing_failed",
            "document_id": document_id,
            "error": str(e)
        }))

@app.get("/api/documents")
async def get_documents(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    user=Depends(get_current_user)
):
    """Get documents with pagination"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        offset = (page - 1) * page_size
        documents = await db_manager.get_documents(
            limit=page_size,
            offset=offset,
            status=status
        )
        
        total_count = await db_manager.get_document_count(status=status)
        
        return {
            "documents": [doc.to_dict() for doc in documents],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": (total_count + page_size - 1) // page_size
            }
        }
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get documents")

@app.get("/api/documents/{document_id}")
async def get_document(document_id: str, user=Depends(get_current_user)):
    """Get specific document details"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        document = await db_manager.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document chunks
        chunks = await db_manager.get_document_chunks(document_id)
        
        # Get processing results
        processing_results = await db_manager.get_processing_results(document_id)
        
        return {
            "document": document.to_dict(),
            "chunks": [chunk.to_dict() for chunk in chunks],
            "processing_results": [result.__dict__ for result in processing_results]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get document")

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, user=Depends(get_current_user)):
    """Delete a document"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        success = await db_manager.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # TODO: Also delete from vector store
        
        return {"success": True, "message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

# Query endpoints
@app.post("/api/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    user=Depends(get_current_user)
):
    """Query documents using natural language"""
    try:
        start_time = datetime.now()
        
        # Initialize QA agent
        qa_agent = QAAgent(app_config)
        
        # Get relevant context from vector store
        context_documents = []
        if vector_store:
            # Query vector store for relevant chunks
            results = vector_store.query_documents(
                query_texts=[request.question],
                n_results=request.max_results
            )
            
            # Format context documents
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    context_documents.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': results['distances'][0][i] if results['distances'] else 0.0
                    })
        
        # Process query
        result = await qa_agent.process({
            'query': request.question,
            'context_documents': context_documents,
            'response_format': request.response_format,
            'include_context': request.include_context
        })
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store query result
        if db_manager:
            query_data = {
                'query_text': request.question,
                'answer_text': result.data['answer'],
                'confidence': result.confidence,
                'response_time': processing_time,
                'context_documents_used': len(context_documents),
                'model_used': app_config['llm']['primary_provider'],
                'created_at': datetime.now(timezone.utc)
            }
            await db_manager.create_query_result(query_data)
        
        return QueryResponse(
            success=True,
            answer=result.data['answer'],
            confidence=result.confidence,
            sources=context_documents[:3],  # Return top 3 sources
            processing_time=processing_time,
            context_used=len(context_documents),
            model_used=app_config['llm']['primary_provider']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process query")

@app.get("/api/queries")
async def get_query_history(
    page: int = 1,
    page_size: int = 50,
    user=Depends(get_current_user)
):
    """Get query history"""
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        queries = await db_manager.get_query_results(limit=page_size)
        
        return {
            "queries": [query.to_dict() for query in queries],
            "pagination": {
                "page": page,
                "page_size": page_size
            }
        }
    except Exception as e:
        logger.error(f"Error getting query history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get query history")

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for now - could implement more sophisticated messaging
            await manager.send_personal_message(f"Message received: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Admin endpoints
@app.post("/api/admin/reindex")
async def reindex_documents(user=Depends(get_current_user)):
    """Reindex all documents in vector store"""
    try:
        # This would reprocess all documents and rebuild the vector index
        # Implementation depends on specific requirements
        return {"success": True, "message": "Reindexing started"}
    except Exception as e:
        logger.error(f"Reindexing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start reindexing")

@app.get("/api/admin/system-info")
async def get_system_info(user=Depends(get_current_user)):
    """Get detailed system information"""
    try:
        return {
            "version": "1.0.0",
            "python_version": sys.version,
            "config": {
                # Return non-sensitive config
                "llm_provider": app_config['llm']['primary_provider'],
                "vector_store": app_config['vector_store']['provider'],
                "embedding_provider": app_config['embeddings']['embedding_provider']
            },
            "stats": await db_manager.get_database_stats() if db_manager else {}
        }
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system information")

# Serve static files for the UI
if Path("web_ui/dist").exists():
    app.mount("/static", StaticFiles(directory="web_ui/dist"), name="static")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Document Processor Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "message": "The requested resource was not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

# CLI interface
def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Processor Agent")
    parser.add_argument("--host", default=app_config['server']['host'], help="Host to bind to")
    parser.add_argument("--port", type=int, default=app_config['server']['port'], help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=app_config['server']['workers'], help="Number of workers")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Run the server
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()
    