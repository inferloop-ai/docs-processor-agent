"""
Document Processing Graph
LangGraph workflow for orchestrating document processing pipeline
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence
from datetime import datetime
import json

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain.schema import BaseMessage, HumanMessage, AIMessage

# Import our agents
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.document_loader import DocumentLoader
from agents.parser import DocumentParser
from agents.embedder import EmbeddingAgent
from agents.metadata_writer import MetadataWriter
from agents.rag_enricher import RAGEnricher
from agents.validator import ValidatorAgent
from sqlstore.database import DatabaseManager
from vectorstore.chroma_config import ChromaConfig

logger = logging.getLogger(__name__)


class DocumentProcessingState(TypedDict):
    """State for document processing workflow"""
    # Input
    document_id: str
    file_path: str
    filename: str
    processing_config: Dict[str, Any]
    
    # Processing stages
    status: str
    current_stage: str
    stage_results: Dict[str, Any]
    
    # Document data
    raw_documents: List[Any]
    parsed_documents: List[Any]
    chunked_documents: List[Any]
    embedded_documents: List[Any]
    
    # Metadata and results
    document_metadata: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    validation_results: Dict[str, Any]
    
    # Error handling
    errors: List[str]
    retry_count: int
    human_review_required: bool
    
    # Messages for LangGraph
    messages: Annotated[Sequence[BaseMessage], add_messages]


class DocumentProcessingGraph:
    """
    Document Processing Graph using LangGraph
    
    Orchestrates the complete document processing pipeline:
    1. Document Loading
    2. Parsing and Chunking
    3. Embedding Generation
    4. Metadata Extraction
    5. RAG Enhancement
    6. Validation
    7. Storage
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize agents
        self.document_loader = DocumentLoader(config)
        self.document_parser = DocumentParser(config)
        self.embedder = EmbeddingAgent(config)
        self.metadata_writer = MetadataWriter(config)
        self.rag_enricher = RAGEnricher(config)
        self.validator = ValidatorAgent(config)
        
        # Initialize storage components
        self.db_manager = DatabaseManager(config.get('database', {}))
        self.vector_store = ChromaConfig(config.get('vector_store', {}))
        
        # Workflow configuration
        self.workflow_config = config.get('workflows', {}).get('document_processing', {})
        self.max_iterations = self.workflow_config.get('max_iterations', 10)
        self.interrupts = self.workflow_config.get('interrupts', [])
        self.checkpointing = self.workflow_config.get('checkpointing', True)
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("DocumentProcessingGraph initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(DocumentProcessingState)
        
        # Add nodes for each processing stage
        workflow.add_node("load_document", self._load_document_node)
        workflow.add_node("parse_document", self._parse_document_node)
        workflow.add_node("generate_embeddings", self._generate_embeddings_node)
        workflow.add_node("extract_metadata", self._extract_metadata_node)
        workflow.add_node("enrich_with_rag", self._enrich_with_rag_node)
        workflow.add_node("validate_processing", self._validate_processing_node)
        workflow.add_node("store_results", self._store_results_node)
        workflow.add_node("handle_error", self._handle_error_node)
        workflow.add_node("human_review", self._human_review_node)
        
        # Define the workflow edges
        workflow.add_edge(START, "load_document")
        
        # From load_document
        workflow.add_conditional_edges(
            "load_document",
            self._should_continue_after_load,
            {
                "continue": "parse_document",
                "error": "handle_error",
                "human_review": "human_review"
            }
        )
        
        # From parse_document
        workflow.add_conditional_edges(
            "parse_document",
            self._should_continue_after_parse,
            {
                "continue": "generate_embeddings",
                "error": "handle_error",
                "human_review": "human_review"
            }
        )
        
        # From generate_embeddings
        workflow.add_conditional_edges(
            "generate_embeddings",
            self._should_continue_after_embeddings,
            {
                "continue": "extract_metadata",
                "error": "handle_error"
            }
        )
        
        # From extract_metadata
        workflow.add_conditional_edges(
            "extract_metadata",
            self._should_continue_after_metadata,
            {
                "continue": "enrich_with_rag",
                "skip_rag": "validate_processing",
                "error": "handle_error"
            }
        )
        
        # From enrich_with_rag
        workflow.add_conditional_edges(
            "enrich_with_rag",
            self._should_continue_after_rag,
            {
                "continue": "validate_processing",
                "error": "handle_error"
            }
        )
        
        # From validate_processing
        workflow.add_conditional_edges(
            "validate_processing",
            self._should_continue_after_validation,
            {
                "continue": "store_results",
                "retry": "load_document",
                "human_review": "human_review",
                "error": "handle_error"
            }
        )
        
        # From store_results
        workflow.add_edge("store_results", END)
        
        # From error handling
        workflow.add_conditional_edges(
            "handle_error",
            self._should_retry_after_error,
            {
                "retry": "load_document",
                "human_review": "human_review",
                "end": END
            }
        )
        
        # From human review
        workflow.add_conditional_edges(
            "human_review",
            self._continue_after_human_review,
            {
                "continue": "load_document",
                "end": END
            }
        )
        
        # Compile the graph with checkpointing
        if self.checkpointing:
            memory = MemorySaver()
            return workflow.compile(checkpointer=memory, interrupt_before=self.interrupts)
        else:
            return workflow.compile()
    
    # Node implementations
    async def _load_document_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Load and initial processing of document"""
        logger.info(f"Loading document: {state['filename']}")
        
        try:
            state["current_stage"] = "loading"
            state["messages"].append(HumanMessage(content=f"Loading document: {state['filename']}"))
            
            # Load document using DocumentLoader
            result = await self.document_loader.execute(state["file_path"])
            
            if result.success:
                state["raw_documents"] = result.data["documents"]
                state["stage_results"]["load"] = result.data
                state["document_metadata"].update(result.data.get("metadata", {}))
                
                # Update status in database
                await self.db_manager.update_document(state["document_id"], {
                    "status": "processing",
                    "processing_started_at": datetime.now()
                })
                
                logger.info(f"Successfully loaded document with {len(state['raw_documents'])} pages")
                
            else:
                state["errors"].append(f"Document loading failed: {result.error}")
                logger.error(f"Document loading failed: {result.error}")
            
        except Exception as e:
            error_msg = f"Error in load_document_node: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    async def _parse_document_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Parse document content and create chunks"""
        logger.info("Parsing document content")
        
        try:
            state["current_stage"] = "parsing"
            state["messages"].append(AIMessage(content="Parsing document content and creating chunks..."))
            
            # Parse documents using DocumentParser
            result = await self.document_parser.execute(state["raw_documents"])
            
            if result.success:
                state["parsed_documents"] = result.data["original_documents"]
                state["chunked_documents"] = result.data["chunked_documents"]
                state["stage_results"]["parse"] = result.data
                state["document_metadata"].update(result.data.get("metadata", {}))
                
                logger.info(f"Successfully parsed document into {len(state['chunked_documents'])} chunks")
                
            else:
                state["errors"].append(f"Document parsing failed: {result.error}")
                logger.error(f"Document parsing failed: {result.error}")
            
        except Exception as e:
            error_msg = f"Error in parse_document_node: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    async def _generate_embeddings_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Generate embeddings for document chunks"""
        logger.info("Generating embeddings for document chunks")
        
        try:
            state["current_stage"] = "embedding"
            state["messages"].append(AIMessage(content="Generating vector embeddings..."))
            
            # Generate embeddings using EmbeddingAgent
            result = await self.embedder.execute({
                "documents": state["chunked_documents"],
                "document_id": state["document_id"]
            })
            
            if result.success:
                state["embedded_documents"] = result.data["embedded_documents"]
                state["stage_results"]["embed"] = result.data
                
                # Store embeddings in vector database
                await self._store_embeddings_in_vector_db(state)
                
                logger.info(f"Successfully generated embeddings for {len(state['embedded_documents'])} chunks")
                
            else:
                state["errors"].append(f"Embedding generation failed: {result.error}")
                logger.error(f"Embedding generation failed: {result.error}")
            
        except Exception as e:
            error_msg = f"Error in generate_embeddings_node: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    async def _extract_metadata_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Extract and store document metadata"""
        logger.info("Extracting document metadata")
        
        try:
            state["current_stage"] = "metadata"
            state["messages"].append(AIMessage(content="Extracting document metadata..."))
            
            # Extract metadata using MetadataWriter
            metadata_input = {
                "document_id": state["document_id"],
                "documents": state["chunked_documents"],
                "existing_metadata": state["document_metadata"]
            }
            
            result = await self.metadata_writer.execute(metadata_input)
            
            if result.success:
                state["stage_results"]["metadata"] = result.data
                state["document_metadata"].update(result.data.get("enhanced_metadata", {}))
                
                logger.info("Successfully extracted document metadata")
                
            else:
                # Metadata extraction is non-critical, log warning but continue
                logger.warning(f"Metadata extraction failed: {result.error}")
            
        except Exception as e:
            logger.warning(f"Error in extract_metadata_node: {str(e)}")
        
        return state
    
    async def _enrich_with_rag_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Enrich document with external knowledge using RAG"""
        logger.info("Enriching document with RAG")
        
        try:
            state["current_stage"] = "rag_enrichment"
            state["messages"].append(AIMessage(content="Enriching with external knowledge..."))
            
            # Check if RAG enrichment is enabled
            if not self.config.get('agents', {}).get('rag_enricher', {}).get('web_search_enabled', False):
                logger.info("RAG enrichment disabled, skipping")
                return state
            
            # Enrich using RAGEnricher
            enrichment_input = {
                "documents": state["chunked_documents"],
                "metadata": state["document_metadata"]
            }
            
            result = await self.rag_enricher.execute(enrichment_input)
            
            if result.success:
                state["stage_results"]["rag"] = result.data
                
                # Update chunks with enriched information
                if result.data.get("enriched_documents"):
                    state["chunked_documents"] = result.data["enriched_documents"]
                
                logger.info("Successfully enriched document with external knowledge")
                
            else:
                # RAG enrichment is optional, log warning but continue
                logger.warning(f"RAG enrichment failed: {result.error}")
            
        except Exception as e:
            logger.warning(f"Error in enrich_with_rag_node: {str(e)}")
        
        return state
    
    async def _validate_processing_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Validate the processing results"""
        logger.info("Validating processing results")
        
        try:
            state["current_stage"] = "validation"
            state["messages"].append(AIMessage(content="Validating processing results..."))
            
            # Prepare validation input
            validation_input = {
                "query": f"Document processing validation for {state['filename']}",
                "answer": f"Processed document with {len(state['chunked_documents'])} chunks",
                "context_documents": [
                    {
                        "content": doc.page_content[:500],
                        "metadata": doc.metadata
                    } for doc in state["chunked_documents"][:5]  # Sample chunks
                ],
                "validation_type": "comprehensive"
            }
            
            result = await self.validator.execute(validation_input)
            
            if result.success:
                state["validation_results"] = result.data
                
                # Check if human review is required
                if result.data.get("human_review_required", False):
                    state["human_review_required"] = True
                
                logger.info(f"Validation completed with status: {result.data.get('overall_status')}")
                
            else:
                state["errors"].append(f"Validation failed: {result.error}")
                logger.error(f"Validation failed: {result.error}")
            
        except Exception as e:
            error_msg = f"Error in validate_processing_node: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    async def _store_results_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Store final processing results"""
        logger.info("Storing processing results")
        
        try:
            state["current_stage"] = "storing"
            state["messages"].append(AIMessage(content="Storing processing results..."))
            
            # Store document chunks in database
            await self._store_chunks_in_database(state)
            
            # Store processing results
            await self._store_processing_results(state)
            
            # Update document status
            await self.db_manager.update_document(state["document_id"], {
                "status": "completed",
                "processing_completed_at": datetime.now(),
                "title": state["document_metadata"].get("title"),
                "author": state["document_metadata"].get("author"),
                "language": state["document_metadata"].get("language"),
                "page_count": state["document_metadata"].get("page_count"),
                "word_count": state["document_metadata"].get("word_count"),
                "character_count": state["document_metadata"].get("character_count")
            })
            
            state["status"] = "completed"
            logger.info(f"Successfully completed processing for document: {state['document_id']}")
            
        except Exception as e:
            error_msg = f"Error in store_results_node: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    async def _handle_error_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Handle processing errors"""
        logger.error(f"Handling errors for document: {state['document_id']}")
        
        try:
            state["current_stage"] = "error_handling"
            
            # Log all errors
            error_summary = "; ".join(state["errors"])
            logger.error(f"Processing errors: {error_summary}")
            
            # Update document with error status
            await self.db_manager.update_document(state["document_id"], {
                "status": "failed",
                "error_message": error_summary,
                "error_details": {
                    "errors": state["errors"],
                    "last_stage": state["current_stage"],
                    "retry_count": state["retry_count"]
                }
            })
            
            state["status"] = "failed"
            
        except Exception as e:
            logger.error(f"Error in handle_error_node: {str(e)}")
        
        return state
    
    async def _human_review_node(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Handle human review requirements"""
        logger.info(f"Human review required for document: {state['document_id']}")
        
        try:
            state["current_stage"] = "human_review"
            
            # Update document status
            await self.db_manager.update_document(state["document_id"], {
                "status": "needs_review",
                "error_message": "Human review required"
            })
            
            # Store human review request
            review_data = {
                "document_id": state["document_id"],
                "review_type": "processing_validation",
                "review_data": {
                    "validation_results": state.get("validation_results", {}),
                    "processing_stages": list(state["stage_results"].keys()),
                    "errors": state["errors"]
                }
            }
            
            logger.info(f"Human review request created for document: {state['document_id']}")
            
        except Exception as e:
            logger.error(f"Error in human_review_node: {str(e)}")
        
        return state
    
    # Conditional edge functions
    def _should_continue_after_load(self, state: DocumentProcessingState) -> str:
        """Determine next step after document loading"""
        if state["errors"]:
            return "error"
        if not state.get("raw_documents"):
            return "error"
        return "continue"
    
    def _should_continue_after_parse(self, state: DocumentProcessingState) -> str:
        """Determine next step after document parsing"""
        if state["errors"]:
            return "error"
        if not state.get("chunked_documents"):
            return "error"
        return "continue"
    
    def _should_continue_after_embeddings(self, state: DocumentProcessingState) -> str:
        """Determine next step after embedding generation"""
        if state["errors"]:
            return "error"
        return "continue"
    
    def _should_continue_after_metadata(self, state: DocumentProcessingState) -> str:
        """Determine next step after metadata extraction"""
        if state["errors"]:
            return "error"
        
        # Check if RAG enrichment should be skipped
        if not self.config.get('agents', {}).get('rag_enricher', {}).get('web_search_enabled', False):
            return "skip_rag"
        
        return "continue"
    
    def _should_continue_after_rag(self, state: DocumentProcessingState) -> str:
        """Determine next step after RAG enrichment"""
        if state["errors"]:
            return "error"
        return "continue"
    
    def _should_continue_after_validation(self, state: DocumentProcessingState) -> str:
        """Determine next step after validation"""
        if state["errors"]:
            return "error"
        
        if state.get("human_review_required"):
            return "human_review"
        
        validation_status = state.get("validation_results", {}).get("overall_status")
        if validation_status == "FAIL" and state["retry_count"] < 2:
            state["retry_count"] += 1
            return "retry"
        
        return "continue"
    
    def _should_retry_after_error(self, state: DocumentProcessingState) -> str:
        """Determine whether to retry after error"""
        if state["retry_count"] >= 3:
            return "end"
        
        # Check if errors are retryable
        retryable_errors = ["timeout", "network", "temporary"]
        has_retryable_error = any(
            any(keyword in error.lower() for keyword in retryable_errors)
            for error in state["errors"]
        )
        
        if has_retryable_error:
            state["retry_count"] += 1
            state["errors"] = []  # Clear errors for retry
            return "retry"
        
        return "end"
    
    def _continue_after_human_review(self, state: DocumentProcessingState) -> str:
        """Determine next step after human review"""
        # This would be implemented based on human review results
        # For now, assume continue
        return "continue"
    
    # Helper methods
    async def _store_embeddings_in_vector_db(self, state: DocumentProcessingState):
        """Store embeddings in vector database"""
        try:
            documents_text = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(state["embedded_documents"]):
                documents_text.append(doc.page_content)
                
                metadata = doc.metadata.copy()
                metadata["document_id"] = state["document_id"]
                metadatas.append(metadata)
                
                ids.append(f"{state['document_id']}_chunk_{i}")
            
            self.vector_store.add_documents(
                documents=documents_text,
                metadatas=metadatas,
                ids=ids
            )
            
        except Exception as e:
            logger.error(f"Error storing embeddings in vector DB: {str(e)}")
            raise
    
    async def _store_chunks_in_database(self, state: DocumentProcessingState):
        """Store document chunks in SQL database"""
        try:
            chunks_data = []
            
            for i, doc in enumerate(state["chunked_documents"]):
                chunk_data = {
                    "document_id": state["document_id"],
                    "chunk_index": i,
                    "chunk_text": doc.page_content,
                    "chunk_length": len(doc.page_content),
                    "start_char": doc.metadata.get("start_char"),
                    "end_char": doc.metadata.get("end_char"),
                    "page_number": doc.metadata.get("page"),
                    "section": doc.metadata.get("section"),
                    "chunk_metadata": doc.metadata,
                    "vector_id": f"{state['document_id']}_chunk_{i}"
                }
                chunks_data.append(chunk_data)
            
            await self.db_manager.create_document_chunks(chunks_data)
            
        except Exception as e:
            logger.error(f"Error storing chunks in database: {str(e)}")
            raise
    
    async def _store_processing_results(self, state: DocumentProcessingState):
        """Store processing results for each stage"""
        try:
            for stage_name, stage_result in state["stage_results"].items():
                result_data = {
                    "document_id": state["document_id"],
                    "agent_name": f"{stage_name}_agent",
                    "processing_stage": stage_name,
                    "success": True,
                    "result_data": stage_result,
                    "processing_metadata": {
                        "workflow_id": state.get("workflow_id"),
                        "processing_config": state["processing_config"]
                    }
                }
                
                await self.db_manager.create_processing_result(result_data)
            
        except Exception as e:
            logger.error(f"Error storing processing results: {str(e)}")
            raise
    
    # Public interface
    async def process_document(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document through the complete pipeline"""
        
        # Prepare initial state
        state = DocumentProcessingState(
            document_id=initial_state["document_id"],
            file_path=initial_state["file_path"],
            filename=initial_state["filename"],
            processing_config=initial_state.get("processing_config", {}),
            status="processing",
            current_stage="initialized",
            stage_results={},
            raw_documents=[],
            parsed_documents=[],
            chunked_documents=[],
            embedded_documents=[],
            document_metadata={},
            processing_metadata={},
            validation_results={},
            errors=[],
            retry_count=0,
            human_review_required=False,
            messages=[]
        )
        
        try:
            # Execute the workflow
            config = {"configurable": {"thread_id": state["document_id"]}}
            
            final_state = await self.graph.ainvoke(state, config)
            
            logger.info(f"Document processing completed with status: {final_state['status']}")
            
            return {
                "success": final_state["status"] == "completed",
                "document_id": final_state["document_id"],
                "status": final_state["status"],
                "errors": final_state["errors"],
                "processing_metadata": final_state["processing_metadata"],
                "human_review_required": final_state["human_review_required"]
            }
            
        except Exception as e:
            logger.error(f"Error in document processing workflow: {str(e)}")
            
            # Update document with error status
            await self.db_manager.update_document(initial_state["document_id"], {
                "status": "failed",
                "error_message": str(e)
            })
            
            return {
                "success": False,
                "document_id": initial_state["document_id"],
                "status": "failed",
                "errors": [str(e)],
                "processing_metadata": {},
                "human_review_required": False
            }