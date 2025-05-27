"""
ChromaDB Configuration and Setup
Handles ChromaDB vector store initialization and management
"""

import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ChromaConfig:
    """ChromaDB configuration and client management"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.client = None
        self.collection = None
        
    def get_client(self) -> chromadb.Client:
        """Get or create ChromaDB client"""
        if self.client is None:
            self.client = self._create_client()
        return self.client
    
    def _create_client(self) -> chromadb.Client:
        """Create ChromaDB client with appropriate settings"""
        
        # Determine if using persistent or in-memory storage
        persist_directory = self.config.get('persist_directory', './chroma_db')
        
        if persist_directory:
            # Persistent client
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory,
                anonymized_telemetry=False
            )
            
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            
            client = chromadb.Client(settings)
            logger.info(f"Created persistent ChromaDB client at {persist_directory}")
            
        else:
            # In-memory client
            client = chromadb.Client()
            logger.info("Created in-memory ChromaDB client")
        
        return client
    
    def get_collection(self, collection_name: str = None, embedding_function=None):
        """Get or create a collection"""
        if collection_name is None:
            collection_name = self.config.get('collection_name', 'documents')
        
        client = self.get_client()
        
        # Setup embedding function
        if embedding_function is None:
            embedding_function = self._get_embedding_function()
        
        try:
            # Try to get existing collection
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"Retrieved existing collection: {collection_name}")
            
        except ValueError:
            # Collection doesn't exist, create it
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"description": f"Document collection: {collection_name}"}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        return collection
    
    def _get_embedding_function(self):
        """Get the appropriate embedding function"""
        
        # Get embedding provider from config
        provider = self.config.get('embedding_provider', 'sentence-transformers')
        model = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        
        if provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found, falling back to sentence-transformers")
                return embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=model
            )
            
        elif provider == 'sentence-transformers':
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model
            )
            
        elif provider == 'huggingface':
            return embedding_functions.HuggingFaceEmbeddingFunction(
                api_key=os.getenv('HUGGINGFACE_API_KEY'),
                model_name=model
            )
            
        else:
            logger.warning(f"Unknown embedding provider: {provider}, using sentence-transformers")
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
    
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: List[Dict] = None, 
                     ids: List[str] = None,
                     collection_name: str = None):
        """Add documents to the collection"""
        
        collection = self.get_collection(collection_name)
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to collection")
    
    def query_documents(self,
                       query_texts: List[str],
                       n_results: int = 10,
                       where: Dict = None,
                       collection_name: str = None):
        """Query documents from the collection"""
        
        collection = self.get_collection(collection_name)
        
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where
        )
        
        return results
    
    def update_documents(self,
                        ids: List[str],
                        documents: List[str] = None,
                        metadatas: List[Dict] = None,
                        collection_name: str = None):
        """Update existing documents"""
        
        collection = self.get_collection(collection_name)
        
        collection.update(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Updated {len(ids)} documents")
    
    def delete_documents(self, ids: List[str], collection_name: str = None):
        """Delete documents from collection"""
        
        collection = self.get_collection(collection_name)
        
        collection.delete(ids=ids)
        
        logger.info(f"Deleted {len(ids)} documents")
    
    def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Get information about the collection"""
        
        collection = self.get_collection(collection_name)
        
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata
        }
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        client = self.get_client()
        collections = client.list_collections()
        return [col.name for col in collections]
    
    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        client = self.get_client()"""
Reference Linker Agent
Handles citation linking, reference management, and cross-document relationships
"""

import asyncio
import re
import json
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse
import hashlib

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentResult


@dataclass
class Reference:
    """Structured reference information"""
    id: str
    type: str  # 'citation', 'url', 'internal', 'bibliography'
    text: str
    target: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.8
    source_document: Optional[str] = None
    position: Optional[int] = None


@dataclass
class CrossReference:
    """Cross-reference between documents"""
    source_doc_id: str
    target_doc_id: str
    reference_type: str
    strength: float
    context: str
    bidirectional: bool = False


class ReferencePattern(BaseModel):
    """Reference pattern for detection"""
    name: str
    pattern: str
    type: str
    confidence: float = Field(ge=0.0, le=1.0)


class ReferenceLinker(BaseAgent):
    """
    Reference Linker Agent
    
    Capabilities:
    - Citation detection and parsing
    - URL and link extraction
    - Cross-document reference mapping
    - Bibliography management
    - Reference validation and verification
    - Academic citation formatting
    - Internal document linking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Load reference linker configuration
        self.linker_config = self.config.get('agents', {}).get('reference_linker', {})
        
        # Reference detection settings
        self.detect_citations = self.linker_config.get('detect_citations', True)
        self.detect_urls = self.linker_config.get('detect_urls', True)
        self.detect_internal_refs = self.linker_config.get('detect_internal_refs', True)
        self.validate_urls = self.linker_config.get('validate_urls', False)
        
        # Processing limits
        self.max_references_per_doc = self.linker_config.get('max_references_per_doc', 100)
        self.reference_context_window = self.linker_config.get('context_window', 200)
        
        # Initialize reference patterns
        self._initialize_reference_patterns()
        
        # Setup reference processing prompts
        self._setup_reference_prompts()
        
        # Reference cache
        self.reference_cache = {}
        
        self.log_info("ReferenceLinker initialized")
    
    def _initialize_reference_patterns(self):
        """Initialize patterns for different reference types"""
        
        self.reference_patterns = [
            # Academic citations
            ReferencePattern(
                name="apa_citation",
                pattern=r'\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+\d{4})\)',
                type="citation",
                confidence=0.9
            ),
            ReferencePattern(
                name="numbered_citation", 
                pattern=r'\[(\d+)\]',
                type="citation",
                confidence=0.8
            ),
            ReferencePattern(
                name="author_year",
                pattern=r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\(\d{4}\))',
                type="citation",
                confidence=0.85
            ),
            
            # URLs and links
            ReferencePattern(
                name="http_url",
                pattern=r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                type="url",
                confidence=0.95
            ),
            ReferencePattern(
                name="doi",
                pattern=r'(?:doi:|DOI:)?\s*(10\.\d+/[^\s]+)',
                type="doi",
                confidence=0.9
            ),
            
            # Internal references
            ReferencePattern(
                name="section_ref",
                pattern=r'(?:Section|section|Chapter|chapter|Part|part)\s+(\d+(?:\.\d+)*)',
                type="internal",
                confidence=0.7
            ),
            ReferencePattern(
                name="figure_ref",
                pattern=r'(?:Figure|figure|Fig\.|fig\.)\s+(\d+(?:\.\d+)*)',
                type="figure",
                confidence=0.8
            ),
            ReferencePattern(
                name="table_ref",
                pattern=r'(?:Table|table|Tab\.|tab\.)\s+(\d+(?:\.\d+)*)',
                type="table",
                confidence=0.8
            ),
            
            # Bibliography entries
            ReferencePattern(
                name="bibliography_entry",
                pattern=r'^([A-Z][a-z]+(?:,\s+[A-Z]\.)*)\s+\((\d{4})\)\.\s+(.+)\.\s*$',
                type="bibliography",
                confidence=0.85
            )
        ]
    
    def _setup_reference_prompts(self):
        """Setup prompts for AI-powered reference processing"""
        
        self.citation_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting and analyzing citations from academic and technical documents.

Your task is to:
1. Identify all citations in the text
2. Classify the citation style (APA, MLA, Chicago, IEEE, etc.)
3. Extract key information (author, year, title, publication)
4. Assess the quality and completeness of citations

Be thorough and accurate in your analysis."""),
            
            ("human", """Text: {text}

Please extract all citations from this text and provide:
1. Complete list of citations found
2. Citation style used
3. Any incomplete or malformed citations
4. Cross-references between citations

Format as JSON:
{{
    "citations": [
        {{
            "text": "citation text",
            "type": "style type",
            "author": "author if identifiable",
            "year": "year if present",
            "complete": true/false
        }}
    ],
    "style": "predominant citation style",
    "cross_references": ["list of internal references"]
}}""")
        ])
        
        self.reference_validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at validating academic references and citations.

Evaluate references for:
1. Completeness of information
2. Proper formatting
3. Consistency with citation style
4. Potential errors or missing elements

Provide specific feedback for improvements."""),
            
            ("human", """References to validate:
{references}

Citation style: {style}

Please validate these references and provide:
1. Overall quality assessment
2. Specific issues found
3. Suggestions for improvement
4. Missing information

Format as JSON.""")
        ])
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Process reference linking for documents
        
        Args:
            input_data: Dict containing documents and processing options
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with reference linking results
        """
        try:
            # Parse input
            documents = input_data.get('documents', [])
            if not documents:
                return AgentResult(
                    success=False,
                    error="No documents provided for reference linking"
                )
            
            document_id = input_data.get('document_id', 'unknown')
            
            # Process references for each document
            all_references = []
            cross_references = []
            document_references = {}
            
            for i, doc in enumerate(documents):
                doc_id = f"{document_id}_doc_{i}"
                
                # Extract references from document
                doc_references = await self._extract_references_from_document(doc, doc_id)
                all_references.extend(doc_references)
                document_references[doc_id] = doc_references
            
            # Find cross-references between documents
            if len(documents) > 1:
                cross_references = await self._find_cross_references(
                    documents, document_references
                )
            
            # Validate and enrich references
            validated_references = await self._validate_references(all_references)
            
            # Create reference map
            reference_map = await self._create_reference_map(
                validated_references, cross_references
            )
            
            # Generate reference statistics
            stats = self._calculate_reference_statistics(
                validated_references, cross_references
            )
            
            result_data = {
                "references": [ref.__dict__ for ref in validated_references],
                "cross_references": [cref.__dict__ for cref in cross_references],
                "reference_map": reference_map,
                "statistics": stats,
                "document_references": {
                    doc_id: [ref.__dict__ for ref in refs] 
                    for doc_id, refs in document_references.items()
                },
                "processing_metadata": {
                    "total_documents": len(documents),
                    "total_references": len(validated_references),
                    "total_cross_references": len(cross_references),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                confidence=0.85,
                metadata={
                    "document_id": document_id,
                    "reference_types_found": list(set(ref.type for ref in validated_references))
                }
            )
            
        except Exception as e:
            self.log_error(f"Error in reference linking: {str(e)}")
            return AgentResult(
                success=False,
                error=f"Reference linking failed: {str(e)}"
            )
    
    async def _extract_references_from_document(self, document: Document, 
                                              doc_id: str) -> List[Reference]:
        """Extract references from a single document"""
        
        content = document.page_content
        references = []
        reference_id_counter = 0
        
        # Apply each reference pattern
        for pattern in self.reference_patterns:
            matches = re.finditer(pattern.pattern, content, re.MULTILINE)
            
            for match in matches:
                reference_id_counter += 1
                ref_id = f"{doc_id}_ref_{reference_id_counter}"
                
                # Extract context around the reference
                start_pos = max(0, match.start() - self.reference_context_window)
                end_pos = min(len(content), match.end() + self.reference_context_window)
                context = content[start_pos:end_pos]
                
                reference = Reference(
                    id=ref_id,
                    type=pattern.type,
                    text=match.group(0),
                    target=match.group(1) if match.groups() else None,
                    metadata={
                        "pattern_name": pattern.name,
                        "context": context,
                        "position": match.start(),
                        "document_source": document.metadata.get('source', 'unknown')
                    },
                    confidence=pattern.confidence,
                    source_document=doc_id,
                    position=match.start()
                )
                
                references.append(reference)
        
        # Use AI for additional reference extraction if enabled
        if len(references) < 10 and len(content) > 500:  # Only for substantial content
            ai_references = await self._extract_references_with_ai(content, doc_id)
            references.extend(ai_references)
        
        # Remove duplicates
        references = self._deduplicate_references(references)
        
        # Limit number of references per document
        if len(references) > self.max_references_per_doc:
            references = sorted(references, key=lambda x: x.confidence, reverse=True)
            references = references[:self.max_references_per_doc]
        
        return references
    
    async def _extract_references_with_ai(self, content: str, doc_id: str) -> List[Reference]:
        """Use AI to extract additional references"""
        
        try:
            llm = self.get_llm_client()
            
            # Limit content size for AI processing
            if len(content) > 4000:
                content = content[:2000] + "\n...\n" + content[-2000:]
            
            prompt = self.citation_extraction_prompt.format_prompt(text=content)
            response = await llm.ainvoke(prompt.to_messages())
            
            # Parse AI response
            try:
                result_data = json.loads(response.content)
                ai_references = []
                
                for i, citation_data in enumerate(result_data.get('citations', [])):
                    ref_id = f"{doc_id}_ai_ref_{i}"
                    
                    reference = Reference(
                        id=ref_id,
                        type="citation",
                        text=citation_data.get('text', ''),
                        target=citation_data.get('author', ''),
                        metadata={
                            "ai_extracted": True,
                            "citation_style": result_data.get('style', 'unknown'),
                            "complete": citation_data.get('complete', False),
                            "year": citation_data.get('year'),
                            "author": citation_data.get('author')
                        },
                        confidence=0.7,  # Lower confidence for AI-extracted
                        source_document=doc_id
                    )
                    
                    ai_references.append(reference)
                
                return ai_references[:10]  # Limit AI-extracted references
                
            except json.JSONDecodeError:
                self.log_warning("Failed to parse AI reference extraction response")
                return []
        
        except Exception as e:
            self.log_warning(f"AI reference extraction failed: {e}")
            return []
    
    def _deduplicate_references(self, references: List[Reference]) -> List[Reference]:
        """Remove duplicate references"""
        
        seen_references = set()
        unique_references = []
        
        for ref in references:
            # Create a hash based on text and position
            ref_hash = hashlib.md5(
                f"{ref.text}_{ref.position}_{ref.type}".encode()
            ).hexdigest()
            
            if ref_hash not in seen_references:
                seen_references.add(ref_hash)
                unique_references.append(ref)
        
        return unique_references
    
    async def _find_cross_references(self, documents: List[Document], 
                                   document_references: Dict[str, List[Reference]]) -> List[CrossReference]:
        """Find cross-references between documents"""
        
        cross_references = []
        doc_ids = list(document_references.keys())
        
        # Compare each pair of documents
        for i, doc_id_1 in enumerate(doc_ids):
            for doc_id_2 in doc_ids[i+1:]:
                
                refs_1 = document_references[doc_id_1]
                refs_2 = document_references[doc_id_2]
                
                # Find matching references
                for ref_1 in refs_1:
                    for ref_2 in refs_2:
                        similarity = self._calculate_reference_similarity(ref_1, ref_2)
                        
                        if similarity > 0.7:  # High similarity threshold
                            cross_ref = CrossReference(
                                source_doc_id=doc_id_1,
                                target_doc_id=doc_id_2,
                                reference_type="similar_reference",
                                strength=similarity,
                                context=f"Similar references: '{ref_1.text}' and '{ref_2.text}'",
                                bidirectional=True
                            )
                            cross_references.append(cross_ref)
        
        # Find internal document references
        for doc_id, doc_refs in document_references.items():
            for ref in doc_refs:
                if ref.type == "internal":
                    # Check if this refers to another document
                    target_doc = self._resolve_internal_reference(ref, doc_ids)
                    if target_doc and target_doc != doc_id:
                        cross_ref = CrossReference(
                            source_doc_id=doc_id,
                            target_doc_id=target_doc,
                            reference_type="internal_link",
                            strength=0.8,
                            context=f"Internal reference: {ref.text}"
                        )
                        cross_references.append(cross_ref)
        
        return cross_references
    
    def _calculate_reference_similarity(self, ref1: Reference, ref2: Reference) -> float:
        """Calculate similarity between two references"""
        
        # Simple similarity based on text overlap
        text1_words = set(ref1.text.lower().split())
        text2_words = set(ref2.text.lower().split())
        
        if not text1_words or not text2_words:
            return 0.0
        
        intersection = text1_words.intersection(text2_words)
        union = text1_words.union(text2_words)
        
        similarity = len(intersection) / len(union) if union else 0.0
        
        # Boost similarity if same type
        if ref1.type == ref2.type:
            similarity *= 1.2
        
        # Boost similarity if both have targets and they match
        if ref1.target and ref2.target and ref1.target == ref2.target:
            similarity *= 1.5
        
        return min(1.0, similarity)
    
    def _resolve_internal_reference(self, ref: Reference, doc_ids: List[str]) -> Optional[str]:
        """Try to resolve internal reference to a document ID"""
        
        # Simple heuristic: if reference mentions a document number
        if ref.target and ref.target.isdigit():
            doc_num = int(ref.target)
            if 1 <= doc_num <= len(doc_ids):
                return doc_ids[doc_num - 1]
        
        return None
    
    async def _validate_references(self, references: List[Reference]) -> List[Reference]:
        """Validate and enrich references"""
        
        validated_references = []
        
        for ref in references:
            # Basic validation
            if not ref.text or len(ref.text.strip()) < 3:
                continue
            
            # URL validation
            if ref.type == "url" and self.validate_urls:
                ref = await self._validate_url_reference(ref)
            
            # Citation validation
            elif ref.type == "citation":
                ref = await self._validate_citation_reference(ref)
            
            validated_references.append(ref)
        
        return validated_references
    
    async def _validate_url_reference(self, ref: Reference) -> Reference:
        """Validate URL reference"""
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.head(ref.text, timeout=5) as response:
                    ref.metadata = ref.metadata or {}
                    ref.metadata['url_status'] = response.status
                    ref.metadata['url_valid'] = response.status == 200
                    
                    if response.status == 200:
                        ref.confidence = min(1.0, ref.confidence + 0.1)
                    else:
                        ref.confidence = max(0.1, ref.confidence - 0.2)
        
        except Exception:
            ref.metadata = ref.metadata or {}
            ref.metadata['url_valid'] = False
            ref.confidence = max(0.1, ref.confidence - 0.3)
        
        return ref
    
    async def _validate_citation_reference(self, ref: Reference) -> Reference:
        """Validate citation reference"""
        
        # Basic citation validation
        ref.metadata = ref.metadata or {}
        
        # Check for common citation elements
        has_year = bool(re.search(r'\b(19|20)\d{2}\b', ref.text))
        has_author = bool(re.search(r'\b[A-Z][a-z]+\b', ref.text))
        
        ref.metadata['has_year'] = has_year
        ref.metadata['has_author'] = has_author
        
        # Adjust confidence based on validation
        if has_year and has_author:
            ref.confidence = min(1.0, ref.confidence + 0.1)
        elif has_year or has_author:
            ref.confidence = ref.confidence  # No change
        else:
            ref.confidence = max(0.3, ref.confidence - 0.2)
        
        return ref
    
    async def _create_reference_map(self, references: List[Reference], 
                                  cross_references: List[CrossReference]) -> Dict[str, Any]:
        """Create a comprehensive reference map"""
        
        # Group references by type
        by_type = {}
        for ref in references:
            if ref.type not in by_type:
                by_type[ref.type] = []
            by_type[ref.type].append(ref.id)
        
        # Group references by document
        by_document = {}
        for ref in references:
            doc_id = ref.source_document
            if doc_id not in by_document:
                by_document[doc_id] = []
            by_document[doc_id].append(ref.id)
        
        # Create cross-reference graph
        cross_ref_graph = {}
        for cref in cross_references:
            if cref.source_doc_id not in cross_ref_graph:
                cross_ref_graph[cref.source_doc_id] = []
            cross_ref_graph[cref.source_doc_id].append({
                "target": cref.target_doc_id,
                "type": cref.reference_type,
                "strength": cref.strength
            })
        
        return {
            "references_by_type": by_type,
            "references_by_document": by_document,
            "cross_reference_graph": cross_ref_graph,
            "total_references": len(references),
            "total_cross_references": len(cross_references)
        }
    
    def _calculate_reference_statistics(self, references: List[Reference], 
                                      cross_references: List[CrossReference]) -> Dict[str, Any]:
        """Calculate reference statistics"""
        
        if not references:
            return {"total_references": 0}
        
        # Count by type
        type_counts = {}
        confidence_scores = []
        
        for ref in references:
            type_counts[ref.type] = type_counts.get(ref.type, 0) + 1
            confidence_scores.append(ref.confidence)
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Count validated references
        validated_count = sum(1 for ref in references if ref.confidence > 0.7)
        
        return {
            "total_references": len(references),
            "references_by_type": type_counts,
            "average_confidence": round(avg_confidence, 3),
            "high_confidence_references": validated_count,
            "validation_rate": round(validated_count / len(references), 3),
            "total_cross_references": len(cross_references),
            "most_common_type": max(type_counts, key=type_counts.get) if type_counts else None
        }
    
    def get_reference_patterns(self) -> List[Dict[str, Any]]:
        """Get current reference patterns"""
        return [
            {
                "name": pattern.name,
                "pattern": pattern.pattern,
                "type": pattern.type,
                "confidence": pattern.confidence
            }
            for pattern in self.reference_patterns
        ]
    
    def add_reference_pattern(self, name: str, pattern: str, ref_type: str, 
                            confidence: float = 0.8):
        """Add a custom reference pattern"""
        
        new_pattern = ReferencePattern(
            name=name,
            pattern=pattern,
            type=ref_type,
            confidence=confidence
        )
        
        self.reference_patterns.append(new_pattern)
        self.log_info(f"Added reference pattern: {name}")
    
    async def extract_bibliography(self, document: Document) -> List[Dict[str, Any]]:
        """Extract bibliography/references section from document"""
        
        content = document.page_content
        
        # Look for common bibliography section markers
        bib_markers = [
            r'(?i)references?\s*$',
            r'(?i)bibliography\s*$',
            r'(?i)works?\s+cited\s*$',
            r'(?i)literature\s+cited\s*$'
        ]
        
        bib_start = None
        for marker in bib_markers:
            match = re.search(marker, content, re.MULTILINE)
            if match:
                bib_start = match.end()
                break
        
        if not bib_start:
            return []
        
        # Extract bibliography section
        bib_content = content[bib_start:]
        
        # Parse bibliography entries
        bibliography = []
        lines = bib_content.split('\n')
        
        current_entry = ""
        for line in lines:
            line = line.strip()
            if not line:
                if current_entry:
                    bibliography.append(current_entry.strip())
                    current_entry = ""
            else:
                current_entry += " " + line
        
        # Add final entry if exists
        if current_entry:
            bibliography.append(current_entry.strip())
        
        # Structure bibliography entries
        structured_bib = []
        for i, entry in enumerate(bibliography[:20]):  # Limit to 20 entries
            structured_bib.append({
                "id": f"bib_entry_{i}",
                "text": entry,
                "type": "bibliography_entry",
                "parsed": self._parse_bibliography_entry(entry)
            })
        
        return structured_bib
    
    def _parse_bibliography_entry(self, entry: str) -> Dict[str, Any]:
        """Parse individual bibliography entry"""
        
        parsed = {
            "author": None,
            "year": None,
            "title": None,
            "publication": None
        }
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', entry)
        if year_match:
            parsed["year"] = year_match.group(0)
        
        # Extract potential author (first capitalized words)
        author_match = re.match(r'^([A-Z][a-z]+(?:,\s+[A-Z]\.?)*)', entry)
        if author_match:
            parsed["author"] = author_match.group(1)
        
        # Extract title (often in quotes or italics)
        title_match = re.search(r'["""]([^"""]+)["""]', entry)
        if not title_match:
            title_match = re.search(r'\*([^*]+)\*', entry)
        if title_match:
            parsed["title"] = title_match.group(1)
        
        return parsed
        client.delete_collection(name=collection_name)
        logger.info(f"Deleted collection: {collection_name}")
    
    def reset_database(self):
        """Reset the entire database (use with caution)"""
        client = self.get_client()
        client.reset()
        logger.warning("Database has been reset")


# Convenience functions for common operations

def create_chroma_client(config: Dict[str, Any] = None) -> ChromaConfig:
    """Create a ChromaDB client with configuration"""
    return ChromaConfig(config)


def setup_default_collection(config: Dict[str, Any] = None):
    """Setup default collection for document processing"""
    chroma_config = ChromaConfig(config)
    collection = chroma_config.get_collection()
    
    logger.info(f"Default collection ready: {collection.name}")
    return chroma_config, collection


# Example usage and testing functions

def test_chroma_setup():
    """Test ChromaDB setup"""
    try:
        config = {
            'persist_directory': './test_chroma',
            'collection_name': 'test_collection',
            'embedding_provider': 'sentence-transformers',
            'embedding_model': 'all-MiniLM-L6-v2'
        }
        
        chroma_config = ChromaConfig(config)
        
        # Test collection creation
        collection = chroma_config.get_collection()
        print(f"Collection created: {collection.name}")
        
        # Test document addition
        test_docs = [
            "This is a test document about machine learning.",
            "Another document discussing natural language processing.",
            "A third document about vector databases."
        ]
        
        test_ids = ["test_1", "test_2", "test_3"]
        test_metadata = [
            {"category": "ml", "author": "test"},
            {"category": "nlp", "author": "test"},  
            {"category": "databases", "author": "test"}
        ]
        
        chroma_config.add_documents(
            documents=test_docs,
            ids=test_ids,
            metadatas=test_metadata
        )
        
        # Test querying
        results = chroma_config.query_documents(
            query_texts=["machine learning algorithms"],
            n_results=2
        )
        
        print(f"Query results: {len(results['documents'][0])} documents found")
        
        # Test collection info
        info = chroma_config.get_collection_info()
        print(f"Collection info: {info}")
        
        print("ChromaDB setup test completed successfully!")
        
    except Exception as e:
        print(f"ChromaDB setup test failed: {str(e)}")
        raise


if __name__ == "__main__":
    test_chroma_setup()