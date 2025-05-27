"""
Document Parser Agent
Advanced AI-powered document parsing and content extraction
"""

import asyncio
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import spacy
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import tiktoken

from .base_agent import BaseAgent, AgentResult


class DocumentStructure(BaseModel):
    """Structured representation of document content"""
    title: Optional[str] = Field(description="Document title")
    sections: List[Dict[str, Any]] = Field(description="Document sections")
    entities: List[Dict[str, str]] = Field(description="Named entities")
    keywords: List[str] = Field(description="Key terms and phrases")
    summary: Optional[str] = Field(description="Document summary")
    language: str = Field(description="Detected language")
    document_type: str = Field(description="Type of document")


@dataclass
class ChunkMetadata:
    """Metadata for document chunks"""
    chunk_id: str
    chunk_index: int
    chunk_length: int
    start_char: int
    end_char: int
    section: Optional[str] = None
    page: Optional[int] = None
    source: str = ""


class DocumentParser(BaseAgent):
    """
    Document Parser Agent
    
    Capabilities:
    - Intelligent document chunking with multiple strategies
    - Content structure analysis and extraction
    - Named entity recognition and extraction
    - Keyword and key phrase identification
    - Document fingerprinting and deduplication
    - Multi-language support
    - Content quality assessment
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Load parser-specific configuration
        self.parser_config = self.config.get('agents', {}).get('parser', {})
        self.embeddings_config = self.config.get('embeddings', {})
        
        # Chunking configuration
        self.chunk_size = self.embeddings_config.get('chunk_size', 1000)
        self.chunk_overlap = self.embeddings_config.get('chunk_overlap', 200)
        self.chunk_strategy = self.embeddings_config.get('chunk_strategy', 'recursive')
        
        # Processing options
        self.extract_metadata = self.parser_config.get('extract_metadata', True)
        self.detect_language = self.parser_config.get('detect_language', True)
        self.extract_entities = self.parser_config.get('extract_entities', True)
        self.extract_keywords = self.parser_config.get('extract_keywords', True)
        self.max_processing_time = self.parser_config.get('max_processing_time', 300)
        
        # Initialize text splitters
        self._initialize_text_splitters()
        
        # Initialize NLP models
        self._initialize_nlp_models()
        
        # Content analysis prompts
        self._setup_analysis_prompts()
        
        self.log_info("DocumentParser initialized")
    
    def _initialize_text_splitters(self):
        """Initialize various text splitting strategies"""
        self.text_splitters = {
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            ),
            'sentence': SentenceTransformersTokenTextSplitter(
                chunk_overlap=self.chunk_overlap,
                tokens_per_chunk=self.chunk_size // 4  # Approximate token to char ratio
            )
        }
        
        # Try to initialize SpaCy splitter
        try:
            self.text_splitters['spacy'] = SpacyTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        except Exception as e:
            self.log_warning(f"Could not initialize SpaCy text splitter: {e}")
    
    def _initialize_nlp_models(self):
        """Initialize NLP models for entity extraction"""
        self.nlp_models = {}
        
        # Try to load SpaCy models
        try:
            import spacy
            # Try different language models
            for model_name in ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']:
                try:
                    self.nlp_models['en'] = spacy.load(model_name)
                    self.log_info(f"Loaded SpaCy model: {model_name}")
                    break
                except OSError:
                    continue
            
            if 'en' not in self.nlp_models:
                self.log_warning("No SpaCy English model found. Entity extraction will be limited.")
                
        except ImportError:
            self.log_warning("SpaCy not available. Entity extraction will be limited.")
    
    def _setup_analysis_prompts(self):
        """Setup prompts for AI-powered content analysis"""
        
        self.structure_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document analyst. Analyze the provided document content and extract structured information.

Your task:
1. Identify the document title if present
2. Break down the content into logical sections
3. Extract key topics and themes
4. Identify the document type (report, article, manual, etc.)
5. Provide a brief summary

Be precise and only extract information that is clearly present in the text."""),
            
            ("human", """Document Content:
{content}

Please analyze this document and provide a structured response with:
- Title (if identifiable)
- Main sections with headings
- Document type
- Key topics/themes
- Brief summary (2-3 sentences)

Format as JSON.""")
        ])
        
        self.entity_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting named entities from text. 
            
Identify and categorize:
- PERSON: People's names
- ORGANIZATION: Companies, institutions, agencies
- LOCATION: Cities, countries, addresses
- DATE: Specific dates and time periods
- MONEY: Monetary amounts
- PRODUCT: Products, services, technologies
- EVENT: Named events, conferences, meetings

Only extract entities that are clearly mentioned in the text."""),
            
            ("human", """Text: {text}

Extract named entities and return as JSON list with format:
[{"text": "entity_text", "label": "CATEGORY", "start": start_pos, "end": end_pos}]""")
        ])
        
        self.keyword_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at identifying key terms and phrases from documents.

Extract the most important keywords and phrases that represent:
- Main topics and themes
- Technical terms and jargon
- Important concepts
- Key processes or methods

Provide 10-20 of the most relevant terms, ranked by importance."""),
            
            ("human", """Document: {text}

Extract key terms and phrases. Return as JSON array of strings, ordered by importance.""")
        ])
    
    async def process(self, input_data: List[Document], **kwargs) -> AgentResult:
        """
        Process documents for parsing and analysis
        
        Args:
            input_data: List of Document objects to parse
            **kwargs: Additional parameters
                - chunk_strategy: Override default chunking strategy
                - extract_structure: Whether to extract document structure
                - max_chunks: Maximum number of chunks to create
        
        Returns:
            AgentResult with parsed content, chunks, and metadata
        """
        try:
            if not input_data:
                return AgentResult(
                    success=False,
                    error="No documents provided for parsing"
                )
            
            # Combine all document content
            combined_content = self._combine_documents(input_data)
            
            # Extract basic metadata
            basic_metadata = await self._extract_basic_metadata(input_data, combined_content)
            
            # Chunk the content
            chunk_strategy = kwargs.get('chunk_strategy', self.chunk_strategy)
            chunked_documents = await self._create_chunks(
                combined_content, 
                input_data, 
                chunk_strategy,
                kwargs.get('max_chunks')
            )
            
            # Extract detailed information if enabled
            detailed_analysis = {}
            if self.extract_metadata or kwargs.get('extract_structure', False):
                detailed_analysis = await self._perform_detailed_analysis(combined_content)
            
            # Calculate content fingerprint
            fingerprint = self._calculate_fingerprint(combined_content)
            
            # Compile result data
            result_data = {
                "original_documents": input_data,
                "chunked_documents": chunked_documents,
                "metadata": {
                    **basic_metadata,
                    **detailed_analysis,
                    "fingerprint": fingerprint,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "chunk_count": len(chunked_documents),
                "total_content_length": len(combined_content)
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                confidence=0.9,
                metadata={
                    "chunk_strategy": chunk_strategy,
                    "processing_features": {
                        "metadata_extraction": self.extract_metadata,
                        "entity_extraction": self.extract_entities,
                        "keyword_extraction": self.extract_keywords
                    }
                }
            )
            
        except Exception as e:
            self.log_error(f"Error in document parsing: {str(e)}")
            return AgentResult(
                success=False,
                error=f"Document parsing failed: {str(e)}"
            )
    
    def _combine_documents(self, documents: List[Document]) -> str:
        """Combine multiple documents into single content string"""
        combined_parts = []
        
        for i, doc in enumerate(documents):
            # Add document separator if multiple documents
            if i > 0:
                combined_parts.append(f"\n{'='*50}\n")
            
            # Add source information if available
            source = doc.metadata.get('source', f'Document {i+1}')
            page = doc.metadata.get('page')
            if page:
                combined_parts.append(f"[Source: {source}, Page: {page}]\n")
            else:
                combined_parts.append(f"[Source: {source}]\n")
            
            combined_parts.append(doc.page_content)
        
        return '\n'.join(combined_parts)
    
    async def _extract_basic_metadata(self, documents: List[Document], content: str) -> Dict[str, Any]:
        """Extract basic metadata from documents"""
        # Count statistics
        char_count = len(content)
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?]+', content))
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Language detection
        language = "unknown"
        if self.detect_language:
            language = await self._detect_language(content)
        
        # Source information
        sources = list(set(doc.metadata.get('source', 'Unknown') for doc in documents))
        page_count = len(documents)
        
        return {
            "page_count": page_count,
            "character_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "language": language,
            "sources": sources,
            "document_count": len(documents)
        }
    
    async def _detect_language(self, text: str) -> str:
        """Detect document language"""
        try:
            from langdetect import detect, detect_langs
            
            # Use first 1000 characters for detection
            sample_text = text[:1000]
            detected = detect(sample_text)
            
            # Get confidence scores
            lang_probs = detect_langs(sample_text)
            confidence = max(lang.prob for lang in lang_probs)
            
            if confidence > 0.8:
                return detected
            else:
                return f"{detected} (low confidence)"
                
        except Exception as e:
            self.log_warning(f"Language detection failed: {e}")
            return "unknown"
    
    async def _create_chunks(self, content: str, original_docs: List[Document], 
                           strategy: str, max_chunks: Optional[int] = None) -> List[Document]:
        """Create document chunks using specified strategy"""
        
        # Select text splitter
        if strategy not in self.text_splitters:
            self.log_warning(f"Unknown chunk strategy '{strategy}', using 'recursive'")
            strategy = 'recursive'
        
        text_splitter = self.text_splitters[strategy]
        
        # Create chunks
        chunks = text_splitter.split_text(content)
        
        # Limit chunks if specified
        if max_chunks and len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
            self.log_info(f"Limited chunks to {max_chunks} (from {len(chunks)})")
        
        # Convert to Document objects with metadata
        chunked_documents = []
        
        for i, chunk_text in enumerate(chunks):
            # Calculate character positions
            start_char = content.find(chunk_text)
            end_char = start_char + len(chunk_text) if start_char != -1 else len(chunk_text)
            
            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                chunk_id=f"chunk_{i}",
                chunk_index=i,
                chunk_length=len(chunk_text),
                start_char=start_char,
                end_char=end_char,
                source=original_docs[0].metadata.get('source', 'Unknown') if original_docs else 'Unknown'
            )
            
            # Try to identify which original document this chunk belongs to
            source_doc_metadata = self._identify_source_document(chunk_text, original_docs)
            
            # Combine metadata
            combined_metadata = {
                **source_doc_metadata,
                **chunk_metadata.__dict__,
                "chunk_strategy": strategy
            }
            
            chunked_documents.append(Document(
                page_content=chunk_text,
                metadata=combined_metadata
            ))
        
        return chunked_documents
    
    def _identify_source_document(self, chunk_text: str, original_docs: List[Document]) -> Dict[str, Any]:
        """Identify which original document a chunk likely came from"""
        if len(original_docs) <= 1:
            return original_docs[0].metadata if original_docs else {}
        
        # Find best match based on text overlap
        best_match = original_docs[0]
        best_score = 0
        
        for doc in original_docs:
            # Simple overlap score based on common words
            chunk_words = set(chunk_text.lower().split())
            doc_words = set(doc.page_content.lower().split())
            
            if len(chunk_words) > 0:
                overlap = len(chunk_words.intersection(doc_words))
                score = overlap / len(chunk_words)
                
                if score > best_score:
                    best_score = score
                    best_match = doc
        
        return best_match.metadata
    
    async def _perform_detailed_analysis(self, content: str) -> Dict[str, Any]:
        """Perform detailed AI-powered content analysis"""
        analysis_results = {}
        
        # Content length check
        if len(content) > 50000:  # Limit content for LLM analysis
            sample_content = content[:25000] + "\n...\n" + content[-25000:]
            self.log_info("Content truncated for detailed analysis due to length")
        else:
            sample_content = content
        
        try:
            # Structure analysis
            if self.extract_metadata:
                structure_result = await self._analyze_document_structure(sample_content)
                analysis_results.update(structure_result)
            
            # Named entity extraction
            if self.extract_entities:
                entities = await self._extract_entities(sample_content)
                analysis_results["entities"] = entities
            
            # Keyword extraction
            if self.extract_keywords:
                keywords = await self._extract_keywords(sample_content)
                analysis_results["keywords"] = keywords
                
        except Exception as e:
            self.log_error(f"Error in detailed analysis: {e}")
            analysis_results["analysis_error"] = str(e)
        
        return analysis_results
    
    async def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure using LLM"""
        try:
            llm = self.get_llm_client()
            
            prompt = self.structure_analysis_prompt.format_prompt(content=content[:8000])
            response = await llm.ainvoke(prompt.to_messages())
            
            # Try to parse JSON response
            try:
                structure_data = json.loads(response.content)
                return {
                    "title": structure_data.get("title"),
                    "document_type": structure_data.get("document_type", "unknown"),
                    "sections": structure_data.get("sections", []),
                    "summary": structure_data.get("summary"),
                    "topics": structure_data.get("topics", [])
                }
            except json.JSONDecodeError:
                # Fallback to basic structure extraction
                return await self._extract_basic_structure(content)
                
        except Exception as e:
            self.log_warning(f"LLM structure analysis failed: {e}")
            return await self._extract_basic_structure(content)
    
    async def _extract_basic_structure(self, content: str) -> Dict[str, Any]:
        """Extract basic document structure using regex patterns"""
        structure = {
            "title": None,
            "document_type": "document",
            "sections": [],
            "summary": None,
            "topics": []
        }
        
        # Try to find title (first line, or largest text, or marked heading)
        lines = content.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        # Simple title detection
        if len(first_line) < 100 and len(first_line) > 5:
            structure["title"] = first_line
        
        # Find section headings (lines that end with newlines and start with caps)
        sections = []
        heading_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS headers
            r'^(\d+\.?\s+[A-Z].+)$',  # Numbered sections
        ]
        
        for line in lines:
            line = line.strip()
            for pattern in heading_patterns:
                match = re.match(pattern, line)
                if match and len(line) > 3 and len(line) < 80:
                    sections.append(match.group(1))
                    break
        
        structure["sections"] = sections[:20]  # Limit to 20 sections
        
        return structure
    
    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities from content"""
        entities = []
        
        # Use SpaCy if available
        if 'en' in self.nlp_models:
            try:
                nlp = self.nlp_models['en']
                # Process in chunks to avoid memory issues
                chunk_size = 100000
                
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i + chunk_size]
                    doc = nlp(chunk)
                    
                    for ent in doc.ents:
                        entities.append({
                            "text": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char + i,
                            "end": ent.end_char + i,
                            "description": spacy.explain(ent.label_) or ent.label_
                        })
                        
            except Exception as e:
                self.log_warning(f"SpaCy entity extraction failed: {e}")
        
        # Fallback to LLM-based extraction for key entities
        if not entities or len(entities) < 5:
            try:
                llm_entities = await self._extract_entities_with_llm(content[:5000])
                entities.extend(llm_entities)
            except Exception as e:
                self.log_warning(f"LLM entity extraction failed: {e}")
        
        # Remove duplicates and sort by position
        unique_entities = []
        seen = set()
        
        for entity in sorted(entities, key=lambda x: x.get('start', 0)):
            key = (entity['text'].lower(), entity['label'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities[:50]  # Limit to 50 entities
    
    async def _extract_entities_with_llm(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities using LLM"""
        try:
            llm = self.get_llm_client()
            
            prompt = self.entity_extraction_prompt.format_prompt(text=content)
            response = await llm.ainvoke(prompt.to_messages())
            
            # Parse JSON response
            entities_data = json.loads(response.content)
            return entities_data if isinstance(entities_data, list) else []
            
        except Exception as e:
            self.log_warning(f"LLM entity extraction error: {e}")
            return []
    
    async def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords and key phrases"""
        keywords = []
        
        # Simple frequency-based keyword extraction
        try:
            # Clean text and tokenize
            import re
            from collections import Counter
            
            # Remove special characters and normalize
            clean_text = re.sub(r'[^\w\s]', ' ', content.lower())
            words = clean_text.split()
            
            # Filter out common stop words
            stop_words = set([
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
            ])
            
            # Filter words
            filtered_words = [w for w in words if len(w) > 3 and w not in stop_words]
            
            # Get most common words
            word_freq = Counter(filtered_words)
            common_words = [word for word, freq in word_freq.most_common(20) if freq > 2]
            
            keywords.extend(common_words)
            
        except Exception as e:
            self.log_warning(f"Basic keyword extraction failed: {e}")
        
        # Use LLM for better keyword extraction
        try:
            llm_keywords = await self._extract_keywords_with_llm(content[:8000])
            keywords.extend(llm_keywords)
        except Exception as e:
            self.log_warning(f"LLM keyword extraction failed: {e}")
        
        # Remove duplicates and return
        return list(set(keywords))[:30]  # Limit to 30 keywords
    
    async def _extract_keywords_with_llm(self, content: str) -> List[str]:
        """Extract keywords using LLM"""
        try:
            llm = self.get_llm_client()
            
            prompt = self.keyword_extraction_prompt.format_prompt(text=content)
            response = await llm.ainvoke(prompt.to_messages())
            
            # Parse JSON response
            keywords_data = json.loads(response.content)
            return keywords_data if isinstance(keywords_data, list) else []
            
        except Exception as e:
            self.log_warning(f"LLM keyword extraction error: {e}")
            return []
    
    def _calculate_fingerprint(self, content: str) -> str:
        """Calculate content fingerprint for deduplication"""
        # Normalize content for fingerprinting
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        
        # Create SHA-256 hash
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def get_chunk_strategies(self) -> List[str]:
        """Get available chunking strategies"""
        return list(self.text_splitters.keys())
    
    def set_chunk_size(self, size: int):
        """Update chunk size and reinitialize splitters"""
        self.chunk_size = size
        self._initialize_text_splitters()
        self.log_info(f"Chunk size updated to {size}")
    
    def set_chunk_overlap(self, overlap: int):
        """Update chunk overlap and reinitialize splitters"""
        self.chunk_overlap = overlap
        self._initialize_text_splitters()
        self.log_info(f"Chunk overlap updated to {overlap}")
    
    def estimate_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Estimate token count for text"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4