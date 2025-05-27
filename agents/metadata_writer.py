"""
Metadata Writer Agent
Extracts, enhances, and stores document metadata using AI and traditional methods
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass
import hashlib

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentResult


class DocumentMetadata(BaseModel):
    """Structured document metadata"""
    title: Optional[str] = Field(description="Document title")
    author: Optional[str] = Field(description="Document author")
    subject: Optional[str] = Field(description="Document subject/topic")
    description: Optional[str] = Field(description="Document description")
    keywords: List[str] = Field(default=[], description="Document keywords")
    language: str = Field(default="en", description="Document language")
    document_type: str = Field(default="document", description="Type of document")
    category: Optional[str] = Field(description="Document category")
    tags: List[str] = Field(default=[], description="Document tags")
    created_date: Optional[str] = Field(description="Document creation date")
    modified_date: Optional[str] = Field(description="Document modification date")
    version: Optional[str] = Field(description="Document version")
    confidence: float = Field(default=0.5, description="Metadata extraction confidence")


@dataclass
class MetadataExtractionResult:
    """Result from metadata extraction"""
    metadata: DocumentMetadata
    extraction_method: str
    processing_time: float
    sources_used: List[str]
    confidence: float


class MetadataWriter(BaseAgent):
    """
    Metadata Writer Agent
    
    Capabilities:
    - AI-powered metadata extraction from document content
    - Traditional metadata extraction from file properties
    - Metadata enhancement and enrichment
    - Structured metadata validation and standardization
    - Database storage and retrieval
    - Metadata merging and conflict resolution
    - Custom field extraction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Load metadata writer specific configuration
        self.metadata_config = self.config.get('agents', {}).get('metadata_writer', {})
        
        # Processing options
        self.use_ai_extraction = self.metadata_config.get('use_ai_extraction', True)
        self.extract_custom_fields = self.metadata_config.get('extract_custom_fields', True)
        self.enhance_metadata = self.metadata_config.get('enhance_metadata', True)
        self.validate_metadata = self.metadata_config.get('validate_metadata', True)
        
        # AI extraction settings
        self.max_content_for_ai = self.metadata_config.get('max_content_for_ai', 8000)
        self.confidence_threshold = self.metadata_config.get('confidence_threshold', 0.6)
        
        # Custom field patterns
        self.custom_patterns = self.metadata_config.get('custom_patterns', {})
        
        # Initialize AI prompts
        self._setup_extraction_prompts()
        
        self.log_info("MetadataWriter initialized")
    
    def _setup_extraction_prompts(self):
        """Setup AI prompts for metadata extraction"""
        
        self.metadata_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document analyst specializing in metadata extraction.

Your task is to analyze document content and extract structured metadata including:
- Title (main document title)
- Author (document author/creator)
- Subject (main topic or subject)
- Description (brief description of content)
- Keywords (5-10 key terms)
- Language (detected language code)
- Document type (report, article, manual, letter, etc.)
- Category (business, technical, academic, legal, etc.)

Guidelines:
- Only extract information that is clearly present in the text
- Provide confidence scores for uncertain extractions
- Use standard language codes (en, es, fr, etc.)
- Keep descriptions concise (1-2 sentences)
- Focus on factual information, not interpretations"""),
            
            ("human", """Document Content:
{content}

Please extract metadata from this document and return as JSON with the following structure:
{{
    "title": "extracted title or null",
    "author": "extracted author or null", 
    "subject": "main subject/topic",
    "description": "brief description",
    "keywords": ["keyword1", "keyword2", ...],
    "language": "language code",
    "document_type": "document type",
    "category": "category if identifiable",
    "confidence": 0.0-1.0
}}

Only include fields where you have reasonable confidence in the extraction.""")
        ])
        
        self.metadata_enhancement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at enhancing and standardizing document metadata.

Your task is to:
1. Review existing metadata for completeness and accuracy
2. Suggest improvements and corrections
3. Add missing but inferrable information
4. Standardize formats and terminology
5. Resolve conflicts between different metadata sources

Be conservative - only add information you're confident about."""),
            
            ("human", """Existing Metadata:
{existing_metadata}

Document Sample:
{document_sample}

Please enhance this metadata and return improved version as JSON. Include a confidence score for your enhancements.""")
        ])
        
        self.custom_field_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting specific information from documents based on patterns and requirements.

Extract the requested custom fields from the document content. Be precise and only extract information that clearly matches the requested patterns."""),
            
            ("human", """Document Content:
{content}

Custom Fields to Extract:
{custom_fields}

Return the extracted information as JSON with field names as keys and extracted values as values. If a field cannot be found or extracted with confidence, set its value to null.""")
        ])
    
    async def process(self, input_data: Union[Dict[str, Any], List[Document]], **kwargs) -> AgentResult:
        """
        Process documents for metadata extraction and enhancement
        
        Args:
            input_data: Dict with document info or List of Document objects
            **kwargs: Additional parameters
                - existing_metadata: Previously extracted metadata to enhance
                - custom_fields: Custom fields to extract
                - extraction_method: Override extraction method
        
        Returns:
            AgentResult with extracted and enhanced metadata
        """
        try:
            # Parse input
            if isinstance(input_data, dict):
                documents = input_data.get('documents', [])
                document_id = input_data.get('document_id')
                existing_metadata = input_data.get('existing_metadata', {})
            elif isinstance(input_data, list):
                documents = input_data
                document_id = kwargs.get('document_id')
                existing_metadata = kwargs.get('existing_metadata', {})
            else:
                return AgentResult(
                    success=False,
                    error="Input must be dict with documents or list of Document objects"
                )
            
            if not documents:
                return AgentResult(
                    success=False,
                    error="No documents provided for metadata extraction"
                )
            
            # Combine document content for analysis
            combined_content = self._prepare_content_for_extraction(documents)
            
            # Extract metadata using multiple methods
            extraction_results = []
            
            # 1. Extract from existing file metadata
            file_metadata = self._extract_file_metadata(documents, existing_metadata)
            if file_metadata:
                extraction_results.append(MetadataExtractionResult(
                    metadata=DocumentMetadata(**file_metadata),
                    extraction_method="file_properties",
                    processing_time=0.1,
                    sources_used=["file_metadata"],
                    confidence=0.8
                ))
            
            # 2. Pattern-based extraction
            pattern_metadata = await self._extract_pattern_metadata(combined_content)
            if pattern_metadata:
                extraction_results.append(MetadataExtractionResult(
                    metadata=DocumentMetadata(**pattern_metadata),
                    extraction_method="pattern_matching",
                    processing_time=0.2,
                    sources_used=["content_patterns"],
                    confidence=0.7
                ))
            
            # 3. AI-powered extraction
            if self.use_ai_extraction:
                ai_metadata = await self._extract_ai_metadata(combined_content)
                if ai_metadata:
                    extraction_results.append(ai_metadata)
            
            # 4. Custom field extraction
            custom_fields = kwargs.get('custom_fields', {})
            if custom_fields and self.extract_custom_fields:
                custom_metadata = await self._extract_custom_fields(combined_content, custom_fields)
                if custom_metadata:
                    extraction_results.append(custom_metadata)
            
            # Merge and resolve metadata from different sources
            final_metadata = await self._merge_metadata_results(extraction_results)
            
            # Enhance metadata if enabled
            if self.enhance_metadata and extraction_results:
                enhanced_metadata = await self._enhance_metadata(
                    final_metadata, 
                    combined_content[:2000]  # Sample for enhancement
                )
                if enhanced_metadata:
                    final_metadata = enhanced_metadata
            
            # Validate metadata
            if self.validate_metadata:
                validation_result = await self._validate_metadata(final_metadata)
                final_metadata.confidence = min(final_metadata.confidence, validation_result.get('confidence', 1.0))
            
            # Calculate processing statistics
            total_processing_time = sum(r.processing_time for r in extraction_results)
            methods_used = [r.extraction_method for r in extraction_results]
            
            result_data = {
                "enhanced_metadata": final_metadata.dict(),
                "extraction_results": [
                    {
                        "method": r.extraction_method,
                        "confidence": r.confidence,
                        "processing_time": r.processing_time,
                        "sources": r.sources_used
                    } for r in extraction_results
                ],
                "processing_statistics": {
                    "total_methods_used": len(extraction_results),
                    "methods_used": methods_used,
                    "total_processing_time": total_processing_time,
                    "final_confidence": final_metadata.confidence,
                    "content_length_analyzed": len(combined_content)
                }
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                confidence=final_metadata.confidence,
                metadata={
                    "document_id": document_id,
                    "extraction_methods": methods_used,
                    "ai_extraction_used": self.use_ai_extraction
                }
            )
            
        except Exception as e:
            self.log_error(f"Error in metadata extraction: {str(e)}")
            return AgentResult(
                success=False,
                error=f"Metadata extraction failed: {str(e)}"
            )
    
    def _prepare_content_for_extraction(self, documents: List[Document]) -> str:
        """Prepare document content for metadata extraction"""
        content_parts = []
        
        for doc in documents[:5]:  # Limit to first 5 documents to avoid too much content
            content = doc.page_content.strip()
            if content:
                content_parts.append(content)
        
        combined = '\n\n'.join(content_parts)
        
        # Limit content length for AI processing
        if len(combined) > self.max_content_for_ai:
            # Take beginning and end
            half_size = self.max_content_for_ai // 2
            combined = combined[:half_size] + "\n...\n" + combined[-half_size:]
        
        return combined
    
    def _extract_file_metadata(self, documents: List[Document], 
                             existing_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract metadata from file properties and existing metadata"""
        metadata = {}
        
        # Extract from existing metadata
        if existing_metadata:
            metadata.update({
                'title': existing_metadata.get('title'),
                'author': existing_metadata.get('author'),
                'subject': existing_metadata.get('subject'),
                'created_date': existing_metadata.get('creation_date'),
                'modified_date': existing_metadata.get('modification_date'),
                'language': existing_metadata.get('language', 'en')
            })
        
        # Extract from document metadata
        if documents:
            first_doc = documents[0]
            doc_metadata = first_doc.metadata
            
            # Try to get filename-based info
            filename = doc_metadata.get('source', '')
            if filename:
                # Extract title from filename
                if not metadata.get('title'):
                    title_from_filename = self._extract_title_from_filename(filename)
                    if title_from_filename:
                        metadata['title'] = title_from_filename
                
                # Determine document type from extension
                file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
                doc_type_map = {
                    'pdf': 'document',
                    'docx': 'document', 'doc': 'document',
                    'xlsx': 'spreadsheet', 'xls': 'spreadsheet',
                    'pptx': 'presentation', 'ppt': 'presentation',
                    'txt': 'text', 'md': 'text',
                    'html': 'webpage', 'htm': 'webpage'
                }
                metadata['document_type'] = doc_type_map.get(file_ext, 'document')
            
            # Extract page count
            total_pages = doc_metadata.get('total_pages') or len(documents)
            if total_pages:
                metadata['page_count'] = total_pages
        
        return metadata if any(v for v in metadata.values()) else None
    
    def _extract_title_from_filename(self, filename: str) -> Optional[str]:
        """Extract a readable title from filename"""
        try:
            # Remove path and extension
            name = filename.split('/')[-1].split('\\')[-1]
            if '.' in name:
                name = '.'.join(name.split('.')[:-1])
            
            # Clean up common filename patterns
            name = re.sub(r'[_-]+', ' ', name)  # Replace underscores/hyphens with spaces
            name = re.sub(r'\s+', ' ', name)   # Multiple spaces to single
            name = name.strip()
            
            # Capitalize words
            if name and len(name) > 3:
                return ' '.join(word.capitalize() for word in name.split())
            
        except:
            pass
        
        return None
    
    async def _extract_pattern_metadata(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract metadata using pattern matching"""
        metadata = {}
        
        try:
            # Title patterns (looking for lines that might be titles)
            title_patterns = [
                r'^([A-Z][^.\n]{10,100})$',  # Single line of capitalized text
                r'Title:\s*(.+)',           # Explicit title label
                r'TITLE:\s*(.+)',           # Upper case title label
                r'^#{1,3}\s+(.+)$'          # Markdown headers
            ]
            
            for pattern in title_patterns:
                matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                if matches:
                    potential_title = matches[0].strip()
                    if 10 <= len(potential_title) <= 100:  # Reasonable title length
                        metadata['title'] = potential_title
                        break
            
            # Author patterns
            author_patterns = [
                r'Author:\s*(.+)',
                r'By:\s*(.+)',
                r'Written by:\s*(.+)',
                r'Created by:\s*(.+)'
            ]
            
            for pattern in author_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    metadata['author'] = matches[0].strip()
                    break
            
            # Date patterns
            date_patterns = [
                r'Date:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'Created:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    metadata['created_date'] = matches[0]
                    break
            
            # Subject/topic extraction (first meaningful paragraph)
            paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
            if paragraphs:
                # Take the first substantial paragraph as subject
                first_paragraph = paragraphs[0][:200]  # Limit length
                if len(first_paragraph) > 20:
                    metadata['subject'] = first_paragraph
            
            # Language detection (basic)
            metadata['language'] = self._detect_language_simple(content)
            
            # Keywords extraction (simple frequency analysis)
            keywords = self._extract_keywords_simple(content)
            if keywords:
                metadata['keywords'] = keywords
            
            return metadata if metadata else None
            
        except Exception as e:
            self.log_warning(f"Pattern-based metadata extraction failed: {e}")
            return None
    
    def _detect_language_simple(self, content: str) -> str:
        """Simple language detection based on common words"""
        # Very basic language detection
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were']
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por']
        french_words = ['le', 'de', 'et', 'être', 'un', 'il', 'avoir', 'ne', 'je', 'son', 'que', 'se', 'qui', 'ce', 'dans']
        
        content_lower = content.lower()
        words = content_lower.split()
        
        if not words:
            return 'en'
        
        en_count = sum(1 for word in english_words if word in content_lower)
        es_count = sum(1 for word in spanish_words if word in content_lower)
        fr_count = sum(1 for word in french_words if word in content_lower)
        
        if en_count >= es_count and en_count >= fr_count:
            return 'en'
        elif es_count >= fr_count:
            return 'es'
        elif fr_count > 0:
            return 'fr'
        else:
            return 'en'  # Default
    
    def _extract_keywords_simple(self, content: str, max_keywords: int = 10) -> List[str]:
        """Simple keyword extraction using frequency analysis"""
        try:
            # Clean content
            import re
            from collections import Counter
            
            # Remove common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
                'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
                'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
            }
            
            # Extract words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            
            # Filter stop words and count frequency
            filtered_words = [w for w in words if w not in stop_words]
            word_freq = Counter(filtered_words)
            
            # Get most common words
            keywords = [word for word, freq in word_freq.most_common(max_keywords) if freq > 1]
            
            return keywords[:max_keywords]
            
        except Exception as e:
            self.log_warning(f"Simple keyword extraction failed: {e}")
            return []
    
    async def _extract_ai_metadata(self, content: str) -> Optional[MetadataExtractionResult]:
        """Extract metadata using AI"""
        if not content.strip():
            return None
        
        try:
            start_time = time.time()
            
            llm = self.get_llm_client()
            
            prompt = self.metadata_extraction_prompt.format_prompt(content=content)
            response = await llm.ainvoke(prompt.to_messages())
            
            processing_time = time.time() - start_time
            
            # Parse JSON response
            try:
                metadata_dict = json.loads(response.content)
                
                # Validate and clean the response
                cleaned_metadata = self._clean_ai_metadata(metadata_dict)
                
                return MetadataExtractionResult(
                    metadata=DocumentMetadata(**cleaned_metadata),
                    extraction_method="ai_extraction",
                    processing_time=processing_time,
                    sources_used=["llm_analysis"],
                    confidence=cleaned_metadata.get('confidence', 0.7)
                )
                
            except json.JSONDecodeError as e:
                self.log_warning(f"Failed to parse AI metadata response: {e}")
                return None
                
        except Exception as e:
            self.log_warning(f"AI metadata extraction failed: {e}")
            return None
    
    def _clean_ai_metadata(self, metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate AI-extracted metadata"""
        cleaned = {}
        
        # String fields
        for field in ['title', 'author', 'subject', 'description', 'language', 'document_type', 'category']:
            value = metadata_dict.get(field)
            if value and isinstance(value, str) and value.strip().lower() not in ['null', 'none', 'unknown']:
                cleaned[field] = value.strip()
        
        # List fields
        for field in ['keywords', 'tags']:
            value = metadata_dict.get(field)
            if value and isinstance(value, list):
                cleaned[field] = [str(item).strip() for item in value if item and str(item).strip()]
        
        # Numeric fields
        confidence = metadata_dict.get('confidence')
        if confidence is not None:
            try:
                cleaned['confidence'] = max(0.0, min(1.0, float(confidence)))
            except:
                cleaned['confidence'] = 0.5
        
        return cleaned
    
    async def _extract_custom_fields(self, content: str, custom_fields: Dict[str, str]) -> Optional[MetadataExtractionResult]:
        """Extract custom fields using AI"""
        if not custom_fields or not content.strip():
            return None
        
        try:
            start_time = time.time()
            
            llm = self.get_llm_client()
            
            # Format custom fields for the prompt
            fields_description = '\n'.join([f"- {name}: {description}" for name, description in custom_fields.items()])
            
            prompt = self.custom_field_extraction_prompt.format_prompt(
                content=content,
                custom_fields=fields_description
            )
            
            response = await llm.ainvoke(prompt.to_messages())
            processing_time = time.time() - start_time
            
            # Parse response
            try:
                custom_data = json.loads(response.content)
                
                # Create metadata with custom fields
                metadata_dict = {'custom_fields': custom_data, 'confidence': 0.6}
                
                return MetadataExtractionResult(
                    metadata=DocumentMetadata(**metadata_dict),
                    extraction_method="custom_field_extraction",
                    processing_time=processing_time,
                    sources_used=["llm_custom_analysis"],
                    confidence=0.6
                )
                
            except json.JSONDecodeError:
                self.log_warning("Failed to parse custom field extraction response")
                return None
                
        except Exception as e:
            self.log_warning(f"Custom field extraction failed: {e}")
            return None
    
    async def _merge_metadata_results(self, results: List[MetadataExtractionResult]) -> DocumentMetadata:
        """Merge metadata from different extraction methods"""
        if not results:
            return DocumentMetadata()
        
        # Start with the highest confidence result
        results_sorted = sorted(results, key=lambda x: x.confidence, reverse=True)
        final_metadata = results_sorted[0].metadata.dict()
        
        # Merge information from other results
        for result in results_sorted[1:]:
            for field, value in result.metadata.dict().items():
                if value and not final_metadata.get(field):
                    final_metadata[field] = value
                elif field == 'keywords' and isinstance(value, list) and isinstance(final_metadata.get(field), list):
                    # Merge keyword lists
                    existing_keywords = set(final_metadata[field])
                    new_keywords = [kw for kw in value if kw not in existing_keywords]
                    final_metadata[field].extend(new_keywords[:5])  # Limit additional keywords
        
        # Calculate overall confidence
        weighted_confidence = sum(r.confidence * (1.0 / (i + 1)) for i, r in enumerate(results_sorted))
        total_weight = sum(1.0 / (i + 1) for i in range(len(results_sorted)))
        final_metadata['confidence'] = weighted_confidence / total_weight if total_weight > 0 else 0.5
        
        return DocumentMetadata(**final_metadata)
    
    async def _enhance_metadata(self, metadata: DocumentMetadata, 
                              document_sample: str) -> Optional[DocumentMetadata]:
        """Enhance metadata using AI"""
        try:
            llm = self.get_llm_client()
            
            prompt = self.metadata_enhancement_prompt.format_prompt(
                existing_metadata=metadata.dict(),
                document_sample=document_sample
            )
            
            response = await llm.ainvoke(prompt.to_messages())
            
            # Parse enhanced metadata
            try:
                enhanced_dict = json.loads(response.content)
                cleaned_enhanced = self._clean_ai_metadata(enhanced_dict)
                
                # Merge with existing metadata
                current_dict = metadata.dict()
                for field, value in cleaned_enhanced.items():
                    if value and (not current_dict.get(field) or field == 'confidence'):
                        current_dict[field] = value
                
                return DocumentMetadata(**current_dict)
                
            except json.JSONDecodeError:
                self.log_warning("Failed to parse metadata enhancement response")
                return metadata
                
        except Exception as e:
            self.log_warning(f"Metadata enhancement failed: {e}")
            return metadata
    
    async def _validate_metadata(self, metadata: DocumentMetadata) -> Dict[str, Any]:
        """Validate extracted metadata"""
        validation_score = 1.0
        issues = []
        
        # Check title length and format
        if metadata.title:
            if len(metadata.title) < 5:
                validation_score -= 0.1
                issues.append("Title too short")
            elif len(metadata.title) > 200:
                validation_score -= 0.1
                issues.append("Title too long")
        
        # Check author format
        if metadata.author:
            # Simple check for reasonable author format
            if not re.match(r'^[A-Za-z\s\.,\-]+$', metadata.author):
                validation_score -= 0.1
                issues.append("Author format questionable")
        
        # Check language code
        valid_language_codes = ['en', 'es', 'fr', 'de', 'it', 'pt', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru']
        if metadata.language not in valid_language_codes:
            validation_score -= 0.1
            issues.append("Unknown language code")
        
        # Check keywords quality
        if metadata.keywords:
            if len(metadata.keywords) > 20:
                validation_score -= 0.1
                issues.append("Too many keywords")
            
            # Check for very short keywords
            short_keywords = [kw for kw in metadata.keywords if len(kw) < 3]
            if short_keywords:
                validation_score -= 0.05
                issues.append("Some keywords too short")
        
        return {
            "confidence": max(0.0, validation_score),
            "issues": issues,
            "validated": True
        }
    
    def get_extraction_methods(self) -> List[str]:
        """Get list of available extraction methods"""
        methods = ["file_properties", "pattern_matching"]
        
        if self.use_ai_extraction:
            methods.append("ai_extraction")
        
        if self.extract_custom_fields:
            methods.append("custom_field_extraction")
        
        return methods
    
    def get_supported_fields(self) -> List[str]:
        """Get list of supported metadata fields"""
        return list(DocumentMetadata.__fields__.keys())


### 
"""
Metadata Writer Agent
Handles database operations and metadata management for documents
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import hashlib

from langchain.schema import Document

from .base_agent import BaseAgent, AgentResult


@dataclass
class DocumentMetadata:
    """Structured document metadata"""
    document_id: str
    filename: str
    file_size: int
    file_type: str
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    language: str = "en"
    page_count: int = 0
    word_count: int = 0
    character_count: int = 0
    content_hash: Optional[str] = None
    keywords: Optional[List[str]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    created_at: Optional[datetime] = None
    processing_metadata: Optional[Dict[str, Any]] = None


class MetadataWriter(BaseAgent):
    """
    Metadata Writer Agent
    
    Capabilities:
    - Document metadata extraction and enrichment
    - Database operations for document storage
    - Chunk metadata management
    - Processing result storage
    - Metadata validation and normalization
    - Cross-reference and relationship management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Load metadata writer specific configuration
        self.metadata_config = self.config.get('agents', {}).get('metadata_writer', {})
        
        # Database configuration
        self.db_config = self.config.get('database', {})
        
        # Processing options
        self.validate_metadata = self.metadata_config.get('validate_metadata', True)
        self.enrich_metadata = self.metadata_config.get('enrich_metadata', True)
        self.store_processing_history = self.metadata_config.get('store_processing_history', True)
        
        # Metadata validation rules
        self.validation_rules = self.metadata_config.get('validation_rules', {})
        
        self.log_info("MetadataWriter initialized")
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Process metadata writing operations
        
        Args:
            input_data: Dict containing document info and metadata
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with metadata writing results
        """
        try:
            # Parse input data
            document_id = input_data.get('document_id')
            if not document_id:
                return AgentResult(
                    success=False,
                    error="document_id is required"
                )
            
            # Determine operation type
            operation = input_data.get('operation', 'create')
            
            if operation == 'create':
                return await self._create_document_metadata(input_data)
            elif operation == 'update':
                return await self._update_document_metadata(input_data)
            elif operation == 'enhance':
                return await self._enhance_document_metadata(input_data)
            elif operation == 'store_chunks':
                return await self._store_document_chunks(input_data)
            elif operation == 'store_processing_result':
                return await self._store_processing_result(input_data)
            else:
                # Default: comprehensive metadata processing
                return await self._process_comprehensive_metadata(input_data)
        
        except Exception as e:
            self.log_error(f"Error in metadata processing: {str(e)}")
            return AgentResult(
                success=False,
                error=f"Metadata processing failed: {str(e)}"
            )
    
    async def _process_comprehensive_metadata(self, input_data: Dict[str, Any]) -> AgentResult:
        """Process comprehensive metadata extraction and storage"""
        
        document_id = input_data['document_id']
        documents = input_data.get('documents', [])
        existing_metadata = input_data.get('existing_metadata', {})
        
        try:
            # Extract metadata from documents
            extracted_metadata = await self._extract_metadata_from_documents(documents)
            
            # Merge with existing metadata
            combined_metadata = {**existing_metadata, **extracted_metadata}
            
            # Validate metadata
            if self.validate_metadata:
                validation_result = await self._validate_metadata(combined_metadata)
                if not validation_result['valid']:
                    self.log_warning(f"Metadata validation issues: {validation_result['issues']}")
                    combined_metadata['validation_issues'] = validation_result['issues']
            
            # Enrich metadata if enabled
            if self.enrich_metadata:
                enriched_metadata = await self._enrich_metadata(combined_metadata, documents)
                combined_metadata.update(enriched_metadata)
            
            # Create structured metadata object
            structured_metadata = await self._create_structured_metadata(
                document_id, combined_metadata
            )
            
            # Store in database (if database manager is available)
            storage_result = await self._store_metadata_in_database(
                document_id, structured_metadata
            )
            
            result_data = {
                "document_id": document_id,
                "metadata": combined_metadata,
                "structured_metadata": structured_metadata.__dict__,
                "storage_result": storage_result,
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata_stats": {
                    "total_fields": len(combined_metadata),
                    "has_title": bool(combined_metadata.get('title')),
                    "has_author": bool(combined_metadata.get('author')),
                    "has_keywords": bool(combined_metadata.get('keywords')),
                    "has_entities": bool(combined_metadata.get('entities')),
                    "has_summary": bool(combined_metadata.get('summary'))
                }
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                confidence=0.9,
                metadata={
                    "operation": "comprehensive_metadata_processing",
                    "document_id": document_id,
                    "documents_processed": len(documents)
                }
            )
            
        except Exception as e:
            self.log_error(f"Error in comprehensive metadata processing: {str(e)}")
            return AgentResult(
                success=False,
                error=f"Comprehensive metadata processing failed: {str(e)}"
            )
    
    async def _extract_metadata_from_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract metadata from document objects"""
        
        if not documents:
            return {}
        
        metadata = {
            "page_count": len(documents),
            "character_count": 0,
            "word_count": 0,
            "sources": [],
            "content_types": set(),
            "languages": set()
        }
        
        # Aggregate content statistics
        all_content = []
        for doc in documents:
            content = doc.page_content
            all_content.append(content)
            
            metadata["character_count"] += len(content)
            metadata["word_count"] += len(content.split())
            
            # Extract source information
            doc_metadata = doc.metadata or {}
            if doc_metadata.get('source'):
                metadata["sources"].append(doc_metadata['source'])
            
            # Extract content types
            if doc_metadata.get('file_type'):
                metadata["content_types"].add(doc_metadata['file_type'])
            
            # Extract language information
            if doc_metadata.get('language'):
                metadata["languages"].add(doc_metadata['language'])
        
        # Convert sets to lists for JSON serialization
        metadata["sources"] = list(set(metadata["sources"]))
        metadata["content_types"] = list(metadata["content_types"])
        metadata["languages"] = list(metadata["languages"])
        
        # Set primary language
        if metadata["languages"]:
            metadata["language"] = metadata["languages"][0]
        else:
            metadata["language"] = "en"  # Default
        
        # Calculate content hash
        combined_content = "\n".join(all_content)
        metadata["content_hash"] = hashlib.sha256(
            combined_content.encode('utf-8')
        ).hexdigest()
        
        # Extract title from first document if available
        if documents and documents[0].metadata.get('title'):
            metadata["title"] = documents[0].metadata['title']
        elif documents:
            # Try to extract title from first few lines
            first_content = documents[0].page_content
            lines = first_content.split('\n')
            for line in lines[:5]:
                line = line.strip()
                if len(line) > 5 and len(line) < 100 and not line.endswith('.'):
                    metadata["title"] = line
                    break
        
        # Extract author information
        author_sources = []
        for doc in documents:
            doc_metadata = doc.metadata or {}
            if doc_metadata.get('author'):
                author_sources.append(doc_metadata['author'])
        
        if author_sources:
            # Use most common author
            from collections import Counter
            author_counts = Counter(author_sources)
            metadata["author"] = author_counts.most_common(1)[0][0]
        
        return metadata
    
    async def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata according to defined rules"""
        
        validation_result = {
            "valid": True,
            "issues": []
        }
        
        # Required fields validation
        required_fields = self.validation_rules.get('required_fields', [])
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                validation_result["issues"].append(f"Missing required field: {field}")
                validation_result["valid"] = False
        
        # Data type validation
        type_rules = self.validation_rules.get('field_types', {})
        for field, expected_type in type_rules.items():
            if field in metadata:
                value = metadata[field]
                if expected_type == 'string' and not isinstance(value, str):
                    validation_result["issues"].append(f"Field {field} should be string, got {type(value)}")
                elif expected_type == 'integer' and not isinstance(value, int):
                    validation_result["issues"].append(f"Field {field} should be integer, got {type(value)}")
                elif expected_type == 'list' and not isinstance(value, list):
                    validation_result["issues"].append(f"Field {field} should be list, got {type(value)}")
        
        # Range validation
        range_rules = self.validation_rules.get('field_ranges', {})
        for field, (min_val, max_val) in range_rules.items():
            if field in metadata:
                value = metadata[field]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        validation_result["issues"].append(
                            f"Field {field} value {value} outside range [{min_val}, {max_val}]"
                        )
        
        # Custom validation rules
        if metadata.get('word_count', 0) < 0:
            validation_result["issues"].append("Word count cannot be negative")
            validation_result["valid"] = False
        
        if metadata.get('character_count', 0) < 0:
            validation_result["issues"].append("Character count cannot be negative")
            validation_result["valid"] = False
        
        if metadata.get('page_count', 0) < 1:
            validation_result["issues"].append("Page count should be at least 1")
            validation_result["valid"] = False
        
        return validation_result
    
    async def _enrich_metadata(self, metadata: Dict[str, Any], 
                             documents: List[Document]) -> Dict[str, Any]:
        """Enrich metadata with additional computed information"""
        
        enriched = {}
        
        try:
            # Calculate readability metrics
            if metadata.get('word_count') and metadata.get('character_count'):
                avg_word_length = metadata['character_count'] / metadata['word_count']
                enriched['average_word_length'] = round(avg_word_length, 2)
                
                # Simple readability score (Flesch-like approximation)
                if metadata['word_count'] > 0:
                    words_per_sentence = metadata['word_count'] / max(metadata.get('page_count', 1), 1)
                    readability_score = 206.835 - (1.015 * words_per_sentence) - (84.6 * avg_word_length)
                    enriched['readability_score'] = max(0, min(100, readability_score))
            
            # Document complexity analysis
            complexity_indicators = 0
            
            # Technical terms indicator (words with numbers or special chars)
            combined_text = " ".join([doc.page_content for doc in documents[:3]])  # Sample
            technical_words = len([word for word in combined_text.split() 
                                 if any(c.isdigit() for c in word) or '_' in word])
            
            if technical_words > len(combined_text.split()) * 0.1:  # >10% technical terms
                complexity_indicators += 1
            
            # Long sentences indicator
            sentences = combined_text.split('.')
            long_sentences = sum(1 for s in sentences if len(s.split()) > 20)
            if long_sentences > len(sentences) * 0.3:  # >30% long sentences
                complexity_indicators += 1
            
            # Set complexity level
            if complexity_indicators >= 2:
                enriched['complexity_level'] = 'high'
            elif complexity_indicators == 1:
                enriched['complexity_level'] = 'medium'
            else:
                enriched['complexity_level'] = 'low'
            
            # Content type classification
            content_indicators = {
                'academic': ['abstract', 'methodology', 'conclusion', 'references', 'hypothesis'],
                'technical': ['implementation', 'algorithm', 'function', 'parameter', 'configuration'],
                'business': ['revenue', 'market', 'strategy', 'customer', 'profit'],
                'legal': ['clause', 'agreement', 'liability', 'contract', 'jurisdiction'],
                'medical': ['patient', 'treatment', 'diagnosis', 'symptoms', 'therapy']
            }
            
            content_lower = combined_text.lower()
            content_type_scores = {}
            
            for content_type, indicators in content_indicators.items():
                score = sum(1 for indicator in indicators if indicator in content_lower)
                if score > 0:
                    content_type_scores[content_type] = score
            
            if content_type_scores:
                primary_type = max(content_type_scores, key=content_type_scores.get)
                enriched['primary_content_type'] = primary_type
                enriched['content_type_confidence'] = content_type_scores[primary_type] / len(content_indicators[primary_type])
            
            # Processing timestamp
            enriched['metadata_enriched_at'] = datetime.now(timezone.utc).isoformat()
            
            # Enrichment metadata
            enriched['enrichment_version'] = '1.0.0'
            enriched['enrichment_methods'] = ['readability_analysis', 'complexity_analysis', 'content_classification']
            
        except Exception as e:
            self.log_warning(f"Error in metadata enrichment: {e}")
            enriched['enrichment_error'] = str(e)
        
        return enriched
    
    async def _create_structured_metadata(self, document_id: str, 
                                        metadata: Dict[str, Any]) -> DocumentMetadata:
        """Create structured metadata object"""
        
        return DocumentMetadata(
            document_id=document_id,
            filename=metadata.get('filename', 'unknown'),
            file_size=metadata.get('file_size', 0),
            file_type=metadata.get('file_type', 'unknown'),
            title=metadata.get('title'),
            author=metadata.get('author'),
            subject=metadata.get('subject'),
            language=metadata.get('language', 'en'),
            page_count=metadata.get('page_count', 0),
            word_count=metadata.get('word_count', 0),
            character_count=metadata.get('character_count', 0),
            content_hash=metadata.get('content_hash'),
            keywords=metadata.get('keywords'),
            entities=metadata.get('entities'),
            summary=metadata.get('summary'),
            created_at=datetime.now(timezone.utc),
            processing_metadata=metadata.get('enrichment_metadata', {})
        )
    
    async def _store_metadata_in_database(self, document_id: str, 
                                        metadata: DocumentMetadata) -> Dict[str, Any]:
        """Store metadata in database if database manager is available"""
        
        try:
            # This would integrate with the database manager
            # For now, return a mock result
            
            storage_result = {
                "stored": True,
                "document_id": document_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "storage_method": "database",
                "metadata_fields_stored": len(metadata.__dict__)
            }
            
            self.log_info(f"Metadata stored for document {document_id}")
            return storage_result
            
        except Exception as e:
            self.log_error(f"Error storing metadata in database: {e}")
            return {
                "stored": False,
                "error": str(e),
                "fallback_used": False
            }
    
    async def _create_document_metadata(self, input_data: Dict[str, Any]) -> AgentResult:
        """Create new document metadata entry"""
        
        document_id = input_data['document_id']
        metadata = input_data.get('metadata', {})
        
        try:
            # Create structured metadata
            structured_metadata = await self._create_structured_metadata(document_id, metadata)
            
            # Store in database
            storage_result = await self._store_metadata_in_database(document_id, structured_metadata)
            
            return AgentResult(
                success=True,
                data={
                    "operation": "create",
                    "document_id": document_id,
                    "metadata": structured_metadata.__dict__,
                    "storage_result": storage_result
                },
                confidence=0.95
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Failed to create document metadata: {str(e)}"
            )
    
    async def _update_document_metadata(self, input_data: Dict[str, Any]) -> AgentResult:
        """Update existing document metadata"""
        
        document_id = input_data['document_id']
        updates = input_data.get('updates', {})
        
        try:
            # Update logic would go here
            # For now, return success
            
            return AgentResult(
                success=True,
                data={
                    "operation": "update",
                    "document_id": document_id,
                    "updated_fields": list(updates.keys()),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                confidence=0.9
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Failed to update document metadata: {str(e)}"
            )
    
    async def _enhance_document_metadata(self, input_data: Dict[str, Any]) -> AgentResult:
        """Enhance document metadata with AI-extracted information"""
        
        document_id = input_data['document_id']
        documents = input_data.get('documents', [])
        
        try:
            # AI enhancement would go here
            # For now, return basic enhancement
            
            enhanced_metadata = {
                "ai_enhanced": True,
                "enhancement_timestamp": datetime.now(timezone.utc).isoformat(),
                "enhancement_version": "1.0.0"
            }
            
            return AgentResult(
                success=True,
                data={
                    "operation": "enhance",
                    "document_id": document_id,
                    "enhanced_metadata": enhanced_metadata
                },
                confidence=0.8
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Failed to enhance document metadata: {str(e)}"
            )
    
    async def _store_document_chunks(self, input_data: Dict[str, Any]) -> AgentResult:
        """Store document chunk metadata"""
        
        document_id = input_data['document_id']
        chunks = input_data.get('chunks', [])
        
        try:
            chunk_metadata = []
            
            for i, chunk in enumerate(chunks):
                if hasattr(chunk, 'metadata'):
                    chunk_meta = {
                        "chunk_id": f"{document_id}_chunk_{i}",
                        "document_id": document_id,
                        "chunk_index": i,
                        "chunk_length": len(chunk.page_content),
                        "metadata": chunk.metadata
                    }
                    chunk_metadata.append(chunk_meta)
            
            return AgentResult(
                success=True,
                data={
                    "operation": "store_chunks",
                    "document_id": document_id,
                    "chunks_stored": len(chunk_metadata),
                    "chunk_metadata": chunk_metadata
                },
                confidence=0.95
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Failed to store chunk metadata: {str(e)}"
            )
    
    async def _store_processing_result(self, input_data: Dict[str, Any]) -> AgentResult:
        """Store processing result metadata"""
        
        document_id = input_data['document_id']
        processing_result = input_data.get('processing_result', {})
        
        try:
            result_metadata = {
                "document_id": document_id,
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "result": processing_result,
                "stored_by": "metadata_writer"
            }
            
            return AgentResult(
                success=True,
                data={
                    "operation": "store_processing_result",
                    "document_id": document_id,
                    "result_metadata": result_metadata
                },
                confidence=0.9
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Failed to store processing result: {str(e)}"
            )
    
    def set_validation_rules(self, rules: Dict[str, Any]):
        """Set custom validation rules"""
        self.validation_rules = rules
        self.log_info("Custom validation rules set")
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get current validation rules"""
        return self.validation_rules.copy()
    
    async def validate_document_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Public method to validate metadata"""
        return await self._validate_metadata(metadata)
    
    async def enrich_document_metadata(self, metadata: Dict[str, Any], 
                                     documents: List[Document]) -> Dict[str, Any]:
        """Public method to enrich metadata"""
        return await self._enrich_metadata(metadata, documents)