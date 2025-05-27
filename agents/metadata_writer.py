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