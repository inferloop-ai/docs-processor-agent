"""
Question Answering Agent
Advanced RAG-based question answering with multiple strategies
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import tiktoken

from langchain.schema import Document, BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentResult


@dataclass
class ContextDocument:
    """Context document for QA"""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str = ""
    page: Optional[int] = None
    chunk_id: Optional[str] = None


class QARequest(BaseModel):
    """Structured QA request"""
    query: str = Field(description="The question to answer")
    context_documents: List[Dict[str, Any]] = Field(default=[], description="Context documents")
    max_context_length: int = Field(default=8000, description="Maximum context length")
    response_format: str = Field(default="detailed", description="Response format")
    include_sources: bool = Field(default=True, description="Include source references")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")


class QAResponse(BaseModel):
    """Structured QA response"""
    answer: str = Field(description="The generated answer")
    confidence: float = Field(description="Confidence score")
    sources: List[Dict[str, Any]] = Field(description="Source documents used")
    context_used: int = Field(description="Amount of context used")
    reasoning: Optional[str] = Field(description="Reasoning behind the answer")


class QAAgent(BaseAgent):
    """
    Question Answering Agent
    
    Capabilities:
    - RAG-based question answering
    - Multi-document context synthesis
    - Source attribution and verification
    - Confidence scoring
    - Multiple response formats
    - Context optimization
    - Query classification and routing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Load QA-specific configuration
        self.qa_config = self.config.get('agents', {}).get('qa_agent', {})
        
        # QA settings
        self.max_context_length = self.qa_config.get('max_context_length', 8000)
        self.max_sources = self.qa_config.get('max_sources', 5)
        self.min_confidence_threshold = self.qa_config.get('min_confidence_threshold', 0.1)
        self.include_reasoning = self.qa_config.get('include_reasoning', False)
        
        # Response formats
        self.supported_formats = ['brief', 'detailed', 'structured', 'academic']
        
        # Initialize text splitter for context management
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        
        # Setup QA prompts
        self._setup_qa_prompts()
        
        self.log_info("QAAgent initialized")
    
    def _setup_qa_prompts(self):
        """Setup various QA prompts for different scenarios"""
        
        # Main QA prompt
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI assistant that answers questions based on provided context documents. 

Your responsibilities:
1. Answer questions accurately using ONLY the information provided in the context
2. If the context doesn't contain enough information, clearly state what's missing
3. Cite specific sources when making claims
4. Provide clear, well-structured answers appropriate to the question type
5. Indicate your confidence level in the answer

Guidelines:
- Be precise and factual
- Don't make assumptions beyond the provided context
- If multiple sources contradict, acknowledge this
- For numerical data, be exact and cite sources
- Structure your response clearly with proper formatting

Context Documents:
{context}"""),
            
            ("human", """Question: {question}

Please provide a comprehensive answer based on the context documents above. Include relevant source citations and indicate your confidence in the answer.""")
        ])
        
        # Brief response prompt
        self.brief_prompt = ChatPromptTemplate.from_messages([
            ("system", """Provide brief, concise answers based only on the provided context. Keep responses under 100 words while maintaining accuracy.

Context: {context}"""),
            ("human", "{question}")
        ])
        
        # Structured response prompt
        self.structured_prompt = ChatPromptTemplate.from_messages([
            ("system", """Provide structured answers in the following format:

**Answer:** [Direct answer to the question]
**Key Points:** 
- Point 1
- Point 2
- Point 3

**Sources:** [List relevant sources]
**Confidence:** [Low/Medium/High]

Context: {context}"""),
            ("human", "{question}")
        ])
        
        # Confidence scoring prompt
        self.confidence_prompt = ChatPromptTemplate.from_messages([
            ("system", """Rate the confidence of this answer on a scale of 0.0 to 1.0 based on:
- How well the context supports the answer
- Completeness of information
- Clarity of the sources
- Potential ambiguity

Return only a number between 0.0 and 1.0."""),
            ("human", """Question: {question}
Answer: {answer}
Context: {context}

Confidence score:""")
        ])
    
    async def process(self, input_data: Union[Dict[str, Any], QARequest], **kwargs) -> AgentResult:
        """
        Process QA request
        
        Args:
            input_data: QA request data
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with answer and metadata
        """
        try:
            # Parse input
            if isinstance(input_data, dict):
                if 'query' not in input_data and 'question' not in input_data:
                    return AgentResult(
                        success=False,
                        error="Input must contain 'query' or 'question' field"
                    )
                
                # Normalize field names
                if 'question' in input_data:
                    input_data['query'] = input_data.pop('question')
                
                qa_request = QARequest(**input_data)
            elif isinstance(input_data, str):
                qa_request = QARequest(query=input_data)
            else:
                qa_request = input_data
            
            # Validate query
            if not qa_request.query or not qa_request.query.strip():
                return AgentResult(
                    success=False,
                    error="Query cannot be empty"
                )
            
            # Classify query type
            query_classification = await self._classify_query(qa_request.query)
            
            # Process context documents
            context_docs = await self._process_context_documents(
                qa_request.context_documents,
                qa_request.max_context_length
            )
            
            if not context_docs:
                return AgentResult(
                    success=False,
                    error="No relevant context documents available to answer the question",
                    confidence=0.0
                )
            
            # Generate answer
            answer_result = await self._generate_answer(
                qa_request,
                context_docs,
                query_classification
            )
            
            if not answer_result['success']:
                return AgentResult(
                    success=False,
                    error=answer_result['error']
                )
            
            # Calculate confidence score
            confidence = await self._calculate_confidence(
                qa_request.query,
                answer_result['answer'],
                context_docs
            )
            
            # Prepare sources
            sources = self._prepare_sources(context_docs, qa_request.include_sources)
            
            # Format final response
            response_data = {
                "answer": answer_result['answer'],
                "confidence": confidence,
                "sources": sources,
                "query_classification": query_classification,
                "context_used": len(context_docs),
                "response_format": qa_request.response_format,
                "metadata": {
                    "model_used": self.config.get('llm', {}).get('primary_provider', 'unknown'),
                    "context_length": sum(len(doc.content) for doc in context_docs),
                    "max_context_length": qa_request.max_context_length,
                    "temperature": qa_request.temperature
                }
            }
            
            if self.include_reasoning and answer_result.get('reasoning'):
                response_data['reasoning'] = answer_result['reasoning']
            
            return AgentResult(
                success=True,
                data=response_data,
                confidence=confidence,
                metadata={
                    "query": qa_request.query,
                    "sources_count": len(sources),
                    "context_documents_used": len(context_docs)
                }
            )
            
        except Exception as e:
            self.log_error(f"Error in QA processing: {str(e)}")
            return AgentResult(
                success=False,
                error=f"QA processing failed: {str(e)}"
            )
    
    async def _classify_query(self, query: str) -> Dict[str, Any]:
        """Classify the type of query"""
        query_lower = query.lower()
        
        classification = {
            "type": "general",
            "intent": "unknown",
            "complexity": "medium",
            "expected_length": "medium"
        }
        
        # Question type detection
        if any(word in query_lower for word in ['what', 'which', 'who', 'where', 'when']):
            classification["type"] = "factual"
            classification["expected_length"] = "short"
        elif any(word in query_lower for word in ['how', 'why']):
            classification["type"] = "explanatory"
            classification["expected_length"] = "long"
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            classification["type"] = "summary"
            classification["expected_length"] = "long"
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            classification["type"] = "comparative"
            classification["expected_length"] = "medium"
        elif '?' not in query and len(query.split()) > 5:
            classification["type"] = "search"
            classification["expected_length"] = "medium"
        
        # Intent detection
        if any(word in query_lower for word in ['define', 'definition', 'meaning']):
            classification["intent"] = "definition"
        elif any(word in query_lower for word in ['list', 'enumerate']):
            classification["intent"] = "enumeration"
        elif any(word in query_lower for word in ['example', 'instance']):
            classification["intent"] = "examples"
        
        # Complexity assessment
        if len(query.split()) > 15 or 'and' in query_lower:
            classification["complexity"] = "high"
        elif len(query.split()) < 5:
            classification["complexity"] = "low"
        
        return classification
    
    async def _process_context_documents(self, context_docs_data: List[Dict[str, Any]], 
                                       max_length: int) -> List[ContextDocument]:
        """Process and optimize context documents"""
        if not context_docs_data:
            return []
        
        # Convert to ContextDocument objects
        context_docs = []
        for doc_data in context_docs_data:
            context_doc = ContextDocument(
                content=doc_data.get('content', ''),
                metadata=doc_data.get('metadata', {}),
                relevance_score=doc_data.get('score', doc_data.get('relevance_score', 0.5)),
                source=doc_data.get('source', doc_data.get('metadata', {}).get('source', 'unknown')),
                page=doc_data.get('page', doc_data.get('metadata', {}).get('page')),
                chunk_id=doc_data.get('chunk_id', doc_data.get('id'))
            )
            
            if context_doc.content.strip():  # Only add non-empty documents
                context_docs.append(context_doc)
        
        # Sort by relevance score
        context_docs.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Optimize context length
        optimized_docs = []
        current_length = 0
        
        for doc in context_docs:
            doc_length = len(doc.content)
            
            if current_length + doc_length <= max_length:
                optimized_docs.append(doc)
                current_length += doc_length
            elif not optimized_docs:  # Ensure at least one document
                # Truncate the first document if it's too long
                truncated_content = doc.content[:max_length]
                doc.content = truncated_content
                optimized_docs.append(doc)
                break
            else:
                break
        
        # Limit number of sources
        if len(optimized_docs) > self.max_sources:
            optimized_docs = optimized_docs[:self.max_sources]
        
        return optimized_docs
    
    async def _generate_answer(self, qa_request: QARequest, 
                             context_docs: List[ContextDocument],
                             query_classification: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer using appropriate prompt strategy"""
        
        try:
            # Prepare context
            context_text = self._format_context(context_docs)
            
            # Select prompt based on response format
            if qa_request.response_format == "brief":
                prompt = self.brief_prompt
            elif qa_request.response_format == "structured":
                prompt = self.structured_prompt
            else:
                prompt = self.qa_prompt
            
            # Get LLM client
            llm = self.get_llm_client()
            
            # Generate response
            formatted_prompt = prompt.format_prompt(
                context=context_text,
                question=qa_request.query
            )
            
            response = await llm.ainvoke(
                formatted_prompt.to_messages(),
                temperature=qa_request.temperature
            )
            
            answer = response.content.strip()
            
            if not answer:
                return {
                    "success": False,
                    "error": "LLM returned empty response"
                }
            
            # Post-process answer
            answer = self._post_process_answer(answer, qa_request.response_format)
            
            return {
                "success": True,
                "answer": answer,
                "reasoning": None  # Could be enhanced with chain-of-thought
            }
            
        except Exception as e:
            self.log_error(f"Error generating answer: {str(e)}")
            return {
                "success": False,
                "error": f"Answer generation failed: {str(e)}"
            }
    
    def _format_context(self, context_docs: List[ContextDocument]) -> str:
        """Format context documents for the prompt"""
        if not context_docs:
            return "No context documents available."
        
        context_parts = []
        
        for i, doc in enumerate(context_docs, 1):
            # Format source information
            source_info = f"Source {i}: {doc.source}"
            if doc.page:
                source_info += f" (Page {doc.page})"
            
            # Add relevance score if available
            if doc.relevance_score and doc.relevance_score > 0:
                source_info += f" [Relevance: {doc.relevance_score:.2f}]"
            
            # Format the document
            context_parts.append(f"{source_info}\n{doc.content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _post_process_answer(self, answer: str, response_format: str) -> str:
        """Post-process the generated answer"""
        
        # Remove any unwanted prefixes
        prefixes_to_remove = [
            "Based on the provided context",
            "According to the documents",
            "The context indicates",
            "From the information provided"
        ]
        
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].lstrip(', :')
        
        # Format based on response type
        if response_format == "structured":
            # Ensure proper markdown formatting
            answer = self._ensure_structured_format(answer)
        elif response_format == "brief":
            # Keep it concise
            if len(answer) > 200:
                sentences = answer.split('.')
                answer = '. '.join(sentences[:2]) + '.'
        
        return answer.strip()
    
    def _ensure_structured_format(self, answer: str) -> str:
        """Ensure structured format has proper sections"""
        
        # Check if already properly formatted
        if "**Answer:**" in answer and "**Sources:**" in answer:
            return answer
        
        # If not structured, add basic structure
        lines = answer.split('\n')
        structured_parts = [
            "**Answer:**",
            answer,
            "",
            "**Confidence:** Medium"
        ]
        
        return '\n'.join(structured_parts)
    
    async def _calculate_confidence(self, query: str, answer: str, 
                                  context_docs: List[ContextDocument]) -> float:
        """Calculate confidence score for the answer"""
        
        try:
            # Use LLM-based confidence scoring
            llm = self.get_llm_client()
            
            context_text = self._format_context(context_docs[:3])  # Use top 3 for confidence
            
            prompt = self.confidence_prompt.format_prompt(
                question=query,
                answer=answer,
                context=context_text
            )
            
            response = await llm.ainvoke(prompt.to_messages(), temperature=0.0)
            
            # Extract confidence score
            confidence_text = response.content.strip()
            
            # Try to extract number
            import re
            matches = re.findall(r'0\.\d+|1\.0|0|1', confidence_text)
            
            if matches:
                confidence = float(matches[0])
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            else:
                # Fallback to heuristic calculation
                confidence = self._calculate_heuristic_confidence(query, answer, context_docs)
            
            return confidence
            
        except Exception as e:
            self.log_warning(f"LLM confidence scoring failed, using heuristic: {e}")
            return self._calculate_heuristic_confidence(query, answer, context_docs)
    
    def _calculate_heuristic_confidence(self, query: str, answer: str, 
                                      context_docs: List[ContextDocument]) -> float:
        """Calculate confidence using heuristic methods"""
        
        confidence_factors = []
        
        # Factor 1: Context relevance
        if context_docs:
            avg_relevance = sum(doc.relevance_score for doc in context_docs) / len(context_docs)
            confidence_factors.append(avg_relevance)
        else:
            confidence_factors.append(0.1)
        
        # Factor 2: Answer length appropriateness
        answer_length = len(answer.split())
        if 10 <= answer_length <= 200:
            length_score = 0.8
        elif 5 <= answer_length < 10 or 200 < answer_length <= 500:
            length_score = 0.6
        else:
            length_score = 0.4
        confidence_factors.append(length_score)
        
        # Factor 3: Uncertainty indicators
        uncertainty_phrases = [
            "i don't know", "unclear", "not specified", "not mentioned",
            "uncertain", "possibly", "might be", "could be", "seems like"
        ]
        
        uncertainty_count = sum(1 for phrase in uncertainty_phrases 
                              if phrase in answer.lower())
        uncertainty_score = max(0.2, 1.0 - (uncertainty_count * 0.2))
        confidence_factors.append(uncertainty_score)
        
        # Factor 4: Question-answer alignment
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words.intersection(answer_words))
        alignment_score = min(1.0, overlap / max(len(query_words), 1) * 2)
        confidence_factors.append(alignment_score)
        
        # Calculate weighted average
        weights = [0.4, 0.2, 0.2, 0.2]  # Prioritize context relevance
        confidence = sum(f * w for f, w in zip(confidence_factors, weights))
        
        return max(self.min_confidence_threshold, min(0.95, confidence))
    
    def _prepare_sources(self, context_docs: List[ContextDocument], 
                        include_sources: bool) -> List[Dict[str, Any]]:
        """Prepare source information for the response"""
        
        if not include_sources or not context_docs:
            return []
        
        sources = []
        for i, doc in enumerate(context_docs):
            source_info = {
                "id": i + 1,
                "source": doc.source,
                "relevance_score": doc.relevance_score,
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            }
            
            if doc.page:
                source_info["page"] = doc.page
            
            if doc.chunk_id:
                source_info["chunk_id"] = doc.chunk_id
            
            # Add metadata
            if doc.metadata:
                source_info["metadata"] = {
                    k: v for k, v in doc.metadata.items() 
                    if k not in ['embedding', 'vector_data']  # Exclude large data
                }
            
            sources.append(source_info)
        
        return sources
    
    def get_supported_formats(self) -> List[str]:
        """Get supported response formats"""
        return self.supported_formats.copy()
    
    def set_max_context_length(self, length: int):
        """Set maximum context length"""
        self.max_context_length = max(1000, min(length, 32000))  # Reasonable bounds
        self.log_info(f"Max context length set to {self.max_context_length}")
    
    def set_max_sources(self, count: int):
        """Set maximum number of sources to use"""
        self.max_sources = max(1, min(count, 20))  # Reasonable bounds
        self.log_info(f"Max sources set to {self.max_sources}")
    
    async def answer_question(self, question: str, context_documents: List[Dict[str, Any]] = None,
                            response_format: str = "detailed") -> Dict[str, Any]:
        """Convenience method for simple question answering"""
        
        qa_request = QARequest(
            query=question,
            context_documents=context_documents or [],
            response_format=response_format
        )
        
        result = await self.process(qa_request)
        
        if result.success:
            return result.data
        else:
            return {
                "answer": f"I apologize, but I couldn't answer your question: {result.error}",
                "confidence": 0.0,
                "sources": [],
                "error": result.error
            }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        try:
            # Try to use tiktoken for accurate counting
            encoding = tiktoken.encoding_for_model("gpt-4")
            return len(encoding.encode(text))
        except:
            # Fallback to rough estimation
            return len(text.split()) * 1.3  # Rough tokens per word