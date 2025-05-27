"""
RAG Enricher Agent
Enriches documents with external knowledge using web search and retrieval
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import requests
from urllib.parse import quote, urljoin

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentResult


@dataclass
class WebSearchResult:
    """Result from web search"""
    title: str
    url: str
    snippet: str
    content: str
    relevance_score: float
    source_domain: str
    publish_date: Optional[str] = None


@dataclass
class EnrichmentSuggestion:
    """Suggestion for document enrichment"""
    type: str  # 'definition', 'context', 'example', 'reference'
    content: str
    source: str
    relevance: float
    position_suggestion: str  # 'beginning', 'end', 'inline'


class DocumentEnrichmentRequest(BaseModel):
    """Request for document enrichment"""
    query_terms: List[str] = Field(description="Terms to search for")
    context_type: str = Field(description="Type of context needed")
    max_results: int = Field(default=5, description="Maximum search results")
    trusted_domains: List[str] = Field(default=[], description="Preferred domains")


class RAGEnricher(BaseAgent):
    """
    RAG Enricher Agent
    
    Capabilities:
    - Web search integration for external knowledge
    - Content relevance scoring and filtering
    - Smart context extraction and summarization
    - Multi-source knowledge synthesis
    - Fact verification and cross-referencing
    - Domain-specific enrichment strategies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Load RAG enricher configuration
        self.rag_config = self.config.get('agents', {}).get('rag_enricher', {})
        self.web_search_config = self.config.get('web_search', {})
        
        # Web search settings
        self.web_search_enabled = self.rag_config.get('web_search_enabled', True)
        self.max_web_results = self.rag_config.get('max_web_results', 5)
        self.search_timeout = self.rag_config.get('search_timeout', 10)
        self.trusted_domains = self.rag_config.get('trusted_domains', [
            'wikipedia.org', 'britannica.com', 'nist.gov', 'ieee.org'
        ])
        
        # Search provider configuration
        self.search_provider = self.web_search_config.get('provider', 'tavily')
        self.search_api_key = self.web_search_config.get('api_key', '')
        
        # Content processing
        self.min_relevance_score = 0.6
        self.max_content_length = 2000
        self.enrichment_strategies = ['definition', 'context', 'examples', 'references']
        
        # Initialize search clients
        self._initialize_search_clients()
        
        # Setup enrichment prompts
        self._setup_enrichment_prompts()
        
        self.log_info(f"RAGEnricher initialized with provider: {self.search_provider}")
    
    def _initialize_search_clients(self):
        """Initialize web search clients"""
        self.search_clients = {}
        
        # Tavily Search
        if self.search_provider == 'tavily' and self.search_api_key:
            try:
                self.search_clients['tavily'] = {
                    'api_key': self.search_api_key,
                    'base_url': 'https://api.tavily.com/search'
                }
                self.log_info("Tavily search client initialized")
            except Exception as e:
                self.log_warning(f"Failed to initialize Tavily: {e}")
        
        # Serper API
        if self.search_provider == 'serper':
            try:
                self.search_clients['serper'] = {
                    'api_key': self.search_api_key,
                    'base_url': 'https://google.serper.dev/search'
                }
                self.log_info("Serper search client initialized")
            except Exception as e:
                self.log_warning(f"Failed to initialize Serper: {e}")
        
        # DuckDuckGo (no API key required)
        self.search_clients['duckduckgo'] = {
            'base_url': 'https://api.duckduckgo.com'
        }
    
    def _setup_enrichment_prompts(self):
        """Setup prompts for content enrichment"""
        
        self.relevance_scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at evaluating the relevance of web content to document topics.

Rate how relevant the provided web content is to the document context on a scale of 0.0 to 1.0:
- 1.0: Highly relevant, directly addresses the topic
- 0.7-0.9: Relevant, provides useful context or details
- 0.4-0.6: Somewhat relevant, tangentially related
- 0.0-0.3: Not relevant or off-topic"""),
            
            ("human", """Document Context:
{document_context}

Web Content:
Title: {web_title}
Content: {web_content}

Rate the relevance (0.0-1.0) and provide a brief explanation.

Format as JSON:
{{"relevance_score": 0.0-1.0, "explanation": "brief explanation"}}""")
        ])
        
        self.enrichment_suggestion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at suggesting how to enrich documents with external knowledge.

Based on the document content and relevant web sources, suggest specific enrichments:
1. Definitions for technical terms
2. Additional context or background
3. Examples or case studies
4. References to authoritative sources

Be specific about where and how to integrate the information."""),
            
            ("human", """Document Content:
{document_content}

Relevant Web Sources:
{web_sources}

Suggest 3-5 specific enrichments with:
- Type (definition, context, example, reference)
- Content to add
- Where to place it
- Why it's valuable

Format as JSON array:
[
    {{
        "type": "definition|context|example|reference",
        "content": "specific content to add",
        "position": "beginning|end|inline|after_section_X",
        "rationale": "why this enrichment is valuable"
    }}
]""")
        ])
        
        self.fact_verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert fact-checker. Compare information from multiple sources to identify:
1. Consistent facts that are supported by multiple sources
2. Contradictions between sources
3. Claims that need additional verification
4. Information that appears to be outdated"""),
            
            ("human", """Original Document Claims:
{document_claims}

External Sources:
{external_sources}

Identify:
- Verified facts (supported by multiple sources)
- Contradictions or inconsistencies
- Claims requiring verification
- Potentially outdated information

Format as JSON:
{{
    "verified_facts": ["fact1", "fact2"],
    "contradictions": [
        {{"document_claim": "claim", "source_claim": "different claim", "source": "source"}}
    ],
    "needs_verification": ["claim1", "claim2"],
    "potentially_outdated": ["info1", "info2"]
}}""")
        ])
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> AgentResult:
        """
        Enrich documents with external knowledge
        
        Args:
            input_data: Dict containing:
                - documents: List of Document objects
                - metadata: Document metadata
            **kwargs: Additional parameters
        
        Returns:
            AgentResult with enriched documents and suggestions
        """
        try:
            # Parse input
            documents = input_data.get('documents', [])
            metadata = input_data.get('metadata', {})
            
            if not documents:
                return AgentResult(
                    success=False,
                    error="No documents provided for enrichment"
                )
            
            if not self.web_search_enabled:
                self.log_info("Web search disabled, skipping RAG enrichment")
                return AgentResult(
                    success=True,
                    data={
                        "enriched_documents": documents,
                        "enrichment_suggestions": [],
                        "web_sources": [],
                        "enrichment_applied": False
                    },
                    confidence=1.0
                )
            
            # Extract key terms and topics for search
            search_terms = await self._extract_search_terms(documents, metadata)
            
            if not search_terms:
                self.log_warning("No search terms extracted, skipping enrichment")
                return AgentResult(
                    success=True,
                    data={
                        "enriched_documents": documents,
                        "enrichment_suggestions": [],
                        "web_sources": [],
                        "enrichment_applied": False
                    },
                    confidence=0.8
                )
            
            # Perform web searches
            web_results = await self._perform_web_searches(search_terms)
            
            # Filter and score results for relevance
            relevant_results = await self._filter_and_score_results(
                web_results, documents, metadata
            )
            
            # Generate enrichment suggestions
            enrichment_suggestions = await self._generate_enrichment_suggestions(
                documents, relevant_results
            )
            
            # Apply enrichments (optional)
            enriched_documents = documents  # For now, keep original documents
            apply_enrichments = kwargs.get('apply_enrichments', False)
            
            if apply_enrichments and enrichment_suggestions:
                enriched_documents = await self._apply_enrichments(
                    documents, enrichment_suggestions
                )
            
            # Perform fact verification if enabled
            verification_results = {}
            if kwargs.get('verify_facts', False):
                verification_results = await self._verify_facts(
                    documents, relevant_results
                )
            
            result_data = {
                "enriched_documents": enriched_documents,
                "enrichment_suggestions": enrichment_suggestions,
                "web_sources": relevant_results,
                "search_terms_used": search_terms,
                "enrichment_applied": apply_enrichments,
                "verification_results": verification_results,
                "enrichment_statistics": {
                    "search_terms_count": len(search_terms),
                    "web_results_found": len(web_results),
                    "relevant_results_count": len(relevant_results),
                    "suggestions_generated": len(enrichment_suggestions)
                }
            }
            
            # Calculate confidence based on results quality
            confidence = self._calculate_enrichment_confidence(
                search_terms, web_results, relevant_results, enrichment_suggestions
            )
            
            return AgentResult(
                success=True,
                data=result_data,
                confidence=confidence,
                metadata={
                    "search_provider": self.search_provider,
                    "web_search_performed": True,
                    "enrichment_strategies_used": self.enrichment_strategies
                }
            )
            
        except Exception as e:
            self.log_error(f"Error in RAG enrichment: {str(e)}")
            return AgentResult(
                success=False,
                error=f"RAG enrichment failed: {str(e)}"
            )
    
    async def _extract_search_terms(self, documents: List[Document], 
                                  metadata: Dict[str, Any]) -> List[str]:
        """Extract key terms for web search"""
        search_terms = set()
        
        # Extract from metadata keywords
        if metadata.get('keywords'):
            search_terms.update(metadata['keywords'][:10])  # Top 10 keywords
        
        if metadata.get('ai_keywords'):
            search_terms.update(metadata['ai_keywords'][:10])
        
        # Extract from document content
        combined_content = ' '.join([doc.page_content for doc in documents])
        
        # Extract technical terms (capitalized, numbers, acronyms)
        technical_terms = re.findall(r'\b[A-Z]{2,}\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', combined_content)
        search_terms.update(technical_terms[:15])
        
        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', combined_content)
        search_terms.update(quoted_terms[:5])
        
        # Use AI to extract key concepts if available
        if len(search_terms) < 5:
            ai_terms = await self._extract_key_concepts_ai(combined_content[:3000])
            search_terms.update(ai_terms)
        
        # Clean and filter search terms
        cleaned_terms = []
        for term in search_terms:
            if isinstance(term, str) and 3 <= len(term) <= 50:
                cleaned_terms.append(term.strip())
        
        return list(set(cleaned_terms))[:20]  # Limit to 20 terms
    
    async def _extract_key_concepts_ai(self, content: str) -> List[str]:
        """Extract key concepts using AI"""
        try:
            llm = self.get_llm_client()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract 5-10 key concepts, terms, or topics from this content that would be good for web search."),
                ("human", f"Content: {content}\n\nReturn as JSON array of strings: [\"term1\", \"term2\", ...]")
            ])
            
            response = await llm.ainvoke(prompt.format_messages())
            
            try:
                concepts = json.loads(response.content)
                return concepts if isinstance(concepts, list) else []
            except json.JSONDecodeError:
                # Fallback: extract from response text
                terms = re.findall(r'"([^"]+)"', response.content)
                return terms[:10]
                
        except Exception as e:
            self.log_warning(f"AI concept extraction failed: {e}")
            return []
    
    async def _perform_web_searches(self, search_terms: List[str]) -> List[WebSearchResult]:
        """Perform web searches for the given terms"""
        all_results = []
        
        # Group search terms to avoid too many API calls
        search_queries = self._create_search_queries(search_terms)
        
        for query in search_queries[:5]:  # Limit to 5 queries
            try:
                if self.search_provider == 'tavily':
                    results = await self._search_tavily(query)
                elif self.search_provider == 'serper':
                    results = await self._search_serper(query)
                else:
                    results = await self._search_duckduckgo(query)
                
                all_results.extend(results)
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.log_warning(f"Search failed for query '{query}': {e}")
                continue
        
        # Remove duplicates and sort by relevance
        unique_results = self._deduplicate_results(all_results)
        return sorted(unique_results, key=lambda x: x.relevance_score, reverse=True)[:self.max_web_results]
    
    def _create_search_queries(self, search_terms: List[str]) -> List[str]:
        """Create effective search queries from terms"""
        queries = []
        
        # Individual important terms
        for term in search_terms[:5]:
            queries.append(term)
        
        # Combine related terms
        if len(search_terms) > 1:
            queries.append(' '.join(search_terms[:3]))
        
        # Add definition queries for technical terms
        for term in search_terms[:3]:
            if term.isupper() or any(char.isdigit() for char in term):
                queries.append(f"{term} definition explanation")
        
        return queries
    
    async def _search_tavily(self, query: str) -> List[WebSearchResult]:
        """Search using Tavily API"""
        if not self.search_clients.get('tavily'):
            return []
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.search_timeout)) as session:
                payload = {
                    "api_key": self.search_clients['tavily']['api_key'],
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": True,
                    "max_results": 5
                }
                
                async with session.post(
                    self.search_clients['tavily']['base_url'],
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_tavily_results(data, query)
                    else:
                        self.log_warning(f"Tavily search failed with status: {response.status}")
                        return []
                        
        except Exception as e:
            self.log_error(f"Tavily search error: {e}")
            return []
    
    async def _search_serper(self, query: str) -> List[WebSearchResult]:
        """Search using Serper API"""
        if not self.search_clients.get('serper'):
            return []
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.search_timeout)) as session:
                headers = {
                    'X-API-KEY': self.search_clients['serper']['api_key'],
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'q': query,
                    'num': 5
                }
                
                async with session.post(
                    self.search_clients['serper']['base_url'],
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_serper_results(data, query)
                    else:
                        self.log_warning(f"Serper search failed with status: {response.status}")
                        return []
                        
        except Exception as e:
            self.log_error(f"Serper search error: {e}")
            return []
    
    async def _search_duckduckgo(self, query: str) -> List[WebSearchResult]:
        """Search using DuckDuckGo (fallback)"""
        try:
            # Simple fallback implementation
            # In a real implementation, you might use duckduckgo-search library
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=5):
                    results.append(WebSearchResult(
                        title=r.get('title', ''),
                        url=r.get('href', ''),
                        snippet=r.get('body', ''),
                        content=r.get('body', ''),
                        relevance_score=0.5,  # Default score
                        source_domain=r.get('href', '').split('/')[2] if '/' in r.get('href', '') else ''
                    ))
                return results
                
        except ImportError:
            self.log_warning("DuckDuckGo search library not available")
            return []
        except Exception as e:
            self.log_error(f"DuckDuckGo search error: {e}")
            return []
    
    def _parse_tavily_results(self, data: Dict[str, Any], query: str) -> List[WebSearchResult]:
        """Parse Tavily search results"""
        results = []
        
        for result in data.get('results', []):
            results.append(WebSearchResult(
                title=result.get('title', ''),
                url=result.get('url', ''),
                snippet=result.get('content', ''),
                content=result.get('raw_content', result.get('content', ''))[:self.max_content_length],
                relevance_score=result.get('score', 0.5),
                source_domain=result.get('url', '').split('/')[2] if '/' in result.get('url', '') else '',
                publish_date=result.get('published_date')
            ))
        
        return results
    
    def _parse_serper_results(self, data: Dict[str, Any], query: str) -> List[WebSearchResult]:
        """Parse Serper search results"""
        results = []
        
        for result in data.get('organic', []):
            results.append(WebSearchResult(
                title=result.get('title', ''),
                url=result.get('link', ''),
                snippet=result.get('snippet', ''),
                content=result.get('snippet', '')[:self.max_content_length],
                relevance_score=0.7,  # Default score for Serper
                source_domain=result.get('link', '').split('/')[2] if '/' in result.get('link', '') else '',
                publish_date=result.get('date')
            ))
        
        return results
    
    def _deduplicate_results(self, results: List[WebSearchResult]) -> List[WebSearchResult]:
        """Remove duplicate search results"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    async def _filter_and_score_results(self, web_results: List[WebSearchResult],
                                      documents: List[Document],
                                      metadata: Dict[str, Any]) -> List[WebSearchResult]:
        """Filter and score web results for relevance"""
        if not web_results:
            return []
        
        # Create document context for relevance scoring
        document_context = self._create_document_context(documents, metadata)
        
        filtered_results = []
        
        for result in web_results:
            # Skip if from untrusted domain (unless no trusted results found)
            if self.trusted_domains and not any(domain in result.source_domain for domain in self.trusted_domains):
                continue
            
            # Score relevance using AI
            relevance_score = await self._score_relevance_ai(document_context, result)
            
            if relevance_score >= self.min_relevance_score:
                result.relevance_score = relevance_score
                filtered_results.append(result)
        
        # If no trusted results, fall back to all results above threshold
        if not filtered_results:
            for result in web_results:
                relevance_score = await self._score_relevance_ai(document_context, result)
                if relevance_score >= self.min_relevance_score:
                    result.relevance_score = relevance_score
                    filtered_results.append(result)
        
        return sorted(filtered_results, key=lambda x: x.relevance_score, reverse=True)
    
    def _create_document_context(self, documents: List[Document], metadata: Dict[str, Any]) -> str:
        """Create context summary for relevance scoring"""
        context_parts = []
        
        # Add title and subject
        if metadata.get('title'):
            context_parts.append(f"Title: {metadata['title']}")
        
        if metadata.get('subject'):
            context_parts.append(f"Subject: {metadata['subject']}")
        
        # Add keywords
        keywords = metadata.get('keywords', []) + metadata.get('ai_keywords', [])
        if keywords:
            context_parts.append(f"Keywords: {', '.join(keywords[:10])}")
        
        # Add content sample
        content_sample = ' '.join([doc.page_content for doc in documents])[:1000]
        context_parts.append(f"Content sample: {content_sample}")
        
        return '\n'.join(context_parts)
    
    async def _score_relevance_ai(self, document_context: str, 
                                web_result: WebSearchResult) -> float:
        """Score relevance using AI"""
        try:
            llm = self.get_llm_client()
            
            prompt = self.relevance_scoring_prompt.format_prompt(
                document_context=document_context[:2000],
                web_title=web_result.title,
                web_content=web_result.content[:1000]
            )
            
            response = await llm.ainvoke(prompt.to_messages())
            
            try:
                result_data = json.loads(response.content)
                return float(result_data.get('relevance_score', 0.5))
            except json.JSONDecodeError:
                # Fallback: extract score from text
                score_match = re.search(r'(\d+\.?\d*)', response.content)
                if score_match:
                    return min(1.0, max(0.0, float(score_match.group(1))))
                return 0.5
                
        except Exception as e:
            self.log_warning(f"AI relevance scoring failed: {e}")
            return 0.5
    
    async def _generate_enrichment_suggestions(self, documents: List[Document],
                                             web_results: List[WebSearchResult]) -> List[EnrichmentSuggestion]:
        """Generate suggestions for enriching documents"""
        if not web_results:
            return []
        
        try:
            llm = self.get_llm_client()
            
            # Prepare document content
            document_content = '\n\n'.join([doc.page_content for doc in documents])[:4000]
            
            # Prepare web sources
            web_sources = []
            for result in web_results[:5]:
                web_sources.append(f"Title: {result.title}\nURL: {result.url}\nContent: {result.content[:500]}")
            
            web_sources_text = '\n\n---\n\n'.join(web_sources)
            
            prompt = self.enrichment_suggestion_prompt.format_prompt(
                document_content=document_content,
                web_sources=web_sources_text
            )
            
            response = await llm.ainvoke(prompt.to_messages())
            
            try:
                suggestions_data = json.loads(response.content)
                
                suggestions = []
                for item in suggestions_data:
                    if isinstance(item, dict):