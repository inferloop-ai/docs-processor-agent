"""
Validator Agent
Multi-strategy validation for document processing and Q&A responses
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentResult


class ValidationStrategy(Enum):
    """Available validation strategies"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    CONSISTENCY = "consistency"
    FACTUAL = "factual"
    SEMANTIC = "semantic"
    QUALITY = "quality"


class ValidationLevel(Enum):
    """Validation thoroughness levels"""
    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    HUMAN_REVIEW = "human_review"


@dataclass
class ValidationResult:
    """Result from a single validation check"""
    strategy: ValidationStrategy
    passed: bool
    score: float
    confidence: float
    feedback: str
    suggestions: List[str]
    execution_time: float


class ValidationRequest(BaseModel):
    """Structured validation request"""
    query: str = Field(description="Original query or question")
    answer: str = Field(description="Answer to validate")
    context_documents: List[Dict[str, Any]] = Field(description="Context used for answer")
    validation_strategies: List[str] = Field(default=["completeness", "relevance", "accuracy"])
    validation_level: str = Field(default="standard")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class ValidationResponse(BaseModel):
    """Structured validation response"""
    overall_status: str = Field(description="PASS, FAIL, or NEEDS_REVIEW")
    overall_confidence: float = Field(description="Overall confidence score")
    detailed_results: Dict[str, Dict[str, Any]] = Field(description="Results by strategy")
    recommendations: List[str] = Field(description="Improvement recommendations")
    human_review_required: bool = Field(description="Whether human review is needed")


class ValidatorAgent(BaseAgent):
    """
    Validator Agent
    
    Capabilities:
    - Multi-strategy validation (completeness, accuracy, relevance, etc.)
    - Configurable validation levels and thresholds
    - Human-in-the-loop integration
    - Quality scoring and feedback generation
    - Automated improvement suggestions
    - Batch validation support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Load validator-specific configuration
        self.validator_config = self.config.get('agents', {}).get('validator', {})
        
        # Validation settings
        self.confidence_threshold = self.validator_config.get('confidence_threshold', 0.7)
        self.validation_strategies = self.validator_config.get('validation_strategies', [
            'completeness', 'accuracy', 'relevance', 'consistency'
        ])
        self.max_retries = self.validator_config.get('max_retries', 3)
        self.human_review_threshold = self.validator_config.get('human_review_threshold', 0.5)
        
        # Initialize validation prompts
        self._setup_validation_prompts()
        
        # Validation metrics
        self.validation_history = []
        
        self.log_info("ValidatorAgent initialized")
    
    def _setup_validation_prompts(self):
        """Setup prompts for different validation strategies"""
        
        # Completeness validation
        self.completeness_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert validator checking the completeness of answers to questions.

Evaluate whether the answer adequately addresses all aspects of the question:
- Does it answer the main question?
- Are sub-questions or implied questions addressed?
- Is important context provided?
- Are key details missing?

Provide a score from 0.0 to 1.0 where:
- 1.0 = Completely addresses all aspects
- 0.7-0.9 = Addresses main points with minor gaps
- 0.4-0.6 = Partial answer with significant gaps
- 0.0-0.3 = Incomplete or inadequate answer"""),
            
            ("human", """Question: {query}

Answer: {answer}

Context Available: {context_summary}

Rate the completeness of this answer (0.0-1.0) and explain your reasoning. 
Also suggest what's missing if the score is below 0.8.

Format as JSON: {{"score": 0.0-1.0, "reasoning": "explanation", "missing_elements": ["item1", "item2"]}}""")
        ])
        
        # Relevance validation
        self.relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert validator checking the relevance of answers to questions.

Evaluate how well the answer stays on topic and directly addresses the question:
- Does the answer directly relate to the question asked?
- Is the information provided pertinent and useful?
- Are there tangential or off-topic elements?
- Is the level of detail appropriate?

Score from 0.0 to 1.0 where:
- 1.0 = Perfectly relevant and on-topic
- 0.7-0.9 = Mostly relevant with minor tangents
- 0.4-0.6 = Somewhat relevant but with significant off-topic content
- 0.0-0.3 = Largely irrelevant or off-topic"""),
            
            ("human", """Question: {query}

Answer: {answer}

Rate the relevance of this answer and provide specific feedback.

Format as JSON: {{"score": 0.0-1.0, "reasoning": "explanation", "improvements": ["suggestion1", "suggestion2"]}}""")
        ])
        
        # Accuracy validation (requires context)
        self.accuracy_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert validator checking the accuracy of answers against provided context.

Compare the answer to the available context documents:
- Are the facts stated in the answer supported by the context?
- Are there any contradictions between answer and context?
- Is information correctly attributed to sources?
- Are there unsupported claims?

Score from 0.0 to 1.0 where:
- 1.0 = All statements fully supported by context
- 0.7-0.9 = Mostly accurate with minor unsupported details
- 0.4-0.6 = Some accurate information but notable inaccuracies
- 0.0-0.3 = Significant inaccuracies or unsupported claims"""),
            
            ("human", """Question: {query}

Answer: {answer}

Context Documents:
{context}

Check the accuracy of the answer against the provided context.

Format as JSON: {{"score": 0.0-1.0, "reasoning": "explanation", "inaccuracies": ["issue1", "issue2"], "well_supported": ["fact1", "fact2"]}}""")
        ])
        
        # Consistency validation
        self.consistency_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert validator checking the internal consistency of answers.

Evaluate the answer for:
- Internal logical consistency
- Consistent terminology and definitions
- No contradictory statements
- Coherent flow of information

Score from 0.0 to 1.0 where:
- 1.0 = Perfectly consistent throughout
- 0.7-0.9 = Generally consistent with minor issues
- 0.4-0.6 = Some inconsistencies that affect clarity
- 0.0-0.3 = Significant inconsistencies or contradictions"""),
            
            ("human", """Question: {query}

Answer: {answer}

Check this answer for internal consistency and logical coherence.

Format as JSON: {{"score": 0.0-1.0, "reasoning": "explanation", "inconsistencies": ["issue1", "issue2"]}}""")
        ])
        
        # Quality validation
        self.quality_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert validator checking the overall quality of answers.

Evaluate the answer for:
- Clarity and readability
- Appropriate structure and organization
- Professional tone and language
- Actionable insights where applicable
- Overall helpfulness to the user

Score from 0.0 to 1.0 where:
- 1.0 = Excellent quality, clear, well-structured, highly helpful
- 0.7-0.9 = Good quality with minor improvements possible
- 0.4-0.6 = Average quality, needs improvement
- 0.0-0.3 = Poor quality, significant issues"""),
            
            ("human", """Question: {query}

Answer: {answer}

Evaluate the overall quality of this answer.

Format as JSON: {{"score": 0.0-1.0, "reasoning": "explanation", "quality_improvements": ["suggestion1", "suggestion2"]}}""")
        ])
    
    async def process(self, input_data: Union[Dict[str, Any], ValidationRequest], **kwargs) -> AgentResult:
        """
        Validate a query-answer pair
        
        Args:
            input_data: Dict or ValidationRequest with query, answer, and context
            **kwargs: Additional parameters
                - validation_level: Override default validation level
                - custom_strategies: List of specific strategies to use
                - require_human_review: Force human review regardless of scores
        
        Returns:
            AgentResult with validation status and detailed feedback
        """
        try:
            # Parse input
            if isinstance(input_data, dict):
                validation_request = ValidationRequest(**input_data)
            else:
                validation_request = input_data
            
            # Override settings if provided
            validation_level = kwargs.get('validation_level', validation_request.validation_level)
            strategies = kwargs.get('custom_strategies', validation_request.validation_strategies)
            
            # Validate required fields
            if not validation_request.query.strip():
                return AgentResult(
                    success=False,
                    error="Query is required for validation"
                )
            
            if not validation_request.answer.strip():
                return AgentResult(
                    success=False,
                    error="Answer is required for validation"
                )
            
            # Run validation strategies
            validation_results = await self._run_validation_strategies(
                validation_request, strategies, validation_level
            )
            
            # Compile overall results
            overall_result = self._compile_validation_results(
                validation_results, validation_request.confidence_threshold
            )
            
            # Check if human review is needed
            human_review_required = (
                kwargs.get('require_human_review', False) or
                overall_result['overall_confidence'] < self.human_review_threshold or
                overall_result['overall_status'] == 'FAIL'
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(validation_results)
            
            result_data = {
                "overall_status": overall_result['overall_status'],
                "overall_confidence": overall_result['overall_confidence'],
                "detailed_results": validation_results,
                "recommendations": recommendations,
                "human_review_required": human_review_required,
                "validation_metadata": {
                    "strategies_used": strategies,
                    "validation_level": validation_level,
                    "threshold": validation_request.confidence_threshold,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Store in history
            self.validation_history.append({
                "query": validation_request.query,
                "answer": validation_request.answer,
                "result": result_data,
                "timestamp": datetime.now()
            })
            
            return AgentResult(
                success=True,
                data=result_data,
                confidence=overall_result['overall_confidence'],
                metadata={
                    "validation_count": len(strategies),
                    "human_review_required": human_review_required
                }
            )
            
        except Exception as e:
            self.log_error(f"Error in validation: {str(e)}")
            return AgentResult(
                success=False,
                error=f"Validation failed: {str(e)}"
            )
    
    async def _run_validation_strategies(self, request: ValidationRequest, 
                                       strategies: List[str], level: str) -> Dict[str, Dict[str, Any]]:
        """Run specified validation strategies"""
        results = {}
        
        # Prepare context summary for strategies that need it
        context_summary = self._create_context_summary(request.context_documents)
        context_full = self._format_context_documents(request.context_documents)
        
        # Run strategies based on level
        if level == "quick":
            # Run essential strategies only
            essential_strategies = ['completeness', 'relevance']
            strategies = [s for s in strategies if s in essential_strategies]
        elif level == "comprehensive":
            # Add additional strategies
            strategies.extend(['quality', 'semantic'])
        
        # Execute validation strategies
        for strategy in strategies:
            try:
                result = await self._execute_validation_strategy(
                    strategy, request, context_summary, context_full
                )
                results[strategy] = result
            except Exception as e:
                self.log_warning(f"Strategy {strategy} failed: {e}")
                results[strategy] = {
                    "score": 0.5,
                    "reasoning": f"Validation strategy failed: {str(e)}",
                    "error": True
                }
        
        return results
    
    async def _execute_validation_strategy(self, strategy: str, request: ValidationRequest,
                                         context_summary: str, context_full: str) -> Dict[str, Any]:
        """Execute a single validation strategy"""
        start_time = time.time()
        
        try:
            llm = self.get_llm_client()
            
            # Select appropriate prompt
            if strategy == "completeness":
                prompt = self.completeness_prompt.format_prompt(
                    query=request.query,
                    answer=request.answer,
                    context_summary=context_summary
                )
            elif strategy == "relevance":
                prompt = self.relevance_prompt.format_prompt(
                    query=request.query,
                    answer=request.answer
                )
            elif strategy == "accuracy":
                prompt = self.accuracy_prompt.format_prompt(
                    query=request.query,
                    answer=request.answer,
                    context=context_full[:8000]  # Limit context size
                )
            elif strategy == "consistency":
                prompt = self.consistency_prompt.format_prompt(
                    query=request.query,
                    answer=request.answer
                )
            elif strategy == "quality":
                prompt = self.quality_prompt.format_prompt(
                    query=request.query,
                    answer=request.answer
                )
            else:
                # Fallback to generic validation
                return await self._generic_validation(strategy, request)
            
            # Get LLM response
            response = await llm.ainvoke(prompt.to_messages())
            
            # Parse response
            try:
                result_data = json.loads(response.content)
                
                return {
                    "score": float(result_data.get("score", 0.5)),
                    "reasoning": result_data.get("reasoning", f"{strategy} validation completed"),
                    "details": result_data,
                    "execution_time": time.time() - start_time
                }
                
            except json.JSONDecodeError:
                # Fallback parsing
                return self._parse_validation_response_fallback(response.content, strategy)
                
        except Exception as e:
            self.log_error(f"Error executing {strategy} validation: {e}")
            return {
                "score": 0.5,
                "reasoning": f"Validation failed: {str(e)}",
                "error": True,
                "execution_time": time.time() - start_time
            }
    
    def _parse_validation_response_fallback(self, response_content: str, strategy: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails"""
        # Try to extract score from response
        score_match = re.search(r'score["\s:]*([0-9.]+)', response_content.lower())
        score = float(score_match.group(1)) if score_match else 0.5
        
        return {
            "score": max(0.0, min(1.0, score)),
            "reasoning": f"{strategy.title()} validation - score extracted from response",
            "raw_response": response_content,
            "execution_time": 0
        }
    
    async def _generic_validation(self, strategy: str, request: ValidationRequest) -> Dict[str, Any]:
        """Generic validation for unsupported strategies"""
        # Basic heuristic validations
        if strategy == "length":
            score = min(1.0, len(request.answer) / 200)  # Reasonable length
            reasoning = "Length-based validation"
        elif strategy == "semantic":
            # Simple semantic check based on word overlap
            query_words = set(request.query.lower().split())
            answer_words = set(request.answer.lower().split())
            overlap = len(query_words.intersection(answer_words))
            score = min(1.0, overlap / max(len(query_words), 1))
            reasoning = "Word overlap semantic validation"
        else:
            score = 0.7  # Default neutral score
            reasoning = f"Generic validation for {strategy}"
        
        return {
            "score": score,
            "reasoning": reasoning,
            "generic": True
        }
    
    def _create_context_summary(self, context_docs: List[Dict[str, Any]]) -> str:
        """Create a summary of context documents"""
        if not context_docs:
            return "No context documents provided"
        
        summary_parts = []
        for i, doc in enumerate(context_docs[:3]):  # Limit to 3 docs for summary
            content = doc.get('content', '')
            source = doc.get('source', f'Document {i+1}')
            summary_parts.append(f"Source {source}: {content[:200]}...")
        
        return '\n'.join(summary_parts)
    
    def _format_context_documents(self, context_docs: List[Dict[str, Any]]) -> str:
        """Format context documents for validation"""
        if not context_docs:
            return "No context documents available"
        
        formatted_docs = []
        for i, doc in enumerate(context_docs):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            source = metadata.get('source', f'Document {i+1}')
            
            formatted_docs.append(f"[{source}]\n{content[:1000]}")
        
        return '\n\n'.join(formatted_docs)
    
    def _compile_validation_results(self, results: Dict[str, Dict[str, Any]], 
                                  threshold: float) -> Dict[str, Any]:
        """Compile individual validation results into overall assessment"""
        if not results:
            return {
                "overall_status": "FAIL",
                "overall_confidence": 0.0
            }
        
        # Calculate weighted average score
        total_score = 0
        total_weight = 0
        failed_strategies = []
        
        # Strategy weights (can be made configurable)
        strategy_weights = {
            'completeness': 0.25,
            'relevance': 0.25,
            'accuracy': 0.20,
            'consistency': 0.15,
            'quality': 0.10,
            'factual': 0.20,
            'semantic': 0.05
        }
        
        for strategy, result in results.items():
            weight = strategy_weights.get(strategy, 0.1)
            score = result.get('score', 0.5)
            
            total_score += score * weight
            total_weight += weight
            
            if score < threshold:
                failed_strategies.append(strategy)
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.5
        
        # Determine status
        if overall_score >= threshold and not failed_strategies:
            status = "PASS"
        elif overall_score >= threshold * 0.8:  # Close to threshold
            status = "NEEDS_REVIEW"
        else:
            status = "FAIL"
        
        return {
            "overall_status": status,
            "overall_confidence": overall_score,
            "failed_strategies": failed_strategies
        }
    
    def _generate_recommendations(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate improvement recommendations based on validation results"""
        recommendations = []
        
        for strategy, result in results.items():
            score = result.get('score', 0.5)
            details = result.get('details', {})
            
            if score < 0.7:  # Below good threshold
                if strategy == "completeness":
                    missing = details.get('missing_elements', [])
                    if missing:
                        recommendations.append(f"Address missing elements: {', '.join(missing)}")
                    else:
                        recommendations.append("Provide more comprehensive coverage of the question")
                
                elif strategy == "relevance":
                    improvements = details.get('improvements', [])
                    if improvements:
                        recommendations.extend(improvements)
                    else:
                        recommendations.append("Focus more directly on the specific question asked")
                
                elif strategy == "accuracy":
                    inaccuracies = details.get('inaccuracies', [])
                    if inaccuracies:
                        recommendations.append(f"Correct inaccuracies: {', '.join(inaccuracies)}")
                    else:
                        recommendations.append("Verify all facts against provided context")
                
                elif strategy == "consistency":
                    inconsistencies = details.get('inconsistencies', [])
                    if inconsistencies:
                        recommendations.append(f"Resolve inconsistencies: {', '.join(inconsistencies)}")
                    else:
                        recommendations.append("Ensure internal consistency throughout the answer")
                
                elif strategy == "quality":
                    quality_improvements = details.get('quality_improvements', [])
                    if quality_improvements:
                        recommendations.extend(quality_improvements)
                    else:
                        recommendations.append("Improve clarity, structure, and overall presentation")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Limit to 10 recommendations
    
    def get_validation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent validation history"""
        return self.validation_history[-limit:]
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self.validation_history:
            return {"total_validations": 0}
        
        total = len(self.validation_history)
        passed = sum(1 for v in self.validation_history 
                    if v['result']['overall_status'] == 'PASS')
        failed = sum(1 for v in self.validation_history 
                    if v['result']['overall_status'] == 'FAIL')
        needs_review = total - passed - failed
        
        avg_confidence = sum(v['result']['overall_confidence'] 
                           for v in self.validation_history) / total
        
        return {
            "total_validations": total,
            "passed": passed,
            "failed": failed,
            "needs_review": needs_review,
            "pass_rate": passed / total,
            "average_confidence": avg_confidence
        }
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for validation"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        self.log_info(f"Confidence threshold set to {self.confidence_threshold}")
    
    def add_custom_strategy(self, strategy_name: str, prompt_template: str):
        """Add a custom validation strategy"""
        # This would allow users to define custom validation strategies
        # Implementation would involve creating new prompt templates
        self.log_info(f"Custom validation strategy '{strategy_name}' would be added here")
    
    async def batch_validate(self, validation_requests: List[ValidationRequest]) -> List[AgentResult]:
        """Validate multiple query-answer pairs in batch"""
        tasks = [self.execute(request) for request in validation_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(AgentResult(
                    success=False,
                    error=str(result),
                    agent_name=self.agent_name
                ))
            else:
                final_results.append(result)
        
        return final_results