"""
Base Agent Class
Foundation class for all document processing agents
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import json
import os

# LLM imports
from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage

# Pydantic for data validation
from pydantic import BaseModel, Field


@dataclass
class AgentResult:
    """
    Standard result object returned by all agents
    
    Attributes:
        success: Whether the operation was successful
        data: The main result data
        error: Error message if operation failed
        confidence: Confidence score (0.0 to 1.0)
        metadata: Additional metadata about the operation
        execution_time: Time taken to execute in seconds
        agent_name: Name of the agent that produced this result
        timestamp: When the result was created
    """
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    agent_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'confidence': self.confidence,
            'metadata': self.metadata or {},
            'execution_time': self.execution_time,
            'agent_name': self.agent_name,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)


class AgentConfig(BaseModel):
    """Configuration model for agents"""
    agent_name: str = Field(default="base_agent")
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: float = Field(default=300.0, gt=0)
    log_level: str = Field(default="INFO")
    enable_metrics: bool = Field(default=True)
    cache_enabled: bool = Field(default=True)


class BaseAgent(ABC):
    """
    Base class for all document processing agents
    
    Provides common functionality including:
    - Configuration management
    - Logging setup
    - LLM client access
    - Error handling
    - Metrics collection
    - Caching
    - Retry logic
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.agent_name = self.__class__.__name__
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Agent configuration
        self.agent_config = self._parse_agent_config()
        
        # LLM clients cache
        self._llm_clients = {}
        
        # Metrics
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
        
        # Cache
        self.cache = {} if self.agent_config.cache_enabled else None
        
        self.logger.info(f"{self.agent_name} initialized with config: {self.agent_config}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the agent"""
        logger = logging.getLogger(f"agents.{self.agent_name.lower()}")
        
        log_level = self.config.get('log_level', 'INFO')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.agent_name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _parse_agent_config(self) -> AgentConfig:
        """Parse agent-specific configuration"""
        agent_config_dict = self.config.get('agents', {}).get(self.agent_name.lower(), {})
        agent_config_dict['agent_name'] = self.agent_name
        
        try:
            return AgentConfig(**agent_config_dict)
        except Exception as e:
            self.logger.warning(f"Invalid agent config, using defaults: {e}")
            return AgentConfig(agent_name=self.agent_name)
    
    def get_llm_client(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Get LLM client for the specified provider
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'google')
            model: Specific model to use
        
        Returns:
            LLM client instance
        """
        # Determine provider
        if provider is None:
            provider = self.config.get('llm', {}).get('primary_provider', 'openai')
        
        # Check cache
        cache_key = f"{provider}_{model or 'default'}"
        if cache_key in self._llm_clients:
            return self._llm_clients[cache_key]
        
        # Get LLM configuration
        llm_config = self.config.get('llm', {})
        provider_config = llm_config.get('models', {}).get(provider, {})
        
        if not provider_config:
            raise ValueError(f"No configuration found for LLM provider: {provider}")
        
        # Create client based on provider
        try:
            if provider == 'openai':
                client = ChatOpenAI(
                    model=model or provider_config.get('model', 'gpt-4-turbo-preview'),
                    temperature=provider_config.get('temperature', 0.1),
                    api_key=provider_config.get('api_key') or os.getenv('OPENAI_API_KEY'),
                    max_retries=self.agent_config.max_retries,
                    request_timeout=self.agent_config.timeout_seconds
                )
            
            elif provider == 'anthropic':
                client = ChatAnthropic(
                    model=model or provider_config.get('model', 'claude-3-sonnet-20240229'),
                    temperature=provider_config.get('temperature', 0.1),
                    anthropic_api_key=provider_config.get('api_key') or os.getenv('ANTHROPIC_API_KEY'),
                    max_retries=self.agent_config.max_retries,
                    timeout=self.agent_config.timeout_seconds
                )
            
            elif provider == 'google':
                client = ChatGoogleGenerativeAI(
                    model=model or provider_config.get('model', 'gemini-pro'),
                    temperature=provider_config.get('temperature', 0.1),
                    google_api_key=provider_config.get('api_key') or os.getenv('GOOGLE_API_KEY'),
                    max_retries=self.agent_config.max_retries,
                    timeout=self.agent_config.timeout_seconds
                )
            
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
            
            # Cache the client
            self._llm_clients[cache_key] = client
            self.logger.info(f"Created {provider} LLM client with model: {model or 'default'}")
            
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to create {provider} LLM client: {str(e)}")
            raise
    
    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Main processing method that must be implemented by each agent
        
        Args:
            input_data: The input data to process
            **kwargs: Additional keyword arguments
        
        Returns:
            AgentResult with the processing results
        """
        pass
    
    async def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Execute the agent with error handling, metrics, and retry logic
        
        Args:
            input_data: The input data to process
            **kwargs: Additional keyword arguments
        
        Returns:
            AgentResult with the processing results
        """
        start_time = time.time()
        execution_id = str(uuid.uuid4())[:8]
        
        self.logger.info(f"Starting execution {execution_id}")
        
        # Update metrics
        self.metrics['total_calls'] += 1
        
        # Retry logic
        for attempt in range(self.agent_config.max_retries + 1):
            try:
                # Check cache if enabled
                if self.cache is not None:
                    cache_key = self._generate_cache_key(input_data, kwargs)
                    if cache_key in self.cache:
                        self.logger.debug(f"Cache hit for execution {execution_id}")
                        cached_result = self.cache[cache_key]
                        cached_result.metadata = cached_result.metadata or {}
                        cached_result.metadata['from_cache'] = True
                        return cached_result
                
                # Execute the main processing
                result = await self.process(input_data, **kwargs)
                
                # Ensure result has required fields
                if not isinstance(result, AgentResult):
                    result = AgentResult(
                        success=True,
                        data=result,
                        agent_name=self.agent_name
                    )
                
                # Set agent name and execution time
                result.agent_name = self.agent_name
                result.execution_time = time.time() - start_time
                
                # Add execution metadata
                if result.metadata is None:
                    result.metadata = {}
                result.metadata.update({
                    'execution_id': execution_id,
                    'attempt': attempt + 1,
                    'agent_version': getattr(self, 'version', '1.0.0')
                })
                
                # Cache result if successful and caching is enabled
                if result.success and self.cache is not None:
                    cache_key = self._generate_cache_key(input_data, kwargs)
                    self.cache[cache_key] = result
                
                # Update metrics
                if result.success:
                    self.metrics['successful_calls'] += 1
                else:
                    self.metrics['failed_calls'] += 1
                
                self.metrics['total_execution_time'] += result.execution_time
                self.metrics['average_execution_time'] = (
                    self.metrics['total_execution_time'] / self.metrics['total_calls']
                )
                
                self.logger.info(
                    f"Execution {execution_id} completed in {result.execution_time:.2f}s "
                    f"with success={result.success}, confidence={result.confidence}"
                )
                
                return result
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == self.agent_config.max_retries:
                    # Final attempt failed
                    execution_time = time.time() - start_time
                    self.metrics['failed_calls'] += 1
                    self.metrics['total_execution_time'] += execution_time
                    self.metrics['average_execution_time'] = (
                        self.metrics['total_execution_time'] / self.metrics['total_calls']
                    )
                    
                    error_result = AgentResult(
                        success=False,
                        error=f"Agent execution failed after {self.agent_config.max_retries + 1} attempts: {str(e)}",
                        agent_name=self.agent_name,
                        execution_time=execution_time,
                        metadata={
                            'execution_id': execution_id,
                            'total_attempts': attempt + 1,
                            'final_error': str(e)
                        }
                    )
                    
                    self.logger.error(f"Execution {execution_id} failed permanently: {str(e)}")
                    return error_result
                
                # Wait before retry (exponential backoff)
                wait_time = min(2 ** attempt, 30)  # Cap at 30 seconds
                self.logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
    
    def _generate_cache_key(self, input_data: Any, kwargs: Dict[str, Any]) -> str:
        """Generate a cache key for the given input"""
        try:
            # Create a simple hash of the input data and kwargs
            import hashlib
            
            # Convert input to string representation
            input_str = str(input_data)
            kwargs_str = str(sorted(kwargs.items()))
            
            combined = f"{input_str}:{kwargs_str}:{self.agent_name}"
            return hashlib.md5(combined.encode()).hexdigest()
        except Exception:
            # If hashing fails, return a timestamp-based key (no caching)
            return f"no_cache_{time.time()}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            'agent_name': self.agent_name,
            'metrics': self.metrics.copy(),
            'cache_size': len(self.cache) if self.cache else 0,
            'config': self.agent_config.dict()
        }
    
    def clear_cache(self):
        """Clear the agent cache"""
        if self.cache is not None:
            self.cache.clear()
            self.logger.info("Agent cache cleared")
    
    def log_info(self, message: str, **kwargs):
        """Log info message with structured data"""
        extra_data = f" - {kwargs}" if kwargs else ""
        self.logger.info(f"{message}{extra_data}")
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        extra_data = f" - {kwargs}" if kwargs else ""
        self.logger.warning(f"{message}{extra_data}")
    
    def log_error(self, message: str, **kwargs):
        """Log error message with structured data"""
        extra_data = f" - {kwargs}" if kwargs else ""
        self.logger.error(f"{message}{extra_data}")
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        extra_data = f" - {kwargs}" if kwargs else ""
        self.logger.debug(f"{message}{extra_data}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform agent health check"""
        try:
            # Test LLM connection if configured
            llm_health = "unknown"
            try:
                llm = self.get_llm_client()
                # Simple test call
                test_messages = [{"role": "user", "content": "Hello"}]
                await llm.ainvoke(test_messages)
                llm_health = "healthy"
            except Exception as e:
                llm_health = f"unhealthy: {str(e)}"
            
            return {
                'agent_name': self.agent_name,
                'status': 'healthy',
                'llm_connection': llm_health,
                'cache_size': len(self.cache) if self.cache else 0,
                'metrics': self.metrics,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                'agent_name': self.agent_name,
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def __repr__(self) -> str:
        return f"{self.agent_name}(config={self.agent_config})"


class AgentRegistry:
    """Registry for managing multiple agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
    
    def register(self, agent: BaseAgent):
        """Register an agent"""
        self.agents[agent.agent_name] = agent
    
    def get(self, agent_name: str) -> Optional[BaseAgent]:
        """Get an agent by name"""
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names"""
        return list(self.agents.keys())
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all registered agents"""
        results = {}
        for name, agent in self.agents.items():
            results[name] = await agent.health_check()
        return results
    
    def get_metrics_all(self) -> Dict[str, Any]:
        """Get metrics from all registered agents"""
        results = {}
        for name, agent in self.agents.items():
            results[name] = agent.get_metrics()
        return results


# Global agent registry
agent_registry = AgentRegistry()


def register_agent(agent: BaseAgent):
    """Register an agent globally"""
    agent_registry.register(agent)


def get_agent(agent_name: str) -> Optional[BaseAgent]:
    """Get a registered agent by name"""
    return agent_registry.get(agent_name)


# Decorators for common agent functionality
def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to add retry logic to agent methods"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator


def measure_time(func):
    """Decorator to measure execution time"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if isinstance(result, AgentResult):
                result.execution_time = execution_time
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            # Log the execution time even on failure
            logger = logging.getLogger(__name__)
            logger.error(f"Function {func.__name__} failed after {execution_time:.2f}s: {str(e)}")
            raise
    return wrapper


def validate_input(schema: BaseModel):
    """Decorator to validate input using Pydantic schema"""
    def decorator(func):
        async def wrapper(self, input_data, **kwargs):
            try:
                # Validate input data
                if isinstance(input_data, dict):
                    validated_data = schema(**input_data)
                else:
                    validated_data = input_data
                
                return await func(self, validated_data, **kwargs)
            except Exception as e:
                return AgentResult(
                    success=False,
                    error=f"Input validation failed: {str(e)}",
                    agent_name=getattr(self, 'agent_name', 'unknown')
                )
        return wrapper
    return decorator