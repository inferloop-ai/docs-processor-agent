"""
Test suite for Document Processor Agent
Tests core functionality of all agents
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json

# Import agents for testing
import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentResult
from agents.document_loader import DocumentLoader
from agents.parser import DocumentParser
from agents.metadata_writer import MetadataWriter
from agents.embedder import EmbeddingAgent
from agents.qa_agent import QAAgent
from agents.validator import ValidatorAgent


class TestBaseAgent:
    """Test the base agent functionality"""
    
    def test_agent_result_creation(self):
        """Test AgentResult creation"""
        result = AgentResult(
            success=True,
            data={"test": "data"},
            confidence=0.8
        )
        
        assert result.success is True
        assert result.data["test"] == "data"
        assert result.confidence == 0.8
        assert result.error is None
    
    def test_agent_config_loading(self):
        """Test configuration loading"""
        config = {"test_key": "test_value"}
        
        class TestAgent(BaseAgent):
            async def process(self, input_data, **kwargs):
                return AgentResult(success=True)
        
        agent = TestAgent(config)
        assert agent.config["test_key"] == "test_value"


class TestDocumentLoader:
    """Test document loading functionality"""
    
    @pytest.fixture
    def sample_text_file(self):
        """Create a temporary text file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.\nFor testing purposes.")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.fixture
    def document_loader(self):
        """Create a DocumentLoader instance for testing"""
        config = {
            'agents': {
                'document_loader': {
                    'supported_formats': ['.txt', '.pdf', '.docx'],
                    'max_file_size_mb': 50
                }
            }
        }
        return DocumentLoader(config)
    
    @pytest.mark.asyncio
    async def test_text_file_loading(self, document_loader, sample_text_file):
        """Test loading a text file"""
        result = await document_loader.process(sample_text_file)
        
        assert result.success is True
        assert result.data is not None
        assert "documents" in result.data
        assert len(result.data["documents"]) > 0
        
        # Check document content
        doc = result.data["documents"][0]
        assert "This is a test document" in doc.page_content
    
    @pytest.mark.asyncio
    async def test_invalid_file_path(self, document_loader):
        """Test handling of invalid file path"""
        result = await document_loader.process("/nonexistent/file.txt")
        
        assert result.success is False
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_unsupported_format(self, document_loader):
        """Test handling unsupported file format"""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            result = await document_loader.process(temp_path)
            assert result.success is False
            assert "Unsupported format" in result.error
        finally:
            os.unlink(temp_path)


class TestDocumentParser:
    """Test document parsing functionality"""
    
    @pytest.fixture
    def document_parser(self):
        """Create a DocumentParser instance for testing"""
        config = {
            'embeddings': {
                'chunk_size': 1000,
                'chunk_overlap': 200
            },
            'llm': {
                'primary_provider': 'openai',
                'models': {
                    'openai': {
                        'model': 'gpt-4-turbo-preview',
                        'temperature': 0.1
                    }
                }
            }
        }
        return DocumentParser(config)
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        from langchain.schema import Document
        
        return [
            Document(
                page_content="This is the first page of a test document. It contains sample text for testing.",
                metadata={'page': 1, 'source': 'test.txt'}
            ),
            Document(
                page_content="This is the second page. It has different content to test parsing.",
                metadata={'page': 2, 'source': 'test.txt'}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_basic_metadata_extraction(self, document_parser, sample_documents):
        """Test basic metadata extraction"""
        result = await document_parser.process(sample_documents)
        
        assert result.success is True
        assert result.data is not None
        assert "metadata" in result.data
        
        metadata = result.data["metadata"]
        assert metadata["page_count"] == 2
        assert metadata["character_count"] > 0
        assert metadata["word_count"] > 0
    
    @pytest.mark.asyncio
    async def test_document_chunking(self, document_parser, sample_documents):
        """Test document chunking"""
        result = await document_parser.process(sample_documents)
        
        assert result.success is True
        assert "chunked_documents" in result.data
        
        chunked_docs = result.data["chunked_documents"]
        assert len(chunked_docs) >= len(sample_documents)
        
        # Check chunk metadata
        for chunk in chunked_docs:
            assert "chunk_id" in chunk.metadata
            assert "chunk_length" in chunk.metadata
    
    @pytest.mark.asyncio
    async def test_fingerprint_generation(self, document_parser, sample_documents):
        """Test document fingerprint generation"""
        result = await document_parser.process(sample_documents)
        
        assert result.success is True
        assert "fingerprint" in result.data
        assert len(result.data["fingerprint"]) == 64  # SHA256 hex length


class TestQAAgent:
    """Test Q&A functionality"""
    
    @pytest.fixture
    def qa_agent(self):
        """Create a QAAgent instance for testing"""
        config = {
            'llm': {
                'primary_provider': 'openai',
                'models': {
                    'openai': {
                        'model': 'gpt-4-turbo-preview',
                        'temperature': 0.1
                    }
                }
            }
        }
        return QAAgent(config)
    
    @pytest.fixture
    def sample_context_documents(self):
        """Create sample context documents"""
        return [
            {
                'content': 'The capital of France is Paris. It is known for the Eiffel Tower.',
                'metadata': {'source': 'geography.txt', 'page': 1},
                'score': 0.9
            },
            {
                'content': 'Paris is a major European city with a rich history and culture.',
                'metadata': {'source': 'history.txt', 'page': 5},
                'score': 0.8
            }
        ]
    
    @pytest.mark.asyncio
    async def test_basic_qa(self, qa_agent, sample_context_documents):
        """Test basic Q&A functionality"""
        query = "What is the capital of France?"
        
        with patch.object(qa_agent, 'get_llm_client') as mock_llm:
            # Mock OpenAI response
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="The capital of France is Paris."))]
            mock_llm.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            
            result = await qa_agent.process({
                'query': query,
                'context_documents': sample_context_documents
            })
        
        assert result.success is True
        assert result.data is not None
        assert "answer" in result.data
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_empty_query(self, qa_agent):
        """Test handling of empty query"""
        result = await qa_agent.process("")
        
        assert result.success is False
        assert "cannot be empty" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, qa_agent, sample_context_documents):
        """Test confidence score calculation"""
        query = "What is the capital of France?"
        
        with patch.object(qa_agent, 'get_llm_client') as mock_llm:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="The capital of France is Paris, which is well-known for its landmarks."))]
            mock_llm.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            
            result = await qa_agent.process({
                'query': query,
                'context_documents': sample_context_documents
            })
        
        assert result.success is True
        assert 0.1 <= result.confidence <= 0.95  # Confidence should be in valid range


class TestValidatorAgent:
    """Test validation functionality"""
    
    @pytest.fixture
    def validator_agent(self):
        """Create a ValidatorAgent instance for testing"""
        config = {
            'agents': {
                'validator': {
                    'confidence_threshold': 0.7,
                    'max_retries': 3
                }
            },
            'llm': {
                'primary_provider': 'openai'
            }
        }
        return ValidatorAgent(config)
    
    @pytest.mark.asyncio
    async def test_completeness_validation(self, validator_agent):
        """Test completeness validation"""
        input_data = {
            'query': 'What is machine learning?',
            'answer': 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.',
            'context_documents': [],
            'validation_type': 'quick'
        }
        
        result = await validator_agent.process(input_data)
        
        assert result.success is True
        assert result.data is not None
        assert "status" in result.data
        assert "overall_confidence" in result.data
        assert "detailed_results" in result.data
    
    @pytest.mark.asyncio
    async def test_relevance_validation(self, validator_agent):
        """Test relevance validation"""
        input_data = {
            'query': 'What is the weather like?',
            'answer': 'The capital of France is Paris.',  # Irrelevant answer
            'context_documents': [],
            'validation_type': 'quick'
        }
        
        result = await validator_agent.process(input_data)
        
        assert result.success is True
        # Should detect low relevance
        detailed_results = result.data.get("detailed_results", {})
        if "relevance" in detailed_results:
            assert detailed_results["relevance"]["score"] < 0.5
    
    @pytest.mark.asyncio
    async def test_empty_input_validation(self, validator_agent):
        """Test validation with missing required fields"""
        input_data = {
            'query': '',  # Empty query
            'answer': 'Some answer',
            'context_documents': []
        }
        
        result = await validator_agent.process(input_data)
        
        assert result.success is False
        assert "required" in result.error.lower()


class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    @pytest.fixture
    def sample_document_file(self):
        """Create a sample document file"""
        content = """
        # Test Document
        
        This is a test document for integration testing.
        
        ## Section 1
        Machine learning is a powerful technology that enables computers to learn from data.
        
        ## Section 2
        Natural language processing helps computers understand human language.
        
        The document contains information about AI technologies.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_document_processing_pipeline(self, sample_document_file):
        """Test the complete document processing pipeline"""
        config = {
            'agents': {
                'document_loader': {
                    'supported_formats': ['.txt'],
                    'max_file_size_mb': 50
                }
            },
            'embeddings': {
                'chunk_size': 500,
                'chunk_overlap': 100
            }
        }
        
        # Step 1: Load document
        loader = DocumentLoader(config)
        load_result = await loader.process(sample_document_file)
        
        assert load_result.success is True
        documents = load_result.data["documents"]
        
        # Step 2: Parse document
        parser = DocumentParser(config)
        parse_result = await parser.process(documents)
        
        assert parse_result.success is True
        metadata = parse_result.data["metadata"]
        chunked_docs = parse_result.data["chunked_documents"]
        
        # Verify metadata
        assert metadata["page_count"] > 0
        assert metadata["word_count"] > 0
        assert len(chunked_docs) > 0
        
        # Step 3: Test Q&A (mocked)
        qa_agent = QAAgent(config)
        
        with patch.object(qa_agent, 'get_llm_client') as mock_llm:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Machine learning enables computers to learn from data automatically."))]
            mock_llm.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            
            qa_result = await qa_agent.process({
                'query': 'What is machine learning?',
                'context_documents': [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': 0.8
                    } for doc in chunked_docs[:2]  # Use first 2 chunks
                ]
            })
        
        assert qa_result.success is True
        assert "answer" in qa_result.data
        assert qa_result.confidence > 0


@pytest.mark.asyncio
async def test_agent_error_handling():
    """Test error handling across agents"""
    config = {}
    
    # Test with invalid configuration
    loader = DocumentLoader(config)
    result = await loader.process("nonexistent_file.txt")
    
    assert result.success is False
    assert result.error is not None


def test_config_validation():
    """Test configuration validation"""
    # Test with minimal config
    config = {}
    agent = BaseAgent(config)
    
    # Should not raise exception
    assert isinstance(agent.config, dict)


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])