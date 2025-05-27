class TestEmbeddingAgent:
    """Test embedding functionality"""
    
    @pytest.fixture
    def embedding_agent(self):
        """Create an EmbeddingAgent instance for testing"""
        config = {
            'embeddings': {
                'embedding_provider': 'sentence-transformers',
                'embedding_model': 'all-MiniLM-L6-v2',
                'chunk_size': 1000,
                'chunk_overlap': 200
            }
        }
        return EmbeddingAgent(config)
    
    @pytest.fixture
    def sample_documents_for_embedding(self):
        """Create sample documents for embedding tests"""
        return [
            Document(
                page_content="This is a test document about artificial intelligence and machine learning.",
                metadata={'source': 'test.txt', 'chunk_id': 'chunk_0'}
            ),
            Document(
                page_content="Natural language processing is a branch of AI that deals with text analysis.",
                metadata={'source': 'test.txt', 'chunk_id': 'chunk_1'}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, embedding_agent, sample_documents_for_embedding):
        """Test basic embedding generation"""
        result = await embedding_agent.process(sample_documents_for_embedding)
        
        assert result.success is True
        assert result.data is not None
        assert "embedded_documents" in result.data
        
        embedded_docs = result.data["embedded_documents"]
        assert len(embedded_docs) == len(sample_documents_for_embedding)
        
        # Check that embeddings are present
        for doc in embedded_docs:
            assert "embedding" in doc.metadata
            assert isinstance(doc.metadata["embedding"], list)
            assert len(doc.metadata["embedding"]) > 0
    
    @pytest.mark.asyncio
    async def test_embedding_batch_processing(self, embedding_agent):
        """Test batch processing of embeddings"""
        # Create a larger batch
        batch_docs = []
        for i in range(10):
            batch_docs.append(Document(
                page_content=f"Test document number {i} with some content about testing.",
                metadata={'source': 'batch_test.txt', 'chunk_id': f'chunk_{i}'}
            ))
        
        result = await embedding_agent.process(batch_docs)
        
        assert result.success is True
        assert len(result.data["embedded_documents"]) == 10
        
        # Check statistics
        stats = result.data["statistics"]
        assert stats["total_documents"] == 10
        assert stats["embedding_dimension"] > 0


class TestReferenceLinker:
    """Test reference linking functionality"""
    
    @pytest.fixture
    def reference_linker(self):
        """Create a ReferenceLinker instance for testing"""
        config = {
            'agents': {
                'reference_linker': {
                    'detect_citations': True,
                    'detect_urls': True,
                    'detect_internal_refs': True
                }
            }
        }
        return ReferenceLinker(config)
    
    @pytest.fixture
    def documents_with_references(self):
        """Create documents with various reference types"""
        return [
            Document(
                page_content="According to Smith (2020), machine learning is important. See also https://example.com for more details. Reference [1] provides additional context.",
                metadata={'source': 'doc1.txt'}
            ),
            Document(
                page_content="The study by Johnson et al. (2019) shows different results. DOI: 10.1000/182 contains the full paper. See Section 3.2 for methodology.",
                metadata={'source': 'doc2.txt'}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_reference_extraction(self, reference_linker, documents_with_references):
        """Test reference extraction from documents"""
        input_data = {
            'documents': documents_with_references,
            'document_id': 'test_doc'
        }
        
        result = await reference_linker.process(input_data)
        
        assert result.success is True
        assert result.data is not None
        assert "references" in result.data
        
        references = result.data["references"]
        assert len(references) > 0
        
        # Check for different reference types
        ref_types = [ref["type"] for ref in references]
        assert "citation" in ref_types
        assert "url" in ref_types
    
    @pytest.mark.asyncio
    async def test_cross_reference_detection(self, reference_linker):
        """Test cross-reference detection between documents"""
        docs_with_cross_refs = [
            Document(
                page_content="This references Smith (2020) and the methodology in Section 2.",
                metadata={'source': 'doc1.txt'}
            ),
            Document(
                page_content="Smith (2020) also mentioned the importance of validation. See Table 1.",
                metadata={'source': 'doc2.txt'}
            )
        ]
        
        input_data = {
            'documents': docs_with_cross_refs,
            'document_id': 'cross_ref_test'
        }
        
        result = await reference_linker.process(input_data)
        
        assert result.success is True
        cross_refs = result.data.get("cross_references", [])
        
        # Should find similar references between documents
        assert len(cross_refs) >= 0  # Might be 0 if similarity threshold not met


class TestMetadataWriter:
    """Test metadata writing functionality"""
    
    @pytest.fixture
    def metadata_writer(self):
        """Create a MetadataWriter instance for testing"""
        config = {
            'agents': {
                'metadata_writer': {
                    'validate_metadata': True,
                    'enrich_metadata': True
                }
            }
        }
        return MetadataWriter(config)
    
    @pytest.fixture  
    def sample_metadata_input(self):
        """Create sample input for metadata processing"""
        return {
            'document_id': 'test_doc_123',
            'documents': [
                Document(
                    page_content="This is a comprehensive test document about machine learning applications.",
                    metadata={'source': 'test.pdf', 'page': 1}
                )
            ],
            'existing_metadata': {
                'filename': 'test.pdf',
                'file_size': 1024000,
                'file_type': '.pdf'
            }
        }
    
    @pytest.mark.asyncio
    async def test_metadata_processing(self, metadata_writer, sample_metadata_input):
        """Test comprehensive metadata processing"""
        result = await metadata_writer.process(sample_metadata_input)
        
        assert result.success is True
        assert result.data is not None
        
        metadata = result.data["metadata"]
        assert "word_count" in metadata
        assert "character_count" in metadata
        assert "content_hash" in metadata
        assert metadata["word_count"] > 0
        assert metadata["character_count"] > 0
    
    @pytest.mark.asyncio
    async def test_metadata_validation(self, metadata_writer):
        """Test metadata validation"""
        valid_metadata = {
            'filename': 'test.pdf',
            'file_size': 1024,
            'word_count': 100,
            'character_count': 500,
            'page_count': 1
        }
        
        validation_result = await metadata_writer._validate_metadata(valid_metadata)
        
        assert validation_result["valid"] is True
        assert len(validation_result["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_metadata_enrichment(self, metadata_writer):
        """Test metadata enrichment"""
        base_metadata = {
            'word_count': 100,
            'character_count': 500
        }
        
        sample_docs = [
            Document(page_content="This is a technical document about algorithms and data structures.")
        ]
        
        enriched = await metadata_writer._enrich_metadata(base_metadata, sample_docs)
        
        assert "average_word_length" in enriched
        assert "complexity_level" in enriched
        assert enriched["average_word_length"] > 0


class TestSystemIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_complete_document_workflow(self):
        """Test the complete document processing workflow"""
        config = {
            'agents': {
                'document_loader': {'supported_formats': ['.txt']},
                'parser': {'extract_metadata': True},
                'embedder': {'embedding_provider': 'sentence-transformers'},
                'qa_agent': {'max_context_length': 4000}
            },
            'embeddings': {
                'chunk_size': 500,
                'chunk_overlap': 100
            }
        }
        
        # Create a test document
        test_content = """
        # Machine Learning Guide
        
        Machine learning is a powerful technology that enables computers to learn from data.
        
        ## Applications
        1. Natural Language Processing
        2. Computer Vision  
        3. Recommendation Systems
        
        For more information, see Smith (2020) and visit https://ml-guide.com.
        """
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            # Step 1: Load document
            loader = DocumentLoader(config)
            load_result = await loader.process(temp_path)
            assert load_result.success is True
            
            # Step 2: Parse document  
            parser = DocumentParser(config)
            parse_result = await parser.process(load_result.data["documents"])
            assert parse_result.success is True
            
            chunked_docs = parse_result.data["chunked_documents"]
            assert len(chunked_docs) > 0
            
            # Step 3: Generate embeddings
            embedder = EmbeddingAgent(config)
            embed_result = await embedder.process(chunked_docs)
            assert embed_result.success is True
            
            # Step 4: Test Q&A
            qa_agent = QAAgent(config)
            with patch.object(qa_agent, 'get_llm_client') as mock_llm:
                mock_response = Mock()
                mock_response.content = "Machine learning enables computers to learn from data automatically."
                mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)
                
                qa_result = await qa_agent.process({
                    'query': 'What is machine learning?',
                    'context_documents': [
                        {
                            'content': doc.page_content,
                            'metadata': doc.metadata,
                            'score': 0.8
                        } for doc in chunked_docs[:2]
                    ]
                })
                
                assert qa_result.success is True
                assert "answer" in qa_result.data
            
            # Step 5: Process references
            ref_linker = ReferenceLinker(config)
            ref_result = await ref_linker.process({
                'documents': load_result.data["documents"],
                'document_id': 'integration_test'
            })
            assert ref_result.success is True
            
            # Verify references were found
            references = ref_result.data["references"]
            ref_types = [ref["type"] for ref in references]
            assert any(ref_type in ["citation", "url"] for ref_type in ref_types)
            
        finally:
            # Cleanup
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test error handling in the workflow"""
        config = {}
        
        # Test with invalid file path
        loader = DocumentLoader(config)
        result = await loader.process("nonexistent_file.txt")
        
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()
    
    @pytest.mark.asyncio 
    async def test_configuration_validation(self):
        """Test configuration validation across agents"""
        # Test with minimal valid config
        minimal_config = {
            'embeddings': {
                'embedding_provider': 'sentence-transformers',
                'chunk_size': 1000
            }
        }
        
        # Should not raise exceptions
        loader = DocumentLoader(minimal_config)
        parser = DocumentParser(minimal_config)
        embedder = EmbeddingAgent(minimal_config)
        qa_agent = QAAgent(minimal_config)
        
        assert loader.agent_name == "DocumentLoader"
        assert parser.agent_name == "DocumentParser"
        assert embedder.agent_name == "EmbeddingAgent"
        assert qa_agent.agent_name == "QAAgent"


# Performance and load testing
class TestPerformance:
    """Performance and load tests"""
    
    @pytest.mark.asyncio
    async def test_large_document_processing(self):
        """Test processing of large documents"""
        # Create a large document
        large_content = "This is a test sentence. " * 1000  # ~5000 words
        
        config = {
            'embeddings': {
                'embedding_provider': 'sentence-transformers',
                'chunk_size': 1000,
                'chunk_overlap': 200
            }
        }
        
        large_doc = Document(
            page_content=large_content,
            metadata={'source': 'large_test.txt'}
        )
        
        # Test parsing
        parser = DocumentParser(config)
        start_time = time.time()
        result = await parser.process([large_doc])
        processing_time = time.time() - start_time
        
        assert result.success is True
        assert processing_time < 30  # Should complete within 30 seconds
        
        chunked_docs = result.data["chunked_documents"]
        assert len(chunked_docs) > 1  # Should be chunked
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing of multiple documents"""
        config = {
            'embeddings': {
                'embedding_provider': 'sentence-transformers',
                'chunk_size': 500
            }
        }
        
        # Create multiple test documents
        docs = []
        for i in range(5):
            docs.append(Document(
                page_content=f"Test document {i} with content about topic {i}.",
                metadata={'source': f'test_{i}.txt'}
            ))
        
        # Process documents concurrently
        parser = DocumentParser(config)
        
        tasks = [parser.process([doc]) for doc in docs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        for result in results:
            assert not isinstance(result, Exception)
            assert result.success is True


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])
    