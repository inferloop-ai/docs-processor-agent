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
        client = self.get_client()
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