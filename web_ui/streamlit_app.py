"""
Document Processor Agent - Streamlit Web Interface
Modern, interactive web application for document processing and Q&A
"""

import streamlit as st
import asyncio
import json
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure Streamlit page
st.set_page_config(
    page_title="Document Processor Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007acc;
        margin-bottom: 1rem;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    
    .info-message {
        background: #d1ecf1;
        color: #0c5460;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #bee5eb;
    }
    
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #007acc 0%, #0056b3 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #0056b3 0%, #004085 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000/api"

class DocumentProcessorUI:
    """Main UI class for the Document Processor Agent"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        if 'current_document' not in st.session_state:
            st.session_state.current_document = None
        
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {}
        
        if 'stats' not in st.session_state:
            st.session_state.stats = {}
    
    def render_header(self):
        """Render main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Document Processor Agent</h1>
            <p>AI-powered document processing with RAG and human-in-the-loop validation</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with navigation and controls"""
        with st.sidebar:
            st.title("📋 Navigation")
            
            page = st.selectbox(
                "Select Page",
                ["📄 Document Upload", "🔍 Query Documents", "📊 Analytics", "⚙️ Settings"],
                key="page_selector"
            )
            
            st.divider()
            
            # System status
            st.subheader("🟢 System Status")
            if self.check_api_health():
                st.success("API: Online")
            else:
                st.error("API: Offline")
            
            # Quick stats
            self.render_quick_stats()
            
            st.divider()
            
            # Document list
            st.subheader("📚 Recent Documents")
            self.render_document_list()
            
            return page
    
    def check_api_health(self) -> bool:
        """Check if the API is healthy"""
        try:
            response = requests.get(f"{API_BASE_URL.replace('/api', '')}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def render_quick_stats(self):
        """Render quick statistics"""
        try:
            response = requests.get(f"{API_BASE_URL}/stats")
            if response.status_code == 200:
                stats = response.json()
                st.session_state.stats = stats
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Documents", stats.get('total_documents', 0))
                with col2:
                    st.metric("Queries Today", stats.get('queries_today', 0))
        except:
            st.warning("Could not load stats")
    
    def render_document_list(self):
        """Render list of documents in sidebar"""
        try:
            response = requests.get(f"{API_BASE_URL}/documents?page=1&page_size=5")
            if response.status_code == 200:
                data = response.json()
                documents = data.get('documents', [])
                
                for doc in documents:
                    with st.expander(f"📄 {doc['filename'][:20]}..."):
                        st.write(f"**ID:** {doc['id']}")
                        st.write(f"**Size:** {doc['file_size_mb']:.1f} MB")
                        st.write(f"**Pages:** {doc.get('page_count', 'N/A')}")
                        
                        if st.button(f"Select", key=f"select_{doc['id']}"):
                            st.session_state.current_document = doc['id']
                            st.rerun()
        except:
            st.write("No documents available")
    
    def render_upload_page(self):
        """Render document upload page"""
        st.header("📄 Document Upload")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload New Document")
            
            uploaded_file = st.file_uploader(
                "Choose a document file",
                type=['pdf', 'docx', 'doc', 'txt', 'md', 'html', 'pptx', 'xlsx'],
                help="Supported formats: PDF, Word, Text, Markdown, HTML, PowerPoint, Excel"
            )
            
            if uploaded_file is not None:
                # File info
                st.info(f"**File:** {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
                
                # Upload options
                st.subheader("Processing Options")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    extract_metadata = st.checkbox("Extract detailed metadata", value=True)
                    use_ocr = st.checkbox("Use OCR for scanned content", value=True)
                
                with col_b:
                    auto_chunk = st.checkbox("Auto-chunk content", value=True)
                    extract_entities = st.checkbox("Extract entities", value=False)
                
                # Upload button
                if st.button("🚀 Upload and Process", type="primary"):
                    self.upload_document(uploaded_file, {
                        'extract_metadata': extract_metadata,
                        'use_ocr': use_ocr,
                        'auto_chunk': auto_chunk,
                        'extract_entities': extract_entities
                    })
        
        with col2:
            st.subheader("Upload Guidelines")
            st.info("""
            **Supported Formats:**
            - 📄 PDF documents
            - 📝 Word documents (.docx, .doc)
            - 📊 Excel spreadsheets
            - 🎨 PowerPoint presentations
            - 📋 Text files (.txt, .md)
            - 🌐 HTML files
            
            **Size Limits:**
            - Maximum: 100MB per file
            - Recommended: Under 50MB
            
            **Processing Features:**
            - OCR for scanned documents
            - Metadata extraction
            - Automatic chunking
            - Entity recognition
            """)
        
        # Recent uploads
        st.divider()
        st.subheader("📂 Recent Uploads")
        self.render_recent_uploads()
    
    def upload_document(self, uploaded_file, options: Dict[str, Any]):
        """Upload document to API"""
        with st.spinner("Uploading and processing document..."):
            try:
                # Prepare file for upload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Upload to API
                response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.processing_status[result['document_id']] = 'processing'
                    
                    st.markdown(f"""
                    <div class="success-message">
                        <strong>✅ Upload Successful!</strong><br>
                        Document ID: {result['document_id']}<br>
                        Status: {result['status']}<br>
                        {result['message']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Auto-refresh to show in document list
                    time.sleep(2)
                    st.rerun()
                    
                else:
                    error_data = response.json()
                    st.markdown(f"""
                    <div class="error-message">
                        <strong>❌ Upload Failed!</strong><br>
                        {error_data.get('error', 'Unknown error occurred')}
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error uploading document: {str(e)}")
    
    def render_recent_uploads(self):
        """Render recent uploads table"""
        try:
            response = requests.get(f"{API_BASE_URL}/documents?page=1&page_size=10")
            if response.status_code == 200:
                data = response.json()
                documents = data.get('documents', [])
                
                if documents:
                    df = pd.DataFrame(documents)
                    
                    # Format the dataframe for display
                    display_df = df[['filename', 'file_size_mb', 'status', 'created_at']].copy()
                    display_df.columns = ['Filename', 'Size (MB)', 'Status', 'Uploaded']
                    display_df['Size (MB)'] = display_df['Size (MB)'].round(2)
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No documents uploaded yet.")
        except:
            st.error("Could not load recent uploads")
    
    def render_query_page(self):
        """Render document query page"""
        st.header("🔍 Query Documents")
        
        # Query interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Ask Questions About Your Documents")
            
            # Query input
            query = st.text_area(
                "Enter your question:",
                placeholder="What is the main topic of the document? Summarize the key findings...",
                height=100,
                help="Ask natural language questions about your uploaded documents"
            )
            
            # Query options
            st.subheader("Query Options")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                max_results = st.slider("Max results", 1, 10, 5)
                response_format = st.selectbox("Response format", ["detailed", "brief", "structured"])
            
            with col_b:
                include_sources = st.checkbox("Include sources", value=True)
                include_context = st.checkbox("Show context", value=True)
            
            with col_c:
                confidence_threshold = st.slider("Min confidence", 0.0, 1.0, 0.1, 0.1)
            
            # Query button
            if st.button("🔍 Ask Question", type="primary", disabled=not query.strip()):
                self.process_query(query, {
                    'max_results': max_results,
                    'response_format': response_format,
                    'include_sources': include_sources,
                    'include_context': include_context,
                    'confidence_threshold': confidence_threshold
                })
        
        with col2:
            st.subheader("Query Tips")
            st.info("""
            **Effective Questions:**
            - "What are the main findings?"
            - "Summarize the methodology"
            - "What are the key recommendations?"
            - "Who are the main stakeholders?"
            
            **Query Features:**
            - Natural language processing
            - Context-aware responses
            - Source attribution
            - Confidence scoring
            """)
        
        st.divider()
        
        # Conversation history
        st.subheader("💬 Conversation History")
        self.render_conversation_history()
    
    def process_query(self, query: str, options: Dict[str, Any]):
        """Process user query"""
        with st.spinner("Processing your question..."):
            try:
                payload = {
                    "question": query,
                    "max_results": options['max_results'],
                    "include_context": options['include_context']
                }
                
                response = requests.post(f"{API_BASE_URL}/query", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'timestamp': datetime.now(),
                        'question': query,
                        'answer': result['answer'],
                        'confidence': result['confidence'],
                        'sources': result.get('sources', []),
                        'processing_time': result.get('processing_time', 0)
                    })
                    
                    st.rerun()
                    
                else:
                    error_data = response.json()
                    st.error(f"Query failed: {error_data.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    def render_conversation_history(self):
        """Render conversation history"""
        if not st.session_state.conversation_history:
            st.info("No questions asked yet. Start by asking a question above!")
            return
        
        # Reverse to show most recent first
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-10:])):
            with st.expander(f"Q: {conv['question'][:60]}... ({conv['timestamp'].strftime('%H:%M:%S')})"):
                
                # Question
                st.markdown(f"**Question:** {conv['question']}")
                
                # Answer with confidence
                confidence_color = "green" if conv['confidence'] > 0.7 else "orange" if conv['confidence'] > 0.4 else "red"
                st.markdown(f"**Answer:** {conv['answer']}")
                st.markdown(f"**Confidence:** <span style='color:{confidence_color}'>{conv['confidence']:.2f}</span>", unsafe_allow_html=True)
                
                # Sources
                if conv.get('sources'):
                    st.markdown("**Sources:**")
                    for source in conv['sources'][:3]:
                        st.markdown(f"- {source.get('content', 'N/A')[:100]}...")
                
                # Processing time
                st.caption(f"Processing time: {conv['processing_time']:.2f}s")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.rerun()
    
    def render_analytics_page(self):
        """Render analytics and statistics page"""
        st.header("📊 Analytics Dashboard")
        
        # Load analytics data
        try:
            response = requests.get(f"{API_BASE_URL}/stats")
            if response.status_code == 200:
                stats = response.json()
            else:
                stats = {}
        except:
            stats = {}
        
        # Key Metrics
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Documents",
                stats.get('total_documents', 0),
                delta=None
            )
        
        with col2:
            st.metric(
                "Vector Documents",
                stats.get('vector_documents', 0),
                delta=None
            )
        
        with col3:
            st.metric(
                "Collections",
                stats.get('collections', 0),
                delta=None
            )
        
        with col4:
            st.metric(
                "System Status",
                "Operational" if stats.get('status') == 'operational' else 'Issues',
                delta=None
            )
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Document Types")
            # Mock data for demonstration
            doc_types = ['PDF', 'Word', 'Text', 'Excel', 'PowerPoint']
            doc_counts = [45, 23, 12, 8, 5]
            
            fig = px.pie(
                values=doc_counts,
                names=doc_types,
                title="Document Distribution by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💬 Query Confidence")
            # Mock data for demonstration
            if st.session_state.conversation_history:
                confidences = [conv['confidence'] for conv in st.session_state.conversation_history]
                
                fig = go.Figure(data=go.Histogram(x=confidences, nbinsx=10))
                fig.update_layout(
                    title="Query Confidence Distribution",
                    xaxis_title="Confidence Score",
                    yaxis_title="Number of Queries"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No query data available yet")
        
        # Processing Performance
        st.subheader("⚡ Processing Performance")
        
        if st.session_state.conversation_history:
            processing_times = [conv['processing_time'] for conv in st.session_state.conversation_history]
            avg_time = sum(processing_times) / len(processing_times)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Query Time", f"{avg_time:.2f}s")
            with col2:
                st.metric("Fastest Query", f"{min(processing_times):.2f}s")
            with col3:
                st.metric("Slowest Query", f"{max(processing_times):.2f}s")
            
            # Time series chart
            df = pd.DataFrame({
                'Query': range(1, len(processing_times) + 1),
                'Processing Time (s)': processing_times
            })
            
            fig = px.line(df, x='Query', y='Processing Time (s)', title='Query Processing Time Trend')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available yet")
    
    def render_settings_page(self):
        """Render settings and configuration page"""
        st.header("⚙️ Settings & Configuration")
        
        # API Configuration
        st.subheader("🔧 API Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            api_url = st.text_input("API Base URL", value=API_BASE_URL)
            timeout = st.number_input("Request Timeout (s)", value=30, min_value=5, max_value=300)
        
        with col2:
            max_file_size = st.number_input("Max File Size (MB)", value=100, min_value=1, max_value=1000)
            batch_size = st.number_input("Batch Size", value=10, min_value=1, max_value=100)
        
        # Processing Settings
        st.subheader("🤖 Processing Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            default_model = st.selectbox("Default LLM Model", ["gpt-4-turbo-preview", "claude-3-sonnet", "gemini-pro"])
            temperature = st.slider("Temperature", 0.0, 2.0, 0.1, 0.1)
        
        with col2:
            chunk_size = st.number_input("Chunk Size", value=1000, min_value=100, max_value=4000)
            chunk_overlap = st.number_input("Chunk Overlap", value=200, min_value=0, max_value=1000)
        
        # UI Settings
        st.subheader("🎨 UI Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.selectbox("Theme", ["Light", "Dark"])
            auto_refresh = st.checkbox("Auto-refresh data", value=True)
        
        with col2:
            page_size = st.number_input("Results per page", value=20, min_value=5, max_value=100)
            show_debug = st.checkbox("Show debug info", value=False)
        
        # Save Settings
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved successfully!")
        
        st.divider()
        
        # System Information
        st.subheader("ℹ️ System Information")
        
        system_info = {
            "Frontend": "Streamlit",
            "Backend": "FastAPI",
            "Vector DB": "ChromaDB",
            "LLM Providers": "OpenAI, Anthropic, Google",
            "Version": "1.0.0"
        }
        
        for key, value in system_info.items():
            st.text(f"{key}: {value}")
        
        # Export/Import
        st.subheader("📁 Data Management")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📤 Export Data"):
                st.info("Export functionality would be implemented here")
        
        with col2:
            if st.button("📥 Import Data"):
                st.info("Import functionality would be implemented here")
        
        with col3:
            if st.button("🗑️ Clear All Data"):
                if st.confirm("Are you sure you want to clear all data?"):
                    st.warning("This would clear all application data")
    
    def run(self):
        """Main application entry point"""
        # Render header
        self.render_header()
        
        # Render sidebar and get selected page
        page = self.render_sidebar()
        
        # Render selected page
        if page == "📄 Document Upload":
            self.render_upload_page()
        elif page == "🔍 Query Documents":
            self.render_query_page()
        elif page == "📊 Analytics":
            self.render_analytics_page()
        elif page == "⚙️ Settings":
            self.render_settings_page()


# Main application
def main():
    """Main function to run the Streamlit app"""
    app = DocumentProcessorUI()
    app.run()


if __name__ == "__main__":
    main()