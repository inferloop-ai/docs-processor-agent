#!/bin/bash

# Document Processor Agent - Automated Setup Script
# This script sets up the complete development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default Python version
PYTHON_VERSION="3.11"

# Print functions
print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_header() { echo -e "\n${BLUE}=== $1 ===${NC}"; }

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --python VERSION    Python version to use (default: 3.11)"
    echo "  -d, --dev               Install development dependencies"
    echo "  -c, --clean             Clean existing installation"
    echo "  -s, --skip-deps         Skip system dependencies installation"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Basic setup"
    echo "  $0 --dev               # Setup with development tools"
    echo "  $0 --python 3.12       # Use Python 3.12"
    echo "  $0 --clean --dev       # Clean install with dev tools"
    echo ""
}

# Parse command line arguments
INSTALL_DEV=false
CLEAN_INSTALL=false
SKIP_SYSTEM_DEPS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        -d|--dev)
            INSTALL_DEV=true
            shift
            ;;
        -c|--clean)
            CLEAN_INSTALL=true
            shift
            ;;
        -s|--skip-deps)
            SKIP_SYSTEM_DEPS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main setup function
main() {
    print_header "Document Processor Agent Setup"
    echo "Setting up the AI-powered document processing system..."
    echo "Project directory: $PROJECT_DIR"
    echo "Python version: $PYTHON_VERSION"
    echo "Development mode: $INSTALL_DEV"
    echo "Clean install: $CLEAN_INSTALL"
    echo ""
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Perform setup steps
    check_system_requirements
    
    if [ "$SKIP_SYSTEM_DEPS" = false ]; then
        install_system_dependencies
    fi
    
    if [ "$CLEAN_INSTALL" = true ]; then
        clean_existing_installation
    fi
    
    setup_python_environment
    install_python_dependencies
    setup_directories
    setup_configuration
    install_nlp_models
    setup_database
    verify_installation
    
    print_header "Setup Complete!"
    print_status "Document Processor Agent is ready to use!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Configure your API keys in .env file"
    echo "3. Run the application: python app.py"
    echo ""
    echo "Documentation: README.md"
    echo "Web UI: http://localhost:8501 (Streamlit)"
    echo "API: http://localhost:8000 (FastAPI)"
    echo ""
}

# Check system requirements
check_system_requirements() {
    print_header "Checking System Requirements"
    
    # Check OS
    OS="$(uname -s)"
    print_info "Operating System: $OS"
    
    # Check Python
    if ! command -v python$PYTHON_VERSION &> /dev/null; then
        if ! command -v python3 &> /dev/null; then
            print_error "Python $PYTHON_VERSION not found. Please install Python $PYTHON_VERSION or higher."
            exit 1
        else
            PYTHON_CMD="python3"
            PYTHON_ACTUAL_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
            print_warning "Python $PYTHON_VERSION not found, using Python $PYTHON_ACTUAL_VERSION"
        fi
    else
        PYTHON_CMD="python$PYTHON_VERSION"
    fi
    
    print_status "Python: $($PYTHON_CMD --version)"
    
    # Check pip
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        print_error "pip not found. Please install pip."
        exit 1
    fi
    
    print_status "pip: $($PYTHON_CMD -m pip --version | cut -d' ' -f1,2)"
    
    # Check git
    if ! command -v git &> /dev/null; then
        print_warning "Git not found. Some features may not work properly."
    else
        print_status "Git: $(git --version)"
    fi
    
    # Check available disk space (require at least 2GB)
    available_space=$(df . | tail -1 | awk '{print $4}')
    required_space=2097152  # 2GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        print_warning "Low disk space. At least 2GB recommended."
    fi
    
    print_status "System requirements check completed"
}

# Install system dependencies
install_system_dependencies() {
    print_header "Installing System Dependencies"
    
    case "$OS" in
        "Linux")
            install_linux_dependencies
            ;;
        "Darwin")
            install_macos_dependencies
            ;;
        *)
            print_warning "Unsupported OS: $OS. Skipping system dependencies."
            ;;
    esac
}

# Install Linux dependencies
install_linux_dependencies() {
    print_info "Installing Linux system dependencies..."
    
    # Detect Linux distribution
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
    else
        DISTRO="unknown"
    fi
    
    case "$DISTRO" in
        "ubuntu"|"debian")
            install_ubuntu_debian_deps
            ;;
        "centos"|"rhel"|"fedora")
            install_redhat_deps
            ;;
        *)
            print_warning "Unknown Linux distribution. Install dependencies manually."
            print_info "Required packages: python3-dev, libmagic1, poppler-utils, tesseract-ocr, build-essential"
            ;;
    esac
}

# Install Ubuntu/Debian dependencies
install_ubuntu_debian_deps() {
    print_info "Installing Ubuntu/Debian dependencies..."
    
    sudo apt-get update
    
    # Essential packages
    sudo apt-get install -y \
        python3-dev \
        python3-pip \
        python3-venv \
        build-essential \
        pkg-config \
        libmagic1 \
        libmagic-dev \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        libtesseract-dev \
        libleptonica-dev \
        libpq-dev \
        postgresql-client \
        curl \
        wget \
        unzip \
        git
    
    # Optional: Install additional language packs for tesseract
    sudo apt-get install -y \
        tesseract-ocr-fra \
        tesseract-ocr-deu \
        tesseract-ocr-spa \
        tesseract-ocr-ita || true
    
    print_status "Ubuntu/Debian dependencies installed"
}

# Install Red Hat/CentOS/Fedora dependencies
install_redhat_deps() {
    print_info "Installing Red Hat/CentOS/Fedora dependencies..."
    
    # Use dnf if available (Fedora), otherwise use yum (CentOS/RHEL)
    if command -v dnf &> /dev/null; then
        PKG_MANAGER="dnf"
    else
        PKG_MANAGER="yum"
    fi
    
    sudo $PKG_MANAGER install -y \
        python3-devel \
        python3-pip \
        gcc \
        gcc-c++ \
        make \
        file-devel \
        poppler-utils \
        tesseract \
        tesseract-langpack-eng \
        postgresql-devel \
        postgresql \
        curl \
        wget \
        unzip \
        git
    
    print_status "Red Hat dependencies installed"
}

# Install macOS dependencies
install_macos_dependencies() {
    print_info "Installing macOS dependencies..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        print_info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install dependencies via Homebrew
    brew install \
        python@$PYTHON_VERSION \
        libmagic \
        poppler \
        tesseract \
        tesseract-lang \
        postgresql \
        pkg-config
    
    print_status "macOS dependencies installed"
}

# Clean existing installation
clean_existing_installation() {
    print_header "Cleaning Existing Installation"
    
    # Remove virtual environment
    if [ -d "venv" ]; then
        print_info "Removing existing virtual environment..."
        rm -rf venv
        print_status "Virtual environment removed"
    fi
    
    # Remove cache directories
    print_info "Cleaning cache directories..."
    rm -rf __pycache__ .pytest_cache .mypy_cache .coverage htmlcov
    
    # Remove temporary files
    rm -rf temp_uploads logs/*.log chroma_db faiss_index *.db
    
    print_status "Cleanup completed"
}

# Setup Python virtual environment
setup_python_environment() {
    print_header "Setting Up Python Environment"
    
    # Create virtual environment
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    print_status "Python environment created and activated"
}

# Install Python dependencies
install_python_dependencies() {
    print_header "Installing Python Dependencies"
    
    # Ensure virtual environment is activated
    source venv/bin/activate
    
    # Install main dependencies
    print_info "Installing core dependencies..."
    pip install -r requirements.txt
    
    # Install development dependencies if requested
    if [ "$INSTALL_DEV" = true ]; then
        print_info "Installing development dependencies..."
        
        # Additional dev tools
        pip install \
            jupyter \
            notebook \
            jupyterlab \
            ipykernel \
            pre-commit \
            black \
            flake8 \
            mypy \
            pylint \
            pytest \
            pytest-cov \
            pytest-asyncio \
            pytest-mock
        
        # Setup pre-commit hooks
        if [ -f ".pre-commit-config.yaml" ]; then
            pre-commit install
            print_status "Pre-commit hooks installed"
        fi
    fi
    
    print_status "Python dependencies installed"
}

# Setup project directories
setup_directories() {
    print_header "Setting Up Project Directories"
    
    # Create necessary directories
    directories=(
        "logs"
        "uploads"
        "temp_uploads"
        "data"
        "chroma_db"
        "faiss_index"
        "models"
        "exports"
        "backups"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_info "Created directory: $dir"
        fi
    done
    
    # Set appropriate permissions
    chmod 755 logs uploads temp_uploads data
    chmod 700 chroma_db faiss_index  # More restrictive for databases
    
    print_status "Project directories created"
}

# Setup configuration files
setup_configuration() {
    print_header "Setting Up Configuration"
    
    # Copy environment file if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_info "Created .env from .env.example"
        else
            create_default_env_file
        fi
    else
        print_info ".env file already exists"
    fi
    
    # Create config.yaml if it doesn't exist
    if [ ! -f "config.yaml" ]; then
        create_default_config_file
    else
        print_info "config.yaml already exists"
    fi
    
    # Create logging configuration
    create_logging_config
    
    print_status "Configuration files created"
}

# Create default .env file
create_default_env_file() {
    print_info "Creating default .env file..."
    
    cat > .env << 'EOF'
# Document Processor Agent - Environment Configuration

# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true
WORKERS=1

# Database Configuration
DATABASE_URL=sqlite:///./documents.db
# For PostgreSQL: DATABASE_URL=postgresql://user:password@localhost:5432/document_processor

# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_TEMPERATURE=0.1

# Anthropic Configuration (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Google AI Configuration (optional)
GOOGLE_API_KEY=your_google_api_key_here

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Vector Store Configuration
VECTOR_STORE_PROVIDER=chroma
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=documents

# Document Processing Configuration
MAX_FILE_SIZE_MB=100
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TEMP_DIR=./temp_uploads

# Validation Configuration
VALIDATION_THRESHOLD=0.7
HUMAN_REVIEW_THRESHOLD=0.5

# Web Search Configuration (optional)
SEARCH_PROVIDER=tavily
SEARCH_API_KEY=your_search_api_key_here

# Security Configuration
SECRET_KEY=your_secret_key_here_change_this_in_production

# Development Configuration
TESTING=false
DEBUG=false
LOG_LEVEL=INFO

# Optional: Redis for caching
# REDIS_URL=redis://localhost:6379/0

# Optional: External services
# WEBHOOK_URL=https://your-webhook-url.com/notify
EOF
    
    print_status "Default .env file created"
    print_warning "Please update the API keys in .env file before running the application"
}

# Create default config.yaml
create_default_config_file() {
    print_info "Creating default config.yaml..."
    
    cat > config.yaml << 'EOF'
# Document Processor Agent Configuration

# Application Settings
app:
  name: "Document Processor Agent"
  version: "1.0.0"
  description: "AI-powered document processing system"

# Server Configuration
server:
  host: "${HOST:0.0.0.0}"
  port: "${PORT:8000}"
  reload: "${RELOAD:false}"
  workers: "${WORKERS:1}"

# Database Configuration
database:
  url: "${DATABASE_URL:sqlite:///./documents.db}"
  pool_size: 10
  echo: false

# LLM Configuration
llm:
  primary_provider: "${LLM_PROVIDER:openai}"
  models:
    openai:
      model: "${OPENAI_MODEL:gpt-4-turbo-preview}"
      api_key: "${OPENAI_API_KEY}"
      temperature: 0.1
    anthropic:
      model: "${ANTHROPIC_MODEL:claude-3-sonnet-20240229}"
      api_key: "${ANTHROPIC_API_KEY}"
      temperature: 0.1

# Embedding Configuration
embeddings:
  embedding_provider: "${EMBEDDING_PROVIDER:openai}"
  embedding_model: "${EMBEDDING_MODEL:text-embedding-3-small}"
  embedding_dimensions: 1536
  chunk_size: 1000
  chunk_overlap: 200
  chunk_strategy: "recursive"

# Vector Store Configuration
vector_store:
  provider: "${VECTOR_STORE_PROVIDER:chroma}"
  persist_directory: "${CHROMA_PERSIST_DIR:./chroma_db}"
  collection_name: "${CHROMA_COLLECTION:documents}"

# Agent Configuration
agents:
  document_loader:
    supported_formats: [".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".csv", ".xlsx", ".pptx"]
    max_file_size_mb: 100
    temp_dir: "./temp_uploads"
    extract_metadata: true
    use_ocr: false
    clean_text: true
  
  parser:
    extract_metadata: true
    detect_language: true
    extract_entities: true
    extract_keywords: true
    max_processing_time: 300
  
  embedder:
    batch_size: 50
    max_retries: 3
    normalize_embeddings: true
  
  qa_agent:
    max_context_length: 8000
    max_sources: 5
    min_confidence_threshold: 0.1
    include_reasoning: false
  
  validator:
    confidence_threshold: 0.7
    human_review_threshold: 0.5
    max_retries: 3
  
  rag_enricher:
    web_search_enabled: false
    max_web_results: 5
    search_timeout: 10
    trusted_domains: ["wikipedia.org", "britannica.com"]

# Web Search Configuration
web_search:
  provider: "${SEARCH_PROVIDER:tavily}"
  api_key: "${SEARCH_API_KEY}"

# Workflow Configuration
workflows:
  document_processing:
    max_iterations: 10
    interrupts: []
    checkpointing: true

# Monitoring Configuration
monitoring:
  enable_metrics: true
  metrics_port: 9090
  log_level: "${LOG_LEVEL:INFO}"

# Security Configuration
security:
  secret_key: "${SECRET_KEY}"
  enable_auth: false
  cors_origins: ["*"]
EOF
    
    print_status "Default config.yaml created"
}

# Create logging configuration
create_logging_config() {
    print_info "Creating logging configuration..."
    
    cat > logging.yaml << 'EOF'
version: 1
disable_existing_loggers: false

formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
  
  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: logs/error.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  agents:
    level: DEBUG
    handlers: [console, file]
    propagate: false
  
  langgraph_flows:
    level: DEBUG
    handlers: [console, file]
    propagate: false
  
  sqlstore:
    level: INFO
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console, file, error_file]
EOF
    
    print_status "Logging configuration created"
}

# Install NLP models
install_nlp_models() {
    print_header "Installing NLP Models"
    
    # Ensure virtual environment is activated
    source venv/bin/activate
    
    # Download spaCy models
    print_info "Downloading spaCy models..."
    
    # Try to download English model
    python -m spacy download en_core_web_sm || print_warning "Failed to download en_core_web_sm"
    
    # Download NLTK data
    print_info "Downloading NLTK data..."
    python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Warning: Failed to download NLTK data: {e}')
" || print_warning "Failed to download NLTK data"
    
    print_status "NLP models installation completed"
}

# Setup database
setup_database() {
    print_header "Setting Up Database"
    
    # Ensure virtual environment is activated
    source venv/bin/activate
    
    # Initialize database
    print_info "Initializing database..."
    
    python -c "
import asyncio
import sys
import os
sys.path.append('.')

async def init_db():
    try:
        from sqlstore.database import DatabaseManager
        config = {'url': os.getenv('DATABASE_URL', 'sqlite:///./documents.db')}
        db_manager = DatabaseManager(config)
        await db_manager.initialize()
        print('Database initialized successfully')
    except Exception as e:
        print(f'Warning: Database initialization failed: {e}')

asyncio.run(init_db())
" || print_warning "Database initialization failed"
    
    print_status "Database setup completed"
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    # Ensure virtual environment is activated
    source venv/bin/activate
    
    # Test imports
    print_info "Testing Python imports..."
    
    python -c "
import sys
import pkg_resources

# Test critical imports
try:
    import fastapi
    print('✓ FastAPI import successful')
except ImportError as e:
    print(f'✗ FastAPI import failed: {e}')

try:
    import streamlit
    print('✓ Streamlit import successful')
except ImportError as e:
    print(f'✗ Streamlit import failed: {e}')

try:
    import langchain
    print('✓ LangChain import successful')
except ImportError as e:
    print(f'✗ LangChain import failed: {e}')

try:
    import chromadb
    print('✓ ChromaDB import successful')
except ImportError as e:
    print(f'✗ ChromaDB import failed: {e}')

try:
    import sqlalchemy
    print('✓ SQLAlchemy import successful')
except ImportError as e:
    print(f'✗ SQLAlchemy import failed: {e}')

# Test custom modules
try:
    from agents.base_agent import BaseAgent
    print('✓ Base agent import successful')
except ImportError as e:
    print(f'✗ Base agent import failed: {e}')

print()
print('Python version:', sys.version)
print('Virtual environment:', hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
"
    
    # Test configuration
    print_info "Testing configuration..."
    if [ -f ".env" ] && [ -f "config.yaml" ]; then
        print_status "Configuration files present"
    else
        print_warning "Some configuration files missing"
    fi
    
    # Test directories
    print_info "Testing directories..."
    required_dirs=("logs" "uploads" "temp_uploads" "data")
    all_dirs_exist=true
    
    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ]; then
            print_status "Directory exists: $dir"
        else
            print_warning "Directory missing: $dir"
            all_dirs_exist=false
        fi
    done
    
    if [ "$all_dirs_exist" = true ]; then
        print_status "All required directories present"
    fi
    
    print_status "Installation verification completed"
}

# Cleanup function for script interruption
cleanup() {
    print_info "Setup interrupted. Cleaning up..."
    exit 1
}

# Set trap for cleanup
trap cleanup INT TERM

# Run main function
main "$@"