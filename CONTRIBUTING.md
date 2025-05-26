# Contributing to Document Processor Agent

Thank you for your interest in contributing to the Document Processor Agent! This guide will help you get started with contributing to this project.

## 🎯 Project Overview

The Document Processor Agent is an advanced AI-powered system that processes documents using multiple agents, RAG (Retrieval-Augmented Generation), and human-in-the-loop validation. The system is built with:

- **Python 3.11+** for the core application
- **LangChain & LangGraph** for AI workflows
- **Multiple LLM providers** (OpenAI, Anthropic, Google)
- **Vector databases** (ChromaDB, FAISS, OpenSearch)
- **PostgreSQL/SQLite** for metadata storage
- **AWS** for cloud deployment

## 🛠️ Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (optional, for containerized development)
- PostgreSQL (optional, for local database)

### Quick Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/document-processor-agent.git
   cd document-processor-agent
   ```

2. **Run the automated setup**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

4. **Activate virtual environment**
   ```bash
   source venv/bin/activate
   ```

5. **Verify installation**
   ```bash
   python -m pytest tests/ -v
   ```

### Manual Setup

If you prefer manual setup:

1. **Create virtual environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Initialize database**
   ```bash
   python -c "from sqlstore.database import initialize_database; import asyncio; asyncio.run(initialize_database())"
   ```

## 📋 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug fixes**
- ✨ **New features**
- 📚 **Documentation improvements**
- 🧪 **Tests**
- 🔧 **Performance optimizations**
- 🎨 **UI/UX improvements**
- 🌐 **Translations** (future)

### Contribution Process

1. **Check existing issues** or create a new one to discuss your idea
2. **Fork the repository** and create a feature branch
3. **Make your changes** following our coding standards
4. **Write tests** for your changes
5. **Update documentation** if needed
6. **Submit a pull request**

## 🏗️ Architecture Guide

### Core Components

```
agents/               # AI agents for different tasks
├── base_agent.py    # Base class for all agents
├── document_loader.py # Multi-format document loading
├── parser.py        # AI-powered content parsing
├── embedder.py      # Vector embedding creation
├── qa_agent.py      # Question answering
└── validator.py     # Quality validation

langgraph_flows/     # Workflow orchestration
├── document_graph.py # Main processing pipeline
├── reflexion_graph.py # Human-in-loop workflows
└── graph_utils.py   # Shared utilities

sqlstore/           # Database layer
├── models.py       # SQLAlchemy models
├── database.py     # Connection management
└── migrations/     # Database migrations

web_ui/            # User interfaces
├── streamlit_app.py # Main web interface
└── gradio_app.py   # Alternative interface
```

### Adding New Agents

When creating a new agent, follow this pattern:

```python
from agents.base_agent import BaseAgent, AgentResult

class MyNewAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Initialize your agent-specific settings
        
    async def process(self, input_data: Any, **kwargs) -> AgentResult:
        try:
            # Your agent logic here
            result_data = {"processed": "data"}
            
            return AgentResult(
                success=True,
                data=result_data,
                confidence=0.9
            )
        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e)
            )
```

### Extending LangGraph Workflows

To add new workflow nodes:

```python
# In your graph file
workflow.add_node("my_new_step", self._my_new_step_function)
workflow.add_edge("existing_step", "my_new_step")

async def _my_new_step_function(self, state: Dict) -> Dict:
    # Process the state
    state["new_data"] = "processed"
    return state
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_agents.py        # Unit tests for agents
├── test_flows.py         # Workflow tests
├── test_integration.py   # End-to-end tests
└── test_performance.py   # Performance benchmarks
```

### Writing Tests

1. **Unit Tests**: Test individual agent functions
   ```python
   @pytest.mark.asyncio
   async def test_my_agent_function():
       agent = MyAgent()
       result = await agent.process("test input")
       assert result.success is True
   ```

2. **Integration Tests**: Test complete workflows
   ```python
   async def test_document_processing_pipeline():
       # Test the entire pipeline
       pass
   ```

3. **Mock External Services**: Use mocks for LLM calls and APIs
   ```python
   @patch('agents.qa_agent.QAAgent.get_llm_client')
   async def test_qa_with_mock(mock_llm):
       # Test with mocked LLM responses
       pass
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agents.py -v

# Run with coverage
pytest --cov=agents --cov-report=html

# Run integration tests only
pytest tests/test_integration.py
```

## 📝 Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (instead of 79)
- **Imports**: Use absolute imports, group by standard/third-party/local
- **Type hints**: Required for all public functions
- **Docstrings**: Google style docstrings

### Code Formatting

We use several tools for code quality:

```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy agents/ langgraph_flows/

# Security scanning
bandit -r agents/
```

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
```

### Documentation Standards

- **README files**: Keep updated for each major component
- **Code comments**: Explain complex logic, not obvious code
- **API documentation**: Auto-generated from docstrings
- **Architecture docs**: Update when adding new components

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment information**
   - Python version
   - Operating system
   - Package versions (`pip list`)

2. **Steps to reproduce**
   - Minimal code example
   - Input data (if possible)
   - Expected vs actual behavior

3. **Error logs**
   - Full error traceback
   - Relevant log entries

### Bug Report Template

```markdown
## Bug Description
Brief description of the bug

## Environment
- Python version: 3.11.x
- OS: Ubuntu 22.04
- Package version: v1.2.3

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Error Logs
```
Error traceback here
```

## Additional Context
Any other relevant information
```

## ✨ Feature Requests

For new features, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the problem** the feature would solve
3. **Propose a solution** with implementation details
4. **Consider alternatives** and their trade-offs

### Feature Request Template

```markdown
## Feature Description
Clear description of the requested feature

## Problem Statement
What problem does this solve?

## Proposed Solution
How should this feature work?

## Alternatives Considered
What other approaches were considered?

## Implementation Notes
Technical details, if any

## Additional Context
Screenshots, mockups, or examples
```

## 🔄 Pull Request Process

### Before Submitting

1. **Create an issue** to discuss major changes
2. **Fork the repository** and create a feature branch
3. **Follow naming conventions**: `feature/add-new-agent` or `fix/parser-bug`
4. **Write tests** for your changes
5. **Update documentation** as needed
6. **Run the full test suite**

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (specify)

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or noted)

## Related Issues
Fixes #123
Related to #456
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by maintainers
3. **Testing** in development environment
4. **Approval** and merge

## 🚀 Release Process

### Versioning

We use **Semantic Versioning** (semver):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Workflow

1. **Development** → `develop` branch
2. **Testing** → Create release candidate
3. **Production** → Tag and release to `main`

## 👥 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Slack**: Real-time communication (link in README)
- **Email**: security@your-domain.com for security issues

### Code of Conduct

Please note that this project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md):

- **Be respectful** and inclusive
- **Be constructive** in feedback
- **Focus on the project** goals
- **Help others** learn and contribute

### Recognition

Contributors are recognized in:

- **README**: Major contributors listed
- **Releases**: Contributors mentioned in release notes
- **All Contributors**: Bot to track all contributions

## 📚 Additional Resources

### Learning Resources

- **LangChain Documentation**: https://docs.langchain.com
- **LangGraph Guide**: https://langchain-ai.github.io/langgraph/
- **Vector Databases**: ChromaDB, FAISS documentation
- **FastAPI**: https://fastapi.tiangolo.com

### Development Tools

- **IDE Setup**: VS Code configuration in `.vscode/`
- **Docker**: Development containers available
- **Database**: PostgreSQL setup scripts
- **AWS**: CloudFormation templates provided

### Common Issues

**Q: Tests are failing with API rate limits**
A: Use the mock clients for testing, set `TESTING=true` in environment

**Q: Database connection errors**
A: Check your `.env` configuration and ensure database is running

**Q: Import errors with agents**
A: Make sure you're in the virtual environment and PYTHONPATH is set

**Q: Docker build fails**
A: Check Docker version and available disk space

## 🙏 Thank You

Thank you for contributing to the Document Processor Agent! Your contributions help make this project better for everyone.

For questions about contributing, please:
1. Check this guide first
2. Search existing issues
3. Create a new issue with the "question" label
4. Join our community discussions

Happy coding! 🚀