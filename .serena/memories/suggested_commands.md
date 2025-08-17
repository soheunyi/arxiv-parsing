# Suggested Commands for arXiv Parsing System

## Setup and Environment
```bash
# Install dependencies
uv sync --group dev

# Install with search API support
uv sync --group dev --group search
```

## Development Workflow
```bash
# Run main parsing system
uv run python main.py

# Test HTML parsing functionality  
uv run python scripts/test_parsing.py

# Test Google Search API integration
uv run python scripts/test_search_api.py
```

## Code Quality
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy src/
```

## Testing
```bash
# All tests with verbose output
uv run pytest tests/ -v

# Unit tests only
uv run pytest tests/unit/ -v

# Integration tests only  
uv run pytest tests/integration/ -v

# Single test file
uv run pytest <path/to/test_file.py> -v

# Coverage report
uv run pytest tests/ --cov=src --cov-report=html
```

## Project Management
```bash
# CLI interface
uv run arxiv-parse --help

# Package management
uv add <package>              # Production dependency
uv add --group dev <package>  # Development dependency
uv add --group search <package>  # Search-related dependency
```