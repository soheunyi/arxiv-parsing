# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ Critical Rules

**NEVER modify test code without explicit permission**
**NEVER change API names and parameters**
**NEVER migrate data arbitrarily**

## Project Objective

This project focuses on **parsing arXiv papers from HTML content and extracting references** with **Google Search API integration** for enhanced paper discovery. This is a specialized parsing system separated from recommendation functionality due to its complexity and importance.

## Essential Commands

### Python Package Management
- **Use `uv` for Python execution and dependency management**
- Install dependencies: `uv sync` (production), `uv sync --group dev` (development)
- Run Python scripts: `uv run python <script>` or `uv run <command>`
- Add dependencies: `uv add <package>` (production), `uv add --group dev <package>` (dev)

### Development Workflow
```bash
# Setup (first time)
uv sync --group dev

# Core development
uv run python main.py                         # Run parsing system
uv run python scripts/test_parsing.py         # Test HTML parsing functionality
uv run python scripts/test_search_api.py      # Test Google Search API integration

# Code quality
uv run ruff format .                          # Format code
uv run ruff check .                           # Lint code
uv run mypy src/                             # Type checking

# Testing
uv run pytest tests/ -v                      # All tests with verbose output
uv run pytest tests/unit/ -v                 # Unit tests only
uv run pytest tests/integration/ -v          # Integration tests only
uv run pytest <path/to/test_file.py> -v      # Single test file
uv run pytest tests/ --cov=src --cov-report=html  # Coverage report
```

## High-Level Architecture

### Core System Design
This is a **specialized arXiv paper parsing system** that extracts structured information from arXiv HTML pages and integrates with Google Search API for enhanced reference discovery and validation.

### Key Components (Planned)

#### HTML Parsing Engine
- **ArXiv HTML Parser**: Extract paper metadata, content, and references from arXiv HTML format
- **Reference Extractor**: Parse citation information from references sections
- **Content Analyzer**: Extract abstract, introduction, methodology sections
- **Metadata Processor**: Handle paper titles, authors, submission dates, categories

#### Google Search API Integration
- **Search Service**: Query Google Search API for related papers and citations
- **Result Validator**: Verify and cross-reference found papers with arXiv data
- **Citation Discovery**: Find external citations and references to arXiv papers
- **Related Work Finder**: Discover related papers not cited in original references

#### Data Processing Pipeline
- **HTML Fetcher**: Async retrieval of arXiv HTML content with rate limiting
- **Content Parser**: BeautifulSoup-based parsing with error handling
- **Reference Resolver**: Match references to actual papers (arXiv IDs, DOIs, URLs)
- **Search Coordinator**: Orchestrate Google Search API queries based on parsed content

### Key Data Flows
1. **Paper Input**: arXiv URL/ID → HTML fetching → Content parsing → Structured data
2. **Reference Extraction**: HTML content → Reference section parsing → Citation list → Paper metadata
3. **Search Enhancement**: Parsed content → Google Search queries → Related papers → Citation validation
4. **Output Generation**: Structured paper data + references + search results → JSON/database storage

## ArXiv HTML Parsing Specifics

### ArXiv HTML Format (2023+)
- ArXiv began providing HTML versions of papers in December 2023
- Not all papers have HTML versions (mainly TeX/LaTeX submissions)
- HTML format provides better accessibility and parsing capabilities
- References section typically well-structured in HTML format

### Parsing Strategies
- **Primary**: Use arXiv HTML format when available (`https://arxiv.org/html/{paper_id}`)
- **Fallback**: Parse PDF content when HTML unavailable
- **Reference Section**: Look for specific HTML tags and patterns used by arXiv
- **Citation Patterns**: Handle various citation formats (academic, arXiv-specific, DOI links)

## Google Search API Integration

### Search API Configuration
- Use Google Custom Search API for academic paper searches
- Configure search to prioritize academic sources (arXiv, ACM, IEEE, etc.)
- Implement rate limiting to respect API quotas
- Handle search result ranking and relevance scoring

### Search Strategies
- **Title-based Search**: Search by exact paper titles for citation verification
- **Author-based Search**: Find related papers by the same authors
- **Topic-based Search**: Discover related work using keywords and abstracts
- **Citation Validation**: Verify if papers cite each other through search results

## Common Development Patterns

### Async/Await Usage
All HTTP requests and file I/O operations use async patterns:
- HTML fetching from arXiv (`aiohttp`)
- Google Search API calls (`aiohttp`)
- File operations for caching and storage
- Database operations if implemented

### Error Handling
- Graceful handling of missing HTML versions (fallback to PDF)
- Rate limiting for both arXiv and Google Search API
- Robust parsing with malformed HTML content
- Network timeouts and retry logic with exponential backoff

### Parsing Patterns
- Use BeautifulSoup4 for HTML parsing with error recovery
- Implement CSS selector-based extraction for specific content sections
- Handle multiple citation formats and edge cases
- Validate extracted data before further processing

## Testing Philosophy

### Comprehensive Test Coverage
1. **Unit Tests**: Test individual parsing functions with sample HTML
2. **Integration Tests**: Test complete workflows with real arXiv papers
3. **API Tests**: Mock and real Google Search API integration tests
4. **Edge Case Tests**: Handle malformed HTML, missing sections, API failures
5. **Performance Tests**: Ensure parsing speed and memory efficiency

### Test Data Management
- Use real arXiv paper samples for integration testing
- Mock Google Search API responses for unit testing
- Test with papers from different arXiv categories and time periods
- Include papers with various reference formats and structures

## Configuration Management

### Environment Setup
- Copy `.env.example` to `.env`
- Configure Google Search API: `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID`
- Set arXiv rate limiting: `ARXIV_RATE_LIMIT_DELAY=3.1` (seconds)
- Configure parsing options: `ENABLE_PDF_FALLBACK=true`

### API Keys and Rate Limits
- Google Search API: 100 free queries per day, paid plans available
- arXiv API: 3-second delay between requests (no API key required)
- Implement intelligent caching to minimize API usage
- Monitor usage and implement budget controls

This system prioritizes accurate parsing, robust error handling, and efficient API usage while maintaining high-quality data extraction from arXiv papers and search enhancement through Google's search capabilities.


# Tests

- When writing tests, you should use the `pytest` framework with `uv`
- Make sure that the tests are comprehensive and be educational.

# Misc (but IMPORTANT)
- **Search documentation for open source projects if possible.**
- **Think CRITICAL when building project structure / designing API / writing tests.**
- **If you are not sure about something, ask me.**
- **Actively use serena MCP to reduce token usage.**