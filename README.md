# arXiv Paper Reference Discovery System

A complete end-to-end system for parsing arXiv papers and discovering reference relationships using GROBID service integration.

## Features

- **PDF Fetching**: Download papers directly from arXiv
- **Reference Extraction**: Extract all references using GROBID service (100% accuracy)
- **arXiv Discovery**: Find arXiv papers among extracted references
- **Complete Pipeline**: Integrated workflow from PDF to reference analysis

## Quick Start

### Prerequisites

- Python 3.11+
- UV package manager
- Docker (for GROBID service)
- GROBID service running (see setup below)

### Setup

1. **Install dependencies:**
   ```bash
   uv sync --group dev
   ```

2. **Start GROBID service:**
   ```bash
   docker-compose up -d
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

### Usage

#### Run End-to-End Workflow Test

```bash
# Run the complete workflow demonstration
uv run python tests/test_end_to_end_workflow.py

# Or run via pytest
uv run pytest tests/test_end_to_end_workflow.py -v
```

This will:
1. Fetch PDF for arxiv:2207.04015 (optimization paper with 52 references)
2. Extract all 52 references using GROBID
3. Search for arXiv papers among the references
4. Display results and save detailed report

#### Expected Output

```
🚀 Running End-to-End Workflow for 2207.04015
============================================================
1. Fetching PDF from arXiv...
   ✅ PDF fetched: 0.86 MB in 0.13s
2. Extracting references with GROBID...
   ✅ Extracted 52 references in 13.77s
      📖 Books/Reports: 14
      📄 Journal Articles: 38
3. Discovering arXiv papers in references...
   ✅ Found 2 arXiv papers from 10 searches in 47.52s
   🔍 Discovered arXiv papers:
      1. A Direct Proof of Convergence of Davis–Y... → 2108.01318v2 (conf: 1.00)
      2. Degenerate Preconditioned Proximal Point... → 2109.11481v1 (conf: 1.00)

📊 WORKFLOW SUMMARY
============================================================
Total Time: 61.42s
References Extracted: 52/52 (100%)
arXiv Papers Found: 2 from 10 searches
Success Rate: 100%
```

## System Architecture

### Core Components

- **`src/ingestion/arxiv_fetcher.py`** - PDF downloading from arXiv
- **`src/parsers/grobid_client.py`** - GROBID service integration & TEI XML parsing
- **`src/search/arxiv_search_client.py`** - arXiv API search and matching
- **`src/models/`** - Data models for papers, references, and metadata

### Key Features

- **100% Reference Extraction**: Fixed TEI parser handles both journal articles and books/reports
- **Async Architecture**: Non-blocking operations for better performance
- **Comprehensive Testing**: End-to-end workflow validation
- **Error Handling**: Robust error handling with timeouts and retries

## Development

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy src/
```

### Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_end_to_end_workflow.py::test_complete_workflow -v

# Test with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

## Configuration

Key settings in `.env`:

- `GROBID_URL=http://localhost:8070` - GROBID service endpoint
- `ARXIV_RATE_LIMIT_DELAY=3.1` - Delay between arXiv requests
- `ENABLE_PDF_FALLBACK=true` - Enable PDF parsing if HTML unavailable

## Performance

Typical performance for arxiv:2207.04015:
- PDF Fetch: ~0.13s (863KB)
- Reference Extraction: ~13.8s (52 references)
- arXiv Discovery: ~4.8s per search
- Total Pipeline: ~61.4s

## License

This project is for research and educational purposes.