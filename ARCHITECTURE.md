# arXiv Parsing System Architecture

## System Overview

The arXiv parsing system is designed as a modular, async-first pipeline for extracting structured references from arXiv papers with intelligent parser selection and Google Search API integration.

## Core Design Principles

1. **HTML-First Strategy**: Prioritize arXiv HTML parsing over PDF processing
2. **Multi-Parser Robustness**: Combine GROBID, AnyStyle, and custom parsers
3. **Intelligent Selection**: Use ML-based meta-learning for optimal parser choice
4. **Async-Native**: Full async/await support for scalable processing
5. **Error Resilience**: Graceful degradation with comprehensive fallback strategies
6. **API Integration**: Seamless Google Search API for reference validation

## High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Layer   │ -> │  Processing Core │ -> │  Output Layer   │
│                 │    │                  │    │                 │
│ • arXiv URLs    │    │ • HTML Parser    │    │ • CSL-JSON      │
│ • PDF Files     │    │ • PDF Fallback   │    │ • BibTeX        │
│ • HTML Content  │    │ • Multi-Parsers  │    │ • Database      │
│                 │    │ • Meta-Learning  │    │ • Search Index  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                               v
                    ┌──────────────────┐
                    │ External Services│
                    │                  │
                    │ • Google Search  │
                    │ • GROBID Server  │
                    │ • arXiv API      │
                    └──────────────────┘
```

## Component Architecture

### 1. Ingestion Layer (`src/ingestion/`)

**ArXivFetcher** (`arxiv_fetcher.py`)
- Async HTTP client for arXiv content
- Smart detection of HTML vs PDF availability
- Rate limiting and retry logic
- Metadata extraction from arXiv API

**HTMLProcessor** (`html_processor.py`)
- BeautifulSoup-based HTML parsing
- Reference section extraction
- Content structure analysis
- Fallback to PDF when HTML unavailable

**PDFProcessor** (`pdf_processor.py`)
- PDF text extraction using pdftotext/PyMuPDF
- Reference section detection heuristics
- Integration with GROBID for structured parsing

### 2. Parser Layer (`src/parsers/`)

**GrobidClient** (`grobid_client.py`)
- Async GROBID server communication
- TEI XML to structured data conversion
- Error handling and service health monitoring
- Batch processing capabilities

**AnystyleClient** (`anystyle_client.py`)
- Ruby subprocess integration
- JSON output processing
- Performance monitoring and caching
- Graceful degradation on failures

**HTMLReferenceExtractor** (`html_extractor.py`)
- Native HTML reference parsing
- arXiv-specific citation pattern recognition
- Direct extraction without external dependencies
- Fast processing for simple cases

**ParserOrchestrator** (`orchestrator.py`)
- Coordinate multiple parsers
- Intelligent parser selection
- Result aggregation and comparison
- Performance monitoring

### 3. Normalization Layer (`src/normalize/`)

**SchemaMapper** (`schema_mapper.py`)
- Convert parser outputs to CSL-JSON
- Handle multiple input formats (TEI, JSON, raw text)
- Field mapping and validation
- Provenance tracking

**DataValidator** (`validator.py`)
- Schema compliance checking
- Data quality metrics
- Duplicate detection
- Consistency validation

### 4. Meta-Learning Layer (`src/meta/`)

**FeatureExtractor** (`features.py`)
- String-based features (length, punctuation, digits)
- Regex-based pattern detection (DOI, arXiv ID, year)
- Parser metadata (confidence, processing time)
- Cross-parser agreement metrics

**ReferenceSelector** (`ref_selector.py`)
- Reference-level parser selection
- Scikit-learn Random Forest/LightGBM models
- Confidence scoring and uncertainty quantification
- Active learning integration

**FieldSelector** (`field_selector.py`)
- Field-level parser selection
- Per-field specialized models
- Ensemble prediction strategies
- Calibrated probability outputs

### 5. Search Integration Layer (`src/search/`)

**GoogleSearchClient** (`google_client.py`)
- Google Custom Search API integration
- Academic source prioritization
- Rate limiting and quota management
- Result ranking and relevance scoring

**CitationValidator** (`citation_validator.py`)
- Cross-reference parsed citations with search results
- DOI resolution and validation
- arXiv ID verification
- Author and title matching

**RelatedWorkFinder** (`related_finder.py`)
- Discover related papers via search
- Topic-based and author-based queries
- Citation network analysis
- Recommendation scoring

### 6. Storage Layer (`src/storage/`)

**CacheManager** (`cache.py`)
- Redis-based caching for expensive operations
- TTL management for different data types
- Cache invalidation strategies
- Performance monitoring

**DatabaseManager** (`database.py`)
- SQLite/PostgreSQL for structured data storage
- Paper metadata and reference storage
- Search result caching
- Query optimization

### 7. Evaluation Layer (`src/eval/`)

**MetricsCalculator** (`metrics.py`)
- Field-level F1 scores
- Exact match accuracy
- DOI/arXiv ID recall rates
- Parser performance benchmarks

**BenchmarkRunner** (`benchmark.py`)
- Automated evaluation pipelines
- Cross-validation frameworks
- Performance regression testing
- Comparative analysis tools

## Data Flow Architecture

### Primary Processing Pipeline

```
arXiv URL/ID
    │
    v
┌───────────────┐
│ ArXivFetcher  │ -> HTML/PDF content + metadata
└───────────────┘
    │
    v
┌───────────────┐
│ HTMLProcessor │ -> Structured content sections
└───────────────┘
    │
    v
┌───────────────┐
│ ParserOrch.   │ -> Multi-parser results
└───────────────┘
    │
    v
┌───────────────┐
│ MetaSelector  │ -> Optimal parser selection
└───────────────┘
    │
    v
┌───────────────┐
│ Normalizer    │ -> CSL-JSON format
└───────────────┘
    │
    v
┌───────────────┐
│ SearchIntegr. │ -> Enhanced with search results
└───────────────┘
    │
    v
Output (JSON/BibTeX/Database)
```

### Search Enhancement Pipeline

```
Parsed References
    │
    v
┌─────────────────┐
│ Citation Queries│ -> Google Search API
└─────────────────┘
    │
    v
┌─────────────────┐
│ Result Validation│ -> Verified citations
└─────────────────┘
    │
    v
┌─────────────────┐
│ Related Discovery│ -> Additional references
└─────────────────┘
```

## Technology Stack

### Core Technologies
- **Python 3.11+**: Primary language with async/await support
- **aiohttp**: Async HTTP client for API calls
- **BeautifulSoup4**: HTML parsing and extraction
- **PyMuPDF/pdftotext**: PDF text extraction
- **scikit-learn**: Machine learning for meta-selection
- **pandas**: Data manipulation and analysis

### External Services
- **GROBID**: Reference extraction from PDFs
- **AnyStyle**: Ruby-based citation parsing
- **Google Custom Search API**: Citation validation and discovery
- **arXiv API**: Paper metadata retrieval

### Storage & Caching
- **SQLite**: Local development database
- **PostgreSQL**: Production database (optional)
- **Redis**: Caching layer for performance
- **JSON/BibTeX**: Output formats

## Configuration Architecture

### Environment Configuration
```python
# config/settings.py
class Settings:
    # API Configuration
    google_search_api_key: str
    google_search_engine_id: str
    arxiv_rate_limit_delay: float = 3.1
    
    # Service Configuration
    grobid_url: str = "http://localhost:8070"
    anystyle_timeout: int = 30
    
    # Processing Configuration
    enable_pdf_fallback: bool = True
    max_concurrent_requests: int = 5
    cache_ttl_hours: int = 24
    
    # ML Configuration
    model_confidence_threshold: float = 0.7
    enable_active_learning: bool = False
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- ✅ Basic project structure
- ✅ ArXiv fetching and HTML parsing
- ✅ GROBID and AnyStyle integration
- ✅ Basic normalization pipeline

### Phase 2: Intelligence Layer (Week 2)
- 🔄 Feature extraction framework
- 🔄 Meta-learning model training
- 🔄 Parser selection optimization
- 🔄 Performance benchmarking

### Phase 3: Search Integration (Week 3)
- ⏳ Google Search API integration
- ⏳ Citation validation pipeline
- ⏳ Related work discovery
- ⏳ Result ranking algorithms

### Phase 4: Production Features (Week 4)
- ⏳ Comprehensive error handling
- ⏳ Performance optimization
- ⏳ Monitoring and observability
- ⏳ Documentation and testing

## Quality Assurance Strategy

### Testing Framework
- **Unit Tests**: Individual component testing with mock data
- **Integration Tests**: End-to-end pipeline testing with real arXiv papers
- **Performance Tests**: Latency and throughput benchmarking
- **API Tests**: External service integration testing

### Monitoring & Observability
- **Performance Metrics**: Processing time, success rates, error rates
- **Quality Metrics**: Parser accuracy, field-level F1 scores
- **System Metrics**: Memory usage, API quota consumption
- **Business Metrics**: Papers processed, references extracted

This architecture provides a robust, scalable foundation for the arXiv parsing system while maintaining flexibility for future enhancements and optimizations.