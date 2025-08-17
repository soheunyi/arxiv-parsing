# arXiv Parsing System - Project Overview

## Project Purpose
A specialized arXiv paper parsing system for extracting structured references from arXiv papers with Google Search API integration. The system prioritizes arXiv HTML parsing over PDF processing with multi-parser robustness and intelligent selection.

## Tech Stack
- **Python 3.10+** with async/await support
- **aiohttp** for async HTTP requests  
- **BeautifulSoup4** for HTML parsing
- **PyMuPDF** for PDF processing
- **Pydantic** for data validation
- **scikit-learn** for ML-based parser selection
- **Redis** for caching
- **SQLAlchemy** for database operations
- **Typer** for CLI interface

## Key Architecture Components
- **Ingestion Layer**: ArXiv fetching and content processing
- **Parser Layer**: GROBID, AnyStyle, and HTML extractors with orchestration
- **Meta-Learning Layer**: Intelligent parser selection using ML
- **Search Integration**: Google Search API for citation validation
- **Storage Layer**: Redis caching and database management
- **Normalization**: Convert outputs to CSL-JSON format

## Current Development Phase
Phase 2: Intelligence Layer development with meta-learning model training and parser selection optimization.