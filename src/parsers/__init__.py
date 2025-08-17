"""
Parser modules for extracting references from academic papers.

This package provides various parsers including GROBID, HTML, AnyStyle,
and the new fast PyMuPDF4LLM parser.
"""

from .base_parser import BaseParser, ParseRequest, ParseResult
from .grobid_client import GrobidClient
from .arxiv_html_parser import ArxivHtmlParser
from .pymupdf_parser import PyMuPDFParser
from .orchestrator import ParserOrchestrator

__all__ = [
    "BaseParser",
    "ParseRequest", 
    "ParseResult",
    "GrobidClient",
    "ArxivHtmlParser",
    "PyMuPDFParser",
    "ParserOrchestrator",
]