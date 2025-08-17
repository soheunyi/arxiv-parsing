"""
Parser orchestrator with intelligent fallback strategy.

This module provides smart parser selection and coordination between
different parsing strategies, including the new fast PyMuPDF4LLM parser.
"""

import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path

from .base_parser import BaseParser, ParseRequest, ParseResult
from .pymupdf_parser import PyMuPDFParser
from .grobid_client import GrobidClient
from .arxiv_html_parser import ArxivHtmlParser
from ..models.schemas import ParserType
from ..config.logging import get_logger


class ParserOrchestrator:
    """
    Orchestrates multiple parsers with intelligent fallback strategy.
    
    Strategy:
    1. HTML first (fastest, if available)
    2. PyMuPDF4LLM for PDFs (fast, good accuracy)
    3. GROBID fallback (slower, high accuracy)
    """
    
    def __init__(self, **config: Any):
        """Initialize orchestrator with configuration."""
        self.logger = get_logger("parsers.orchestrator")
        self.config = config
        
        # Parser instances
        self.parsers: Dict[ParserType, BaseParser] = {}
        
        # Configuration
        self.pymupdf_confidence_threshold = config.get("pymupdf_confidence_threshold", 0.7)
        self.enable_fallback_to_grobid = config.get("enable_fallback_to_grobid", True)
        self.prefer_speed_over_accuracy = config.get("prefer_speed_over_accuracy", False)
        self.max_concurrent_parsers = config.get("max_concurrent_parsers", 3)
        
        # Fallback chain definition
        self.fallback_chain = [
            ParserType.PYMUPDF,    # Fast first attempt for PDFs
            ParserType.GROBID,     # High accuracy fallback
        ]
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all parsers."""
        if self._initialized:
            return
        
        self.logger.info("Initializing parser orchestrator...")
        
        # Initialize available parsers
        parser_configs = {
            ParserType.PYMUPDF: self.config.get("pymupdf", {}),
            ParserType.GROBID: self.config.get("grobid", {}),
            ParserType.HTML: self.config.get("html", {}),
        }
        
        # Initialize each parser
        for parser_type, parser_config in parser_configs.items():
            try:
                if parser_type == ParserType.PYMUPDF:
                    parser = PyMuPDFParser(**parser_config)
                elif parser_type == ParserType.GROBID:
                    parser = GrobidClient(**parser_config)
                elif parser_type == ParserType.HTML:
                    parser = ArxivHtmlParser(**parser_config)
                else:
                    continue
                
                await parser.initialize()
                self.parsers[parser_type] = parser
                self.logger.info(f"✅ {parser_type.value} parser initialized")
                
            except Exception as e:
                self.logger.warning(f"❌ Failed to initialize {parser_type.value} parser: {e}")
        
        if not self.parsers:
            raise RuntimeError("No parsers could be initialized")
        
        self._initialized = True
        self.logger.info(f"Parser orchestrator ready with {len(self.parsers)} parsers")
    
    async def cleanup(self):
        """Cleanup all parsers."""
        for parser in self.parsers.values():
            try:
                await parser.cleanup()
            except Exception as e:
                self.logger.warning(f"Error cleaning up parser: {e}")
        
        self.parsers.clear()
        self._initialized = False
        self.logger.info("Parser orchestrator cleanup completed")
    
    async def parse_with_strategy(self, request: ParseRequest) -> ParseResult:
        """
        Parse content using intelligent strategy selection.
        
        Args:
            request: Parse request containing content and options
            
        Returns:
            Parse result with extracted references
        """
        if not self._initialized:
            await self.initialize()
        
        # 1. Try HTML first if content type is HTML
        if request.content_type == "html" and ParserType.HTML in self.parsers:
            try:
                self.logger.info(f"Using HTML parser for {request.paper_id}")
                return await self.parsers[ParserType.HTML].parse(request)
            except Exception as e:
                self.logger.warning(f"HTML parsing failed: {e}")
        
        # 2. For PDFs, use intelligent fallback strategy
        if request.content_type == "pdf":
            return await self._parse_pdf_with_fallback(request)
        
        # 3. Default fallback for other content types
        return await self._parse_with_first_available(request)
    
    async def _parse_pdf_with_fallback(self, request: ParseRequest) -> ParseResult:
        """Parse PDF with PyMuPDF → GROBID fallback strategy."""
        
        # Try PyMuPDF4LLM first (fast)
        if ParserType.PYMUPDF in self.parsers:
            try:
                self.logger.info(f"Trying PyMuPDF4LLM for {request.paper_id}")
                pymupdf_result = await self.parsers[ParserType.PYMUPDF].parse(request)
                
                # Check if result meets quality threshold
                confidence = pymupdf_result.processing_metadata.confidence_score or 0.0
                ref_count = len(pymupdf_result.references)
                
                # Decision criteria for accepting PyMuPDF result
                should_accept = (
                    confidence >= self.pymupdf_confidence_threshold or
                    ref_count >= 5 or  # Found reasonable number of references
                    self.prefer_speed_over_accuracy or
                    not self.enable_fallback_to_grobid or
                    ParserType.GROBID not in self.parsers
                )
                
                if should_accept:
                    self.logger.info(
                        f"✅ Accepting PyMuPDF result: {ref_count} refs, "
                        f"confidence {confidence:.3f}, "
                        f"time {pymupdf_result.processing_metadata.processing_time_ms:.0f}ms"
                    )
                    return pymupdf_result
                
                # Store PyMuPDF result for potential combination
                self.logger.info(
                    f"⚠️ PyMuPDF below threshold ({confidence:.3f} < {self.pymupdf_confidence_threshold}), "
                    f"trying GROBID fallback"
                )
                fast_result = pymupdf_result
                
            except Exception as e:
                self.logger.warning(f"PyMuPDF parsing failed: {e}")
                fast_result = None
        else:
            fast_result = None
        
        # Fallback to GROBID for higher precision
        if self.enable_fallback_to_grobid and ParserType.GROBID in self.parsers:
            try:
                self.logger.info(f"Trying GROBID fallback for {request.paper_id}")
                grobid_result = await self.parsers[ParserType.GROBID].parse(request)
                
                # If we have both results, choose the better one
                if fast_result:
                    return self._select_better_result(fast_result, grobid_result, request.paper_id)
                
                return grobid_result
                
            except Exception as e:
                self.logger.error(f"GROBID fallback failed: {e}")
                
                # Return PyMuPDF result if available, even if low confidence
                if fast_result:
                    self.logger.info(f"🔄 Falling back to PyMuPDF result after GROBID failure")
                    return fast_result
                
                raise RuntimeError("All PDF parsing strategies failed")
        
        # Return PyMuPDF result if GROBID not enabled/available
        if fast_result:
            return fast_result
        
        raise RuntimeError("No suitable PDF parser available")
    
    def _select_better_result(
        self, 
        pymupdf_result: ParseResult, 
        grobid_result: ParseResult, 
        paper_id: str
    ) -> ParseResult:
        """Select the better result between PyMuPDF and GROBID."""
        
        pymupdf_refs = len(pymupdf_result.references)
        grobid_refs = len(grobid_result.references)
        
        pymupdf_conf = pymupdf_result.processing_metadata.confidence_score or 0.0
        grobid_conf = grobid_result.processing_metadata.confidence_score or 0.0
        
        pymupdf_time = pymupdf_result.processing_metadata.processing_time_ms or 0.0
        grobid_time = grobid_result.processing_metadata.processing_time_ms or 0.0
        
        # Selection criteria (weighted)
        pymupdf_score = (
            pymupdf_refs * 0.3 +       # Reference count
            pymupdf_conf * 0.4 +       # Confidence
            (1.0 if pymupdf_time < 5000 else 0.5) * 0.3  # Speed bonus
        )
        
        grobid_score = (
            grobid_refs * 0.3 +        # Reference count
            grobid_conf * 0.5 +        # Higher weight for GROBID confidence
            (1.0 if grobid_time < 10000 else 0.5) * 0.2  # Reasonable time
        )
        
        # Choose better result
        if grobid_score > pymupdf_score:
            self.logger.info(
                f"🏆 Selected GROBID: {grobid_refs} refs, {grobid_conf:.3f} conf, "
                f"score {grobid_score:.2f} vs PyMuPDF {pymupdf_score:.2f}"
            )
            return grobid_result
        else:
            self.logger.info(
                f"🏆 Selected PyMuPDF: {pymupdf_refs} refs, {pymupdf_conf:.3f} conf, "
                f"score {pymupdf_score:.2f} vs GROBID {grobid_score:.2f}"
            )
            return pymupdf_result
    
    async def _parse_with_first_available(self, request: ParseRequest) -> ParseResult:
        """Parse with first available parser."""
        for parser_type in self.fallback_chain:
            if parser_type in self.parsers:
                try:
                    return await self.parsers[parser_type].parse(request)
                except Exception as e:
                    self.logger.warning(f"{parser_type.value} parsing failed: {e}")
                    continue
        
        raise RuntimeError("All available parsers failed")
    
    async def parse_batch(self, requests: List[ParseRequest]) -> List[ParseResult]:
        """Parse multiple requests concurrently."""
        if not self._initialized:
            await self.initialize()
        
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_parsers)
        
        async def parse_with_semaphore(request: ParseRequest) -> ParseResult:
            async with semaphore:
                return await self.parse_with_strategy(request)
        
        # Execute batch parsing
        tasks = [parse_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                error_result = ParseResult(
                    references=[],
                    processing_metadata=self._create_error_metadata(str(result)),
                    errors=[str(result)]
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    def _create_error_metadata(self, error_message: str):
        """Create error metadata."""
        from ..models.schemas import ProcessingMetadata
        return ProcessingMetadata(
            parser_type=ParserType.COMBINED,
            error_message=error_message,
            confidence_score=0.0
        )
    
    def get_available_parsers(self) -> List[ParserType]:
        """Get list of available parser types."""
        return list(self.parsers.keys())
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all parsers."""
        if not self._initialized:
            await self.initialize()
        
        health_results = {}
        
        for parser_type, parser in self.parsers.items():
            try:
                health = await parser.health_check()
                health_results[parser_type.value] = health
            except Exception as e:
                health_results[parser_type.value] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        overall_status = "healthy" if all(
            result.get("status") == "healthy" 
            for result in health_results.values()
        ) else "degraded"
        
        return {
            "status": overall_status,
            "parsers": health_results,
            "strategy": {
                "pymupdf_confidence_threshold": self.pymupdf_confidence_threshold,
                "enable_fallback_to_grobid": self.enable_fallback_to_grobid,
                "prefer_speed_over_accuracy": self.prefer_speed_over_accuracy,
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from all parsers."""
        stats = {}
        for parser_type, parser in self.parsers.items():
            stats[parser_type.value] = parser.get_statistics()
        return stats