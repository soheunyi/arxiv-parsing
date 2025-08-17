"""
Base parser interface for the arXiv parsing system.

This module defines the abstract base class and common interfaces for all
reference parsers in the system, ensuring consistent behavior and error handling.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel

from ..config.logging import get_logger
from ..models.paper import Paper, PaperContent
from ..models.reference import Reference, ParsedReference
from ..models.schemas import ParserType, ProcessingMetadata, ProcessingStatus


class ParseRequest(BaseModel):
    """Request object for parsing operations."""
    
    paper_id: str
    content: Union[str, Path]  # Text content or file path
    content_type: str  # "pdf", "html", "text", "tei_xml"
    parser_options: Dict[str, Any] = {}
    timeout_seconds: Optional[int] = None
    
    class Config:
        arbitrary_types_allowed = True


class ParseResult(BaseModel):
    """Result object for parsing operations."""
    
    references: List[Reference]
    processing_metadata: ProcessingMetadata
    raw_output: Optional[str] = None
    warnings: List[str] = []
    errors: List[str] = []
    
    @property
    def success(self) -> bool:
        """Check if parsing was successful."""
        return len(self.errors) == 0
    
    @property
    def reference_count(self) -> int:
        """Get number of extracted references."""
        return len(self.references)


class ParserCapabilities(BaseModel):
    """Parser capabilities and features."""
    
    parser_type: ParserType
    supported_content_types: List[str]
    supports_async: bool = True
    supports_batch: bool = False
    confidence_scoring: bool = False
    field_level_extraction: bool = False
    requires_external_service: bool = False
    service_dependencies: List[str] = []
    
    max_file_size_mb: Optional[int] = None
    max_processing_time_seconds: Optional[int] = None
    
    quality_features: Dict[str, bool] = {
        "author_extraction": False,
        "title_extraction": False, 
        "venue_extraction": False,
        "date_extraction": False,
        "doi_extraction": False,
        "citation_context": False,
    }


class BaseParser(ABC):
    """
    Abstract base class for all reference parsers.
    
    This class defines the common interface and provides shared functionality
    for all parser implementations in the system.
    """
    
    def __init__(self, **config: Any):
        """
        Initialize the parser with configuration.
        
        Args:
            **config: Parser-specific configuration options
        """
        self.config = config
        self.logger = get_logger(f"parsers.{self.parser_type.value}")
        self._is_initialized = False
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
        }
    
    @property
    @abstractmethod
    def parser_type(self) -> ParserType:
        """Return the parser type."""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> ParserCapabilities:
        """Return parser capabilities."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the parser (e.g., start services, load models).
        
        This method should be called before using the parser and should
        handle any setup required for the parser to function.
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up parser resources.
        
        This method should be called when the parser is no longer needed
        and should handle any cleanup required.
        """
        pass
    
    @abstractmethod
    async def _parse_implementation(self, request: ParseRequest) -> ParseResult:
        """
        Core parsing implementation to be provided by subclasses.
        
        Args:
            request: Parse request containing content and options
            
        Returns:
            Parse result with extracted references
        """
        pass
    
    async def parse(self, request: ParseRequest) -> ParseResult:
        """
        Parse content and extract references.
        
        This method provides the main entry point for parsing operations
        with error handling, timing, and statistics tracking.
        
        Args:
            request: Parse request containing content and options
            
        Returns:
            Parse result with extracted references and metadata
        """
        if not self._is_initialized:
            await self.initialize()
            self._is_initialized = True
        
        start_time = time.time()
        self._stats["total_requests"] += 1
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Set timeout if not specified
            if request.timeout_seconds is None:
                max_time = self.capabilities.max_processing_time_seconds
                request.timeout_seconds = max_time or 120
            
            # Perform parsing with timeout
            result = await asyncio.wait_for(
                self._parse_implementation(request),
                timeout=request.timeout_seconds
            )
            
            # Update processing metadata
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            result.processing_metadata.processing_time_ms = processing_time
            result.processing_metadata.parser_type = self.parser_type
            
            self._stats["successful_requests"] += 1
            self._stats["total_processing_time"] += processing_time
            
            self.logger.info(
                f"Successfully parsed content",
                extra={
                    "paper_id": request.paper_id,
                    "parser_type": self.parser_type.value,
                    "reference_count": result.reference_count,
                    "processing_time_ms": processing_time,
                    "confidence_score": result.processing_metadata.confidence_score,
                }
            )
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Parsing timed out after {request.timeout_seconds} seconds"
            self.logger.error(error_msg, extra={"paper_id": request.paper_id})
            
            return self._create_error_result(
                error_msg,
                (time.time() - start_time) * 1000
            )
            
        except Exception as e:
            error_msg = f"Parsing failed: {str(e)}"
            self.logger.error(
                error_msg,
                extra={"paper_id": request.paper_id},
                exc_info=True
            )
            
            return self._create_error_result(
                error_msg,
                (time.time() - start_time) * 1000
            )
        
        finally:
            self._stats["total_processing_time"] += (time.time() - start_time) * 1000
    
    def _validate_request(self, request: ParseRequest) -> None:
        """
        Validate the parse request.
        
        Args:
            request: Parse request to validate
            
        Raises:
            ValueError: If request is invalid
        """
        # Check content type support
        if request.content_type not in self.capabilities.supported_content_types:
            raise ValueError(
                f"Content type '{request.content_type}' not supported by "
                f"{self.parser_type.value} parser. Supported types: "
                f"{', '.join(self.capabilities.supported_content_types)}"
            )
        
        # Check file size if applicable
        if isinstance(request.content, Path):
            if not request.content.exists():
                raise ValueError(f"File not found: {request.content}")
            
            if self.capabilities.max_file_size_mb:
                file_size_mb = request.content.stat().st_size / (1024 * 1024)
                if file_size_mb > self.capabilities.max_file_size_mb:
                    raise ValueError(
                        f"File size ({file_size_mb:.1f}MB) exceeds maximum "
                        f"({self.capabilities.max_file_size_mb}MB)"
                    )
    
    def _create_error_result(self, error_message: str, processing_time_ms: float) -> ParseResult:
        """
        Create an error result.
        
        Args:
            error_message: Error message
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            Error result with empty references
        """
        self._stats["failed_requests"] += 1
        
        metadata = ProcessingMetadata(
            parser_type=self.parser_type,
            processing_time_ms=processing_time_ms,
            error_message=error_message,
            confidence_score=0.0
        )
        
        return ParseResult(
            references=[],
            processing_metadata=metadata,
            errors=[error_message]
        )
    
    async def parse_batch(self, requests: List[ParseRequest]) -> List[ParseResult]:
        """
        Parse multiple requests in batch.
        
        Args:
            requests: List of parse requests
            
        Returns:
            List of parse results
        """
        if not self.capabilities.supports_batch:
            # Fall back to sequential processing
            results = []
            for request in requests:
                result = await self.parse(request)
                results.append(result)
            return results
        
        # Subclasses can override for true batch processing
        return await asyncio.gather(*[self.parse(req) for req in requests])
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get parser statistics.
        
        Returns:
            Dictionary containing parser statistics
        """
        total_requests = self._stats["total_requests"]
        
        return {
            "parser_type": self.parser_type.value,
            "total_requests": total_requests,
            "successful_requests": self._stats["successful_requests"],
            "failed_requests": self._stats["failed_requests"],
            "success_rate": (
                self._stats["successful_requests"] / total_requests
                if total_requests > 0 else 0.0
            ),
            "average_processing_time_ms": (
                self._stats["total_processing_time"] / total_requests
                if total_requests > 0 else 0.0
            ),
            "total_processing_time_ms": self._stats["total_processing_time"],
        }
    
    def reset_statistics(self) -> None:
        """Reset parser statistics."""
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the parser.
        
        Returns:
            Health check result
        """
        try:
            if not self._is_initialized:
                await self.initialize()
                self._is_initialized = True
            
            # Subclasses can override for more specific health checks
            return {
                "status": "healthy",
                "parser_type": self.parser_type.value,
                "initialized": self._is_initialized,
                "capabilities": self.capabilities.dict(),
                "statistics": self.get_statistics(),
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "unhealthy",
                "parser_type": self.parser_type.value,
                "error": str(e),
            }
    
    def __str__(self) -> str:
        """String representation of parser."""
        return f"{self.parser_type.value.title()}Parser"
    
    def __repr__(self) -> str:
        """Detailed string representation of parser."""
        return (
            f"{self.__class__.__name__}("
            f"type={self.parser_type.value}, "
            f"initialized={self._is_initialized}, "
            f"requests={self._stats['total_requests']}"
            f")"
        )