"""
Paper data model for the arXiv parsing system.

This module defines the Paper model and related structures for representing
arXiv papers with their metadata, content, and processing information.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from pydantic import Field, validator

from .reference import Reference
from .schemas import (
    AuthorName,
    BaseSchema,
    ContentType,
    DateInfo,
    Identifier,
    ProcessingStatus,
    QualityMetrics,
)


class ArxivCategory(BaseSchema):
    """arXiv category information."""
    
    primary: str  # e.g., "cs.AI"
    secondary: List[str] = Field(default_factory=list)  # e.g., ["cs.LG", "stat.ML"]
    
    @validator("primary")
    def validate_primary_category(cls, v: str) -> str:
        """Validate primary category format."""
        if not v or "." not in v:
            raise ValueError("Primary category must be in format 'subject.subclass'")
        return v


class PaperContent(BaseSchema):
    """Paper content in various formats."""
    
    title: Optional[str] = None
    abstract: Optional[str] = None
    
    # Full text content
    full_text: Optional[str] = None
    html_content: Optional[str] = None
    pdf_path: Optional[Path] = None
    
    # Structured sections
    sections: Dict[str, str] = Field(default_factory=dict)
    
    # Content metadata
    content_type: ContentType = ContentType.UNKNOWN
    word_count: Optional[int] = None
    language: str = "en"
    encoding: str = "utf-8"
    
    @validator("sections")
    def validate_sections(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate sections dictionary."""
        # Normalize section keys to lowercase
        return {k.lower(): v for k, v in v.items()}
    
    @property
    def has_full_text(self) -> bool:
        """Check if full text content is available."""
        return bool(self.full_text or self.html_content or self.pdf_path)
    
    def get_section(self, section_name: str) -> Optional[str]:
        """Get section content by name (case-insensitive)."""
        return self.sections.get(section_name.lower())


class PaperMetadata(BaseSchema):
    """Paper metadata information."""
    
    # arXiv specific
    arxiv_id: str
    arxiv_url: Optional[str] = None
    version: int = 1
    
    # Publication details
    title: str
    authors: List[AuthorName] = Field(default_factory=list)
    
    # Dates
    submission_date: Optional[DateInfo] = None
    publication_date: Optional[DateInfo] = None
    last_updated: Optional[DateInfo] = None
    
    # Categories and subjects
    categories: Optional[ArxivCategory] = None
    subjects: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    
    # Identifiers
    identifiers: List[Identifier] = Field(default_factory=list)
    doi: Optional[str] = None
    
    # Comments and notes
    comments: Optional[str] = None
    journal_ref: Optional[str] = None
    
    @validator("arxiv_id")
    def validate_arxiv_id(cls, v: str) -> str:
        """Validate arXiv ID format."""
        if not v:
            raise ValueError("arXiv ID cannot be empty")
        
        # Handle both old and new arXiv ID formats
        # Old: subject-class/YYMMnnn
        # New: YYMM.nnnn[vN]
        v = v.strip()
        
        # Remove 'arxiv:' prefix if present
        if v.lower().startswith('arxiv:'):
            v = v[6:]
        
        # Basic validation - more specific validation can be added
        if len(v) < 5:
            raise ValueError(f"arXiv ID '{v}' is too short")
        
        return v
    
    @validator("arxiv_url", always=True)
    def generate_arxiv_url(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Generate arXiv URL if not provided."""
        if v:
            return v
        
        arxiv_id = values.get("arxiv_id")
        if arxiv_id:
            return f"https://arxiv.org/abs/{arxiv_id}"
        
        return None
    
    @validator("title")
    def validate_title(cls, v: str) -> str:
        """Validate and clean title."""
        if not v or not v.strip():
            raise ValueError("Title cannot be empty")
        
        # Clean up common title formatting issues
        v = v.strip()
        v = ' '.join(v.split())  # Normalize whitespace
        
        return v
    
    def add_identifier(self, id_type: str, value: str, confidence: Optional[float] = None) -> None:
        """Add an identifier to the paper."""
        identifier = Identifier(type=id_type, value=value, confidence=confidence)
        
        # Remove existing identifier of same type
        self.identifiers = [id for id in self.identifiers if id.type != id_type]
        self.identifiers.append(identifier)
    
    def get_identifier(self, id_type: str) -> Optional[Identifier]:
        """Get identifier by type."""
        for identifier in self.identifiers:
            if identifier.type == id_type:
                return identifier
        return None
    
    @property
    def author_names(self) -> List[str]:
        """Get list of author full names."""
        return [str(author) for author in self.authors if str(author)]
    
    @property
    def primary_category(self) -> Optional[str]:
        """Get primary arXiv category."""
        return self.categories.primary if self.categories else None


class ProcessingHistory(BaseSchema):
    """Processing history for a paper."""
    
    processing_attempts: List[Dict[str, Any]] = Field(default_factory=list)
    last_processed: Optional[datetime] = None
    last_successful_parse: Optional[datetime] = None
    total_processing_time_ms: float = 0.0
    
    def add_attempt(
        self,
        parser_type: str,
        status: ProcessingStatus,
        processing_time_ms: float,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a processing attempt."""
        attempt = {
            "timestamp": datetime.utcnow(),
            "parser_type": parser_type,
            "status": status,
            "processing_time_ms": processing_time_ms,
            "error_message": error_message,
            "metadata": metadata or {}
        }
        
        self.processing_attempts.append(attempt)
        self.last_processed = attempt["timestamp"]
        self.total_processing_time_ms += processing_time_ms
        
        if status == ProcessingStatus.COMPLETED:
            self.last_successful_parse = attempt["timestamp"]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of processing attempts."""
        if not self.processing_attempts:
            return 0.0
        
        successful = sum(
            1 for attempt in self.processing_attempts
            if attempt["status"] == ProcessingStatus.COMPLETED
        )
        
        return successful / len(self.processing_attempts)
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time."""
        if not self.processing_attempts:
            return 0.0
        
        total_time = sum(
            attempt["processing_time_ms"] for attempt in self.processing_attempts
        )
        
        return total_time / len(self.processing_attempts)


class Paper(BaseSchema):
    """Main paper model containing all paper information."""
    
    # Core metadata
    metadata: PaperMetadata
    
    # Content
    content: Optional[PaperContent] = None
    
    # References
    references: List[Reference] = Field(default_factory=list)
    
    # Processing information
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_history: ProcessingHistory = Field(default_factory=ProcessingHistory)
    
    # Quality assessment
    quality_metrics: Optional[QualityMetrics] = None
    
    # Search and enhancement
    related_papers: List[str] = Field(default_factory=list)  # arXiv IDs
    citation_count: Optional[int] = None
    
    # Cache and optimization
    cache_key: Optional[str] = None
    last_cache_update: Optional[datetime] = None
    
    @validator("cache_key", always=True)
    def generate_cache_key(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Generate cache key based on arXiv ID."""
        if v:
            return v
        
        metadata = values.get("metadata")
        if metadata and hasattr(metadata, "arxiv_id"):
            return f"paper:{metadata.arxiv_id}"
        
        return None
    
    def add_reference(self, reference: Reference) -> None:
        """Add a reference to the paper."""
        # Check for duplicates based on some criteria
        existing_titles = {ref.title.lower() for ref in self.references if ref.title}
        
        if reference.title and reference.title.lower() not in existing_titles:
            self.references.append(reference)
    
    def get_references_by_type(self, ref_type: str) -> List[Reference]:
        """Get references by type."""
        return [ref for ref in self.references if ref.reference_type == ref_type]
    
    def update_processing_status(
        self,
        status: ProcessingStatus,
        parser_type: str,
        processing_time_ms: float,
        error_message: Optional[str] = None
    ) -> None:
        """Update processing status and history."""
        self.processing_status = status
        self.processing_history.add_attempt(
            parser_type=parser_type,
            status=status,
            processing_time_ms=processing_time_ms,
            error_message=error_message
        )
        self.update_timestamp()
    
    @property
    def arxiv_id(self) -> str:
        """Get arXiv ID from metadata."""
        return self.metadata.arxiv_id
    
    @property
    def title(self) -> str:
        """Get title from metadata."""
        return self.metadata.title
    
    @property
    def reference_count(self) -> int:
        """Get number of references."""
        return len(self.references)
    
    @property
    def has_content(self) -> bool:
        """Check if paper has content."""
        return self.content is not None and self.content.has_full_text
    
    @property
    def is_processed(self) -> bool:
        """Check if paper has been successfully processed."""
        return self.processing_status == ProcessingStatus.COMPLETED
    
    def to_dict(self, include_content: bool = True) -> Dict[str, Any]:
        """Convert to dictionary with optional content exclusion."""
        data = self.dict()
        
        if not include_content and "content" in data:
            # Remove heavy content fields for lightweight operations
            data["content"] = {
                "content_type": data["content"]["content_type"],
                "word_count": data["content"]["word_count"],
                "language": data["content"]["language"],
                "has_full_text": self.has_content
            }
        
        return data
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            Path: lambda v: str(v),
            datetime: lambda v: v.isoformat(),
        }