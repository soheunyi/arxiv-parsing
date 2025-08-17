"""
Core Pydantic schemas and data models for the arXiv parsing system.

This module defines the foundational data structures used throughout the system,
including base classes, enums, and common validation patterns.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class ParserType(str, Enum):
    """Available parser types."""
    GROBID = "grobid"
    HTML = "html"
    ANYSTYLE = "anystyle"
    PYMUPDF = "pymupdf"
    COMBINED = "combined"


class ContentType(str, Enum):
    """Content type enumeration."""
    PDF = "pdf"
    HTML = "html"
    TEI_XML = "tei_xml"
    PLAIN_TEXT = "plain_text"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence level for parsed data."""
    VERY_HIGH = "very_high"  # >0.9
    HIGH = "high"           # >0.7
    MEDIUM = "medium"       # >0.5
    LOW = "low"            # >0.3
    VERY_LOW = "very_low"  # <=0.3


class BaseSchema(BaseModel):
    """Base schema with common fields."""
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class ProcessingMetadata(BaseModel):
    """Metadata for processing operations."""
    
    parser_type: ParserType
    processing_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @property
    def confidence_level(self) -> Optional[ConfidenceLevel]:
        """Get confidence level based on score."""
        if self.confidence_score is None:
            return None
        
        if self.confidence_score > 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence_score > 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence_score > 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence_score > 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class AuthorName(BaseModel):
    """Author name representation."""
    
    first: Optional[str] = None
    middle: Optional[str] = None
    last: Optional[str] = None
    suffix: Optional[str] = None
    full_name: Optional[str] = None
    
    @validator("full_name", always=True)
    def generate_full_name(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Generate full name if not provided."""
        if v:
            return v
        
        parts = []
        if values.get("first"):
            parts.append(values["first"])
        if values.get("middle"):
            parts.append(values["middle"])
        if values.get("last"):
            parts.append(values["last"])
        if values.get("suffix"):
            parts.append(values["suffix"])
        
        return " ".join(parts) if parts else None
    
    def __str__(self) -> str:
        """String representation."""
        return self.full_name or ""


class DateInfo(BaseModel):
    """Date information with uncertainty handling."""
    
    raw_date: Optional[str] = None
    parsed_date: Optional[datetime] = None
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    date_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @validator("year")
    def validate_year(cls, v: Optional[int]) -> Optional[int]:
        """Validate year is reasonable."""
        if v is not None and (v < 1900 or v > 2030):
            raise ValueError(f"Year {v} is outside reasonable range (1900-2030)")
        return v


class Identifier(BaseModel):
    """Generic identifier with type information."""
    
    type: str  # "doi", "arxiv", "pmid", "url", etc.
    value: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @validator("value")
    def validate_value(cls, v: str) -> str:
        """Validate identifier value is not empty."""
        if not v.strip():
            raise ValueError("Identifier value cannot be empty")
        return v.strip()


class Location(BaseModel):
    """Location information for references."""
    
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    page_start: Optional[str] = None
    page_end: Optional[str] = None
    article_number: Optional[str] = None
    
    @validator("pages", always=True)
    def generate_pages(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Generate pages string from start/end if not provided."""
        if v:
            return v
        
        start = values.get("page_start")
        end = values.get("page_end")
        
        if start and end:
            return f"{start}-{end}"
        elif start:
            return start
        
        return None


class ProcessingStep(BaseModel):
    """Individual processing step information."""
    
    step_name: str
    status: ProcessingStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("duration_ms", always=True)
    def calculate_duration(cls, v: Optional[float], values: Dict[str, Any]) -> Optional[float]:
        """Calculate duration if not provided."""
        if v is not None:
            return v
        
        start_time = values.get("start_time")
        end_time = values.get("end_time")
        
        if start_time and end_time:
            return (end_time - start_time).total_seconds() * 1000
        
        return None


class ValidationError(BaseModel):
    """Validation error information."""
    
    field: str
    message: str
    value: Any
    severity: str = "error"  # "error", "warning", "info"
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class QualityMetrics(BaseModel):
    """Quality metrics for parsed data."""
    
    completeness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    accuracy_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    consistency_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    validation_errors: List[ValidationError] = Field(default_factory=list)
    validation_warnings: List[ValidationError] = Field(default_factory=list)
    
    @property
    def overall_quality_score(self) -> Optional[float]:
        """Calculate overall quality score."""
        scores = [
            self.completeness_score,
            self.accuracy_score,
            self.consistency_score
        ]
        valid_scores = [s for s in scores if s is not None]
        
        if not valid_scores:
            return None
        
        return sum(valid_scores) / len(valid_scores)
    
    @property
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.validation_errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return len(self.validation_warnings) > 0


class FeatureVector(BaseModel):
    """Feature vector for meta-learning."""
    
    features: Dict[str, Union[float, int, str, bool]]
    feature_names: List[str]
    extraction_time: datetime = Field(default_factory=datetime.utcnow)
    
    @validator("feature_names", always=True)
    def validate_feature_names(cls, v: List[str], values: Dict[str, Any]) -> List[str]:
        """Validate feature names match features dict."""
        features = values.get("features", {})
        if set(v) != set(features.keys()):
            raise ValueError("Feature names must match features dict keys")
        return v


class SearchQuery(BaseModel):
    """Search query representation."""
    
    query: str
    query_type: str  # "title", "author", "doi", "freetext"
    max_results: int = 10
    timeout_seconds: int = 10
    
    @validator("query")
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SearchResult(BaseModel):
    """Search result representation."""
    
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Export commonly used types
__all__ = [
    "ProcessingStatus",
    "ParserType", 
    "ContentType",
    "ConfidenceLevel",
    "BaseSchema",
    "ProcessingMetadata",
    "AuthorName",
    "DateInfo",
    "Identifier",
    "Location",
    "ProcessingStep",
    "ValidationError",
    "QualityMetrics",
    "FeatureVector",
    "SearchQuery",
    "SearchResult",
]