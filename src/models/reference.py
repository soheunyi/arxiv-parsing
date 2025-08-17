"""
Reference data model for the arXiv parsing system.

This module defines the Reference model and related structures for representing
citations and bibliographic references extracted from papers.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import Field, validator

from .schemas import (
    AuthorName,
    BaseSchema,
    DateInfo,
    Identifier,
    Location,
    ParserType,
    ProcessingMetadata,
    QualityMetrics,
)


class ReferenceType(str, Enum):
    """Types of references."""
    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    THESIS = "thesis"
    PREPRINT = "preprint"
    TECHNICAL_REPORT = "technical_report"
    WEBSITE = "website"
    SOFTWARE = "software"
    DATASET = "dataset"
    UNKNOWN = "unknown"


class Venue(BaseSchema):
    """Publication venue information."""
    
    name: Optional[str] = None
    short_name: Optional[str] = None
    venue_type: str = "unknown"  # "journal", "conference", "workshop", etc.
    
    # Journal-specific
    issn: Optional[str] = None
    
    # Conference-specific
    conference_series: Optional[str] = None
    conference_year: Optional[int] = None
    
    @validator("name")
    def clean_venue_name(cls, v: Optional[str]) -> Optional[str]:
        """Clean and normalize venue name."""
        if not v:
            return v
        
        # Remove common prefixes/suffixes
        v = v.strip()
        
        # Normalize common venue name variations
        # This could be expanded with a more comprehensive mapping
        cleanups = [
            ("Proc. ", "Proceedings of "),
            ("Conf. ", "Conference "),
            ("Int. ", "International "),
            ("IEEE Trans. ", "IEEE Transactions on "),
        ]
        
        for short, long in cleanups:
            if v.startswith(short):
                v = long + v[len(short):]
        
        return v


class CSLJSONReference(BaseSchema):
    """CSL-JSON format reference for bibliographic data."""
    
    # CSL-JSON standard fields
    type: str = "article-journal"  # CSL-JSON item type
    id: Optional[str] = None
    
    # Title information
    title: Optional[str] = None
    title_short: Optional[str] = None
    
    # Author information
    author: List[Dict[str, str]] = Field(default_factory=list)
    editor: List[Dict[str, str]] = Field(default_factory=list)
    
    # Date information
    issued: Optional[Dict[str, Any]] = None
    submitted: Optional[Dict[str, Any]] = None
    
    # Publication information
    container_title: Optional[str] = None  # Journal/conference name
    container_title_short: Optional[str] = None
    
    # Location information
    volume: Optional[str] = None
    issue: Optional[str] = None
    page: Optional[str] = None
    page_first: Optional[str] = None
    
    # Identifiers
    DOI: Optional[str] = None
    ISBN: Optional[str] = None
    ISSN: Optional[str] = None
    URL: Optional[str] = None
    
    # Publisher information
    publisher: Optional[str] = None
    publisher_place: Optional[str] = None
    
    # Additional fields
    abstract: Optional[str] = None
    keyword: Optional[str] = None
    note: Optional[str] = None
    
    @classmethod
    def from_reference(cls, reference: "Reference") -> "CSLJSONReference":
        """Create CSL-JSON from Reference object."""
        csl_data = {
            "type": cls._map_reference_type(reference.reference_type),
            "title": reference.title,
        }
        
        # Convert authors
        if reference.authors:
            csl_data["author"] = [
                {
                    "family": author.last or "",
                    "given": " ".join(filter(None, [author.first, author.middle])) or "",
                }
                for author in reference.authors
            ]
        
        # Convert date
        if reference.publication_date and reference.publication_date.year:
            csl_data["issued"] = {
                "date-parts": [[
                    reference.publication_date.year,
                    reference.publication_date.month or 1,
                    reference.publication_date.day or 1
                ]]
            }
        
        # Venue information
        if reference.venue:
            csl_data["container-title"] = reference.venue.name
            csl_data["container-title-short"] = reference.venue.short_name
        
        # Location information
        if reference.location:
            if reference.location.volume:
                csl_data["volume"] = reference.location.volume
            if reference.location.issue:
                csl_data["issue"] = reference.location.issue
            if reference.location.pages:
                csl_data["page"] = reference.location.pages
            if reference.location.page_start:
                csl_data["page-first"] = reference.location.page_start
        
        # Identifiers
        doi_id = reference.get_identifier("doi")
        if doi_id:
            csl_data["DOI"] = doi_id.value
        
        url_id = reference.get_identifier("url")
        if url_id:
            csl_data["URL"] = url_id.value
        
        return cls(**csl_data)
    
    @staticmethod
    def _map_reference_type(ref_type: ReferenceType) -> str:
        """Map internal reference type to CSL-JSON type."""
        mapping = {
            ReferenceType.JOURNAL_ARTICLE: "article-journal",
            ReferenceType.CONFERENCE_PAPER: "paper-conference",
            ReferenceType.BOOK: "book",
            ReferenceType.BOOK_CHAPTER: "chapter",
            ReferenceType.THESIS: "thesis",
            ReferenceType.PREPRINT: "manuscript",
            ReferenceType.TECHNICAL_REPORT: "report",
            ReferenceType.WEBSITE: "webpage",
            ReferenceType.SOFTWARE: "software",
            ReferenceType.DATASET: "dataset",
            ReferenceType.UNKNOWN: "document",
        }
        return mapping.get(ref_type, "document")


class ParsedReference(BaseSchema):
    """Raw parsed reference from a specific parser."""
    
    parser_type: ParserType
    raw_text: str
    parsed_fields: Dict[str, Any] = Field(default_factory=dict)
    processing_metadata: ProcessingMetadata
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    
    def get_field_confidence(self, field_name: str) -> Optional[float]:
        """Get confidence score for a specific field."""
        return self.confidence_scores.get(field_name)


class Reference(BaseSchema):
    """Main reference model containing all reference information."""
    
    # Raw citation text
    raw_citation: Optional[str] = None
    
    # Parsed information
    title: Optional[str] = None
    authors: List[AuthorName] = Field(default_factory=list)
    
    # Publication details
    venue: Optional[Venue] = None
    publication_date: Optional[DateInfo] = None
    
    # Reference type and location
    reference_type: ReferenceType = ReferenceType.UNKNOWN
    location: Optional[Location] = None
    
    # Identifiers
    identifiers: List[Identifier] = Field(default_factory=list)
    
    # Abstract and keywords
    abstract: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    
    # Publisher information
    publisher: Optional[str] = None
    
    # Parsing information
    parsed_versions: List[ParsedReference] = Field(default_factory=list)
    best_parser: Optional[ParserType] = None
    
    # Quality and validation
    quality_metrics: Optional[QualityMetrics] = None
    is_validated: bool = False
    validation_source: Optional[str] = None  # "doi", "google_search", etc.
    
    # Position in original document
    position_in_text: Optional[int] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    
    @validator("title")
    def clean_title(cls, v: Optional[str]) -> Optional[str]:
        """Clean and normalize title."""
        if not v:
            return v
        
        # Basic title cleaning
        v = v.strip()
        v = ' '.join(v.split())  # Normalize whitespace
        
        # Remove common formatting artifacts
        if v.endswith('.'):
            v = v[:-1]
        
        return v
    
    def add_parsed_version(self, parsed_ref: ParsedReference) -> None:
        """Add a parsed version from a specific parser."""
        # Remove existing version from same parser
        self.parsed_versions = [
            p for p in self.parsed_versions 
            if p.parser_type != parsed_ref.parser_type
        ]
        
        self.parsed_versions.append(parsed_ref)
        
        # Update best parser based on confidence
        if (not self.best_parser or 
            parsed_ref.processing_metadata.confidence_score and
            parsed_ref.processing_metadata.confidence_score > 
            self._get_best_confidence()):
            self.best_parser = parsed_ref.parser_type
    
    def _get_best_confidence(self) -> float:
        """Get confidence score of current best parser."""
        if not self.best_parser:
            return 0.0
        
        for parsed in self.parsed_versions:
            if (parsed.parser_type == self.best_parser and 
                parsed.processing_metadata.confidence_score):
                return parsed.processing_metadata.confidence_score
        
        return 0.0
    
    def add_identifier(self, id_type: str, value: str, confidence: Optional[float] = None) -> None:
        """Add an identifier to the reference."""
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
    
    def get_parsed_by_parser(self, parser_type: ParserType) -> Optional[ParsedReference]:
        """Get parsed version by parser type."""
        for parsed in self.parsed_versions:
            if parsed.parser_type == parser_type:
                return parsed
        return None
    
    @property
    def doi(self) -> Optional[str]:
        """Get DOI if available."""
        doi_id = self.get_identifier("doi")
        return doi_id.value if doi_id else None
    
    @property
    def arxiv_id(self) -> Optional[str]:
        """Get arXiv ID if available."""
        arxiv_id = self.get_identifier("arxiv")
        return arxiv_id.value if arxiv_id else None
    
    @property
    def url(self) -> Optional[str]:
        """Get URL if available."""
        url_id = self.get_identifier("url")
        return url_id.value if url_id else None
    
    @property
    def author_names(self) -> List[str]:
        """Get list of author full names."""
        return [str(author) for author in self.authors if str(author)]
    
    @property
    def venue_name(self) -> Optional[str]:
        """Get venue name if available."""
        return self.venue.name if self.venue else None
    
    @property
    def publication_year(self) -> Optional[int]:
        """Get publication year if available."""
        return self.publication_date.year if self.publication_date else None
    
    @property
    def is_complete(self) -> bool:
        """Check if reference has essential information."""
        return bool(self.title and self.authors)
    
    @property
    def completeness_score(self) -> float:
        """Calculate completeness score (0-1)."""
        fields = [
            self.title,
            self.authors,
            self.venue_name,
            self.publication_year,
            self.doi or self.arxiv_id or self.url,
        ]
        
        filled_fields = sum(1 for field in fields if field)
        return filled_fields / len(fields)
    
    def to_csl_json(self) -> CSLJSONReference:
        """Convert to CSL-JSON format."""
        return CSLJSONReference.from_reference(self)
    
    def to_bibtex(self) -> str:
        """Convert to BibTeX format."""
        # This is a basic implementation - could be enhanced
        entry_type = "article"
        if self.reference_type == ReferenceType.BOOK:
            entry_type = "book"
        elif self.reference_type == ReferenceType.CONFERENCE_PAPER:
            entry_type = "inproceedings"
        elif self.reference_type == ReferenceType.THESIS:
            entry_type = "phdthesis"
        
        # Generate citation key
        key = self._generate_bibtex_key()
        
        lines = [f"@{entry_type}{{{key},"]
        
        if self.title:
            lines.append(f'  title = {{{self.title}}},')
        
        if self.authors:
            author_str = " and ".join(str(author) for author in self.authors)
            lines.append(f'  author = {{{author_str}}},')
        
        if self.venue_name:
            if entry_type == "article":
                lines.append(f'  journal = {{{self.venue_name}}},')
            else:
                lines.append(f'  booktitle = {{{self.venue_name}}},')
        
        if self.publication_year:
            lines.append(f'  year = {{{self.publication_year}}},')
        
        if self.location:
            if self.location.volume:
                lines.append(f'  volume = {{{self.location.volume}}},')
            if self.location.issue:
                lines.append(f'  number = {{{self.location.issue}}},')
            if self.location.pages:
                lines.append(f'  pages = {{{self.location.pages}}},')
        
        if self.doi:
            lines.append(f'  doi = {{{self.doi}}},')
        
        if self.url:
            lines.append(f'  url = {{{self.url}}},')
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _generate_bibtex_key(self) -> str:
        """Generate BibTeX citation key."""
        parts = []
        
        # Add first author last name
        if self.authors:
            first_author = self.authors[0]
            if first_author.last:
                parts.append(first_author.last.lower())
        
        # Add year
        if self.publication_year:
            parts.append(str(self.publication_year))
        
        # Add title words (first few significant words)
        if self.title:
            title_words = [
                word.lower() for word in self.title.split()
                if len(word) > 3 and word.lower() not in {"the", "and", "for", "with"}
            ]
            if title_words:
                parts.append(title_words[0])
        
        return "_".join(parts) if parts else f"ref_{self.id}"
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        use_enum_values = True