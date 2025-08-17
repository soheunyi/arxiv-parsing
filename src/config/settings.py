"""
Configuration management using Pydantic Settings.

This module provides centralized configuration management for the arXiv parsing system,
loading settings from environment variables with validation and type conversion.
"""

from pathlib import Path
from typing import List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Main application settings with environment variable support."""

    # ============================================================================
    # Core Application Settings
    # ============================================================================
    
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment"
    )
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log output format"
    )

    # ============================================================================
    # External Services Configuration
    # ============================================================================
    
    # GROBID Configuration
    grobid_url: str = Field(
        default="http://localhost:8070",
        description="GROBID service URL"
    )
    
    grobid_timeout: int = Field(
        default=300,
        description="GROBID request timeout in seconds"
    )
    
    grobid_max_connections: int = Field(
        default=10,
        description="Maximum concurrent connections to GROBID"
    )
    
    # arXiv Configuration
    arxiv_base_url: str = Field(
        default="https://arxiv.org",
        description="arXiv base URL"
    )
    
    arxiv_rate_limit_delay: float = Field(
        default=3.1,
        description="Delay between arXiv requests in seconds"
    )
    
    arxiv_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for arXiv requests"
    )
    
    arxiv_timeout: int = Field(
        default=30,
        description="arXiv request timeout in seconds"
    )
    
    # Google Search API Configuration
    google_search_api_key: Optional[str] = Field(
        default=None,
        description="Google Search API key"
    )
    
    google_search_engine_id: Optional[str] = Field(
        default=None,
        description="Google Custom Search Engine ID"
    )
    
    google_search_rate_limit: int = Field(
        default=100,
        description="Google Search API daily rate limit"
    )
    
    google_search_timeout: int = Field(
        default=10,
        description="Google Search API timeout in seconds"
    )

    # ============================================================================
    # Storage and Caching Configuration
    # ============================================================================
    
    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password"
    )
    
    redis_timeout: int = Field(
        default=5,
        description="Redis connection timeout in seconds"
    )
    
    redis_max_connections: int = Field(
        default=20,
        description="Maximum Redis connections in pool"
    )
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./data/arxiv_parsing.db",
        description="Database connection URL"
    )
    
    # File Storage Paths
    data_dir: Path = Field(
        default=Path("./data"),
        description="Data directory path"
    )
    
    cache_dir: Path = Field(
        default=Path("./data/cache"),
        description="Cache directory path"
    )
    
    output_dir: Path = Field(
        default=Path("./output"),
        description="Output directory path"
    )
    
    models_dir: Path = Field(
        default=Path("./data/models"),
        description="ML models directory path"
    )

    # ============================================================================
    # Processing Configuration
    # ============================================================================
    
    # Parser Configuration
    enable_html_parser: bool = Field(
        default=True,
        description="Enable HTML parser"
    )
    
    enable_pdf_fallback: bool = Field(
        default=True,
        description="Enable PDF fallback when HTML unavailable"
    )
    
    enable_anystyle_parser: bool = Field(
        default=False,
        description="Enable AnyStyle parser (requires Ruby)"
    )
    
    default_parser_priority: List[str] = Field(
        default=["grobid", "html", "anystyle"],
        description="Parser priority order"
    )
    
    # Batch Processing
    max_concurrent_requests: int = Field(
        default=5,
        description="Maximum concurrent requests"
    )
    
    batch_size: int = Field(
        default=10,
        description="Default batch processing size"
    )
    
    max_retry_attempts: int = Field(
        default=3,
        description="Maximum retry attempts for failed operations"
    )
    
    # Feature Extraction and ML
    enable_meta_learning: bool = Field(
        default=True,
        description="Enable meta-learning features"
    )
    
    model_update_interval: str = Field(
        default="24h",
        description="Model update interval"
    )
    
    feature_cache_ttl: int = Field(
        default=86400,
        description="Feature cache TTL in seconds"
    )

    # ============================================================================
    # Performance and Monitoring
    # ============================================================================
    
    # Rate Limiting
    global_rate_limit: int = Field(
        default=100,
        description="Global rate limit per window"
    )
    
    rate_limit_window: int = Field(
        default=3600,
        description="Rate limit window in seconds"
    )
    
    # Timeouts
    http_timeout: int = Field(
        default=30,
        description="General HTTP timeout in seconds"
    )
    
    parser_timeout: int = Field(
        default=120,
        description="Parser timeout in seconds"
    )
    
    search_timeout: int = Field(
        default=10,
        description="Search timeout in seconds"
    )
    
    # Memory Limits
    max_memory_usage: str = Field(
        default="1GB",
        description="Maximum memory usage"
    )
    
    max_file_size: str = Field(
        default="50MB",
        description="Maximum file size for processing"
    )

    # ============================================================================
    # Development and Testing
    # ============================================================================
    
    # Test Configuration
    test_data_dir: Path = Field(
        default=Path("./tests/fixtures"),
        description="Test data directory"
    )
    
    enable_integration_tests: bool = Field(
        default=True,
        description="Enable integration tests"
    )
    
    mock_external_services: bool = Field(
        default=False,
        description="Mock external services for testing"
    )
    
    # Development Tools
    enable_profiling: bool = Field(
        default=False,
        description="Enable performance profiling"
    )
    
    enable_debug_logging: bool = Field(
        default=False,
        description="Enable debug logging"
    )
    
    save_intermediate_results: bool = Field(
        default=False,
        description="Save intermediate processing results"
    )

    # ============================================================================
    # Security Configuration
    # ============================================================================
    
    # API Security
    api_key_header: str = Field(
        default="X-API-Key",
        description="API key header name"
    )
    
    rate_limit_by_ip: bool = Field(
        default=True,
        description="Enable rate limiting by IP address"
    )
    
    # Input Validation
    max_paper_id_length: int = Field(
        default=50,
        description="Maximum paper ID length"
    )
    
    allowed_file_extensions: List[str] = Field(
        default=[".pdf", ".html", ".xml"],
        description="Allowed file extensions"
    )
    
    sanitize_inputs: bool = Field(
        default=True,
        description="Enable input sanitization"
    )

    # ============================================================================
    # Optional AnyStyle Configuration
    # ============================================================================
    
    anystyle_ruby_path: str = Field(
        default="/usr/bin/ruby",
        description="Path to Ruby executable"
    )
    
    anystyle_gem_path: str = Field(
        default="/usr/local/bin/anystyle",
        description="Path to AnyStyle gem"
    )
    
    anystyle_timeout: int = Field(
        default=60,
        description="AnyStyle processing timeout in seconds"
    )

    # ============================================================================
    # Validators
    # ============================================================================
    
    @validator("data_dir", "cache_dir", "output_dir", "models_dir", "test_data_dir")
    def ensure_path_exists(cls, v: Path) -> Path:
        """Ensure directory paths exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("grobid_url", "arxiv_base_url")
    def validate_urls(cls, v: str) -> str:
        """Validate URL format."""
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: {v}")
        return v
    
    @validator("default_parser_priority")
    def validate_parser_priority(cls, v: List[str]) -> List[str]:
        """Validate parser priority list."""
        valid_parsers = {"grobid", "html", "anystyle"}
        for parser in v:
            if parser not in valid_parsers:
                raise ValueError(f"Invalid parser: {parser}. Must be one of {valid_parsers}")
        return v
    
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is properly set."""
        if v == "production":
            # Additional validation for production environment
            pass
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings