"""
Logging configuration for the arXiv parsing system.

This module provides structured logging setup with JSON formatting,
correlation IDs, and proper error handling for production environments.
"""

import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import orjson
from pydantic import BaseModel

from .settings import get_settings


class LogFormatter(logging.Formatter):
    """Custom JSON log formatter with structured output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        settings = get_settings()
        
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if available
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id
        
        # Add request ID if available
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        # Add user ID if available
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        
        # Add paper ID if available
        if hasattr(record, "paper_id"):
            log_data["paper_id"] = record.paper_id
        
        # Add parser type if available
        if hasattr(record, "parser_type"):
            log_data["parser_type"] = record.parser_type
        
        # Add timing information if available
        if hasattr(record, "duration"):
            log_data["duration_ms"] = record.duration
        
        # Add memory usage if available
        if hasattr(record, "memory_usage"):
            log_data["memory_usage_mb"] = record.memory_usage
        
        # Add exception information
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key.startswith("extra_"):
                log_data[key[6:]] = value  # Remove 'extra_' prefix
        
        # Format based on settings
        if settings.log_format == "json":
            return orjson.dumps(log_data).decode()
        else:
            # Simple text format for development
            timestamp = log_data["timestamp"]
            level = log_data["level"]
            logger = log_data["logger"]
            message = log_data["message"]
            location = f"{log_data['module']}:{log_data['line']}"
            
            return f"{timestamp} {level:8} {logger:20} {location:20} {message}"


class LogConfig(BaseModel):
    """Logging configuration model."""
    
    version: int = 1
    disable_existing_loggers: bool = False
    
    formatters: Dict[str, Dict[str, Any]] = {
        "default": {
            "()": LogFormatter,
        },
        "simple": {
            "format": "%(asctime)s %(levelname)-8s %(name)-20s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    }
    
    handlers: Dict[str, Dict[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "default",
            "filename": "output/logs/arxiv_parsing.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "default",
            "filename": "output/logs/arxiv_parsing_errors.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        }
    }
    
    loggers: Dict[str, Dict[str, Any]] = {
        "": {  # Root logger
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "arxiv_parsing": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "arxiv_parsing.parsers": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "arxiv_parsing.ingestion": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "arxiv_parsing.search": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "aiohttp": {
            "level": "WARNING",
            "handlers": ["console"],
            "propagate": False,
        },
        "urllib3": {
            "level": "WARNING",
            "handlers": ["console"],
            "propagate": False,
        },
    }


def setup_logging(
    level: Optional[str] = None,
    include_file_handlers: bool = True,
    correlation_id: Optional[str] = None
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Log level to use (overrides settings)
        include_file_handlers: Whether to include file handlers
        correlation_id: Correlation ID to add to all log records
    """
    settings = get_settings()
    
    # Determine log level
    log_level = level or settings.log_level
    
    # Create log directories
    log_dir = settings.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log configuration
    config = LogConfig()
    
    # Update log levels
    for logger_config in config.loggers.values():
        logger_config["level"] = log_level
    
    for handler_config in config.handlers.values():
        if handler_config.get("level"):
            handler_config["level"] = log_level
    
    # Remove file handlers if not needed (e.g., in tests)
    if not include_file_handlers:
        for logger_config in config.loggers.values():
            handlers = logger_config.get("handlers", [])
            logger_config["handlers"] = [h for h in handlers if not h.endswith("_file")]
    
    # Apply configuration
    logging.config.dictConfig(config.dict())
    
    # Add correlation ID filter if provided
    if correlation_id:
        add_correlation_filter(correlation_id)
    
    # Log startup message
    logger = logging.getLogger("arxiv_parsing.config")
    logger.info(
        "Logging configured",
        extra={
            "extra_log_level": log_level,
            "extra_log_format": settings.log_format,
            "extra_environment": settings.environment,
            "extra_include_file_handlers": include_file_handlers,
        }
    )


def add_correlation_filter(correlation_id: str) -> None:
    """Add correlation ID to all log records."""
    
    class CorrelationFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.correlation_id = correlation_id
            return True
    
    # Add filter to all handlers
    correlation_filter = CorrelationFilter()
    for handler in logging.root.handlers:
        handler.addFilter(correlation_filter)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"arxiv_parsing.{name}")


class LogContext:
    """Context manager for adding extra fields to log records."""
    
    def __init__(self, **kwargs: Any):
        """Initialize with extra fields to add to log records."""
        self.extra_fields = kwargs
        self.original_factory = logging.getLogRecordFactory()
    
    def __enter__(self) -> "LogContext":
        """Enter context and modify log record factory."""
        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = self.original_factory(*args, **kwargs)
            for key, value in self.extra_fields.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and restore original log record factory."""
        logging.setLogRecordFactory(self.original_factory)


# Convenience functions for common log contexts
def log_with_paper_id(paper_id: str) -> LogContext:
    """Add paper ID to log records."""
    return LogContext(paper_id=paper_id)


def log_with_parser(parser_type: str) -> LogContext:
    """Add parser type to log records."""
    return LogContext(parser_type=parser_type)


def log_with_timing(duration_ms: float) -> LogContext:
    """Add timing information to log records."""
    return LogContext(duration=duration_ms)


def log_with_memory(memory_mb: float) -> LogContext:
    """Add memory usage to log records."""
    return LogContext(memory_usage=memory_mb)