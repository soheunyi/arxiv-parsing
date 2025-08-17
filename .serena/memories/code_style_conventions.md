# Code Style and Conventions

## Python Standards
- **Python Version**: 3.10+ required
- **Line Length**: 88 characters (ruff configuration)
- **Import Sorting**: isort via ruff
- **Code Formatting**: ruff format (replaces black)

## Type Annotations
- **Required**: Full type annotations for all functions and methods
- **Strict Typing**: mypy strict mode enabled
- **Pydantic Models**: Use for data validation and serialization

## Async Patterns
- **Async First**: All I/O operations use async/await
- **HTTP Requests**: Use aiohttp for external API calls
- **File Operations**: Use aiofiles for file I/O
- **Database**: Async SQLAlchemy patterns

## Naming Conventions
- **Classes**: PascalCase (e.g., `ArxivFetcher`, `GrobidClient`)
- **Functions/Variables**: snake_case (e.g., `fetch_paper`, `parse_references`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- **Private Methods**: Leading underscore (`_helper_method`)

## Documentation
- **Docstrings**: Required for all public methods and classes
- **Type Hints**: Comprehensive type annotations
- **Comments**: Explain complex logic and business rules

## Error Handling
- **Graceful Degradation**: Handle external service failures
- **Specific Exceptions**: Use custom exception classes
- **Logging**: Structured logging with appropriate levels
- **Retry Logic**: Implement exponential backoff for network calls

## Testing Patterns
- **Pytest**: Primary testing framework
- **Async Tests**: Use pytest-asyncio for async test functions
- **Mocking**: pytest-mock for external service mocking
- **Markers**: Use test markers (unit, integration, slow, performance)