# Task Completion Checklist

## Code Quality Checks
```bash
# 1. Format code
uv run ruff format .

# 2. Lint code  
uv run ruff check .

# 3. Type checking
uv run mypy src/

# 4. Run tests
uv run pytest tests/ -v
```

## Before Committing
1. **Run all quality checks** listed above
2. **Ensure tests pass** - both unit and integration tests
3. **Update documentation** if APIs or behavior changed
4. **Check imports** - verify all required dependencies are in pyproject.toml
5. **Validate async patterns** - ensure proper async/await usage

## Integration Testing
- Test with real arXiv papers when modifying parsers
- Validate Google Search API integration if search components changed
- Performance testing for significant changes to core parsing logic

## Environment Variables
- Ensure `.env` file is configured for development
- Test with both HTML and PDF parsing scenarios
- Validate error handling with malformed inputs

## Documentation Updates
- Update ARCHITECTURE.md for structural changes
- Add docstrings for new public methods
- Update CLI help text if command-line interface changed