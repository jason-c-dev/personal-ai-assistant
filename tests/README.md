# Agent Testing Suite

This directory contains comprehensive tests for the Personal AI Assistant agent system, covering all major components and integration scenarios.

## Test Structure

### Unit Tests (src/agent/*.test.py)
- `agent_config.test.py` - Configuration management tests
- `core_agent.test.py` - Core agent functionality tests  
- `mcp_client.test.py` - MCP client integration tests
- `strands_mcp_tools.test.py` - Strands tool wrapper tests

### Integration Tests (tests/integration/)
- `test_agent_integration.py` - End-to-end system integration tests

## Running Tests

### Run All Tests
```bash
python -m pytest
```

### Run Specific Test Categories
```bash
# Unit tests only
python -m pytest src/ -m unit

# Integration tests only  
python -m pytest tests/integration/ -m integration

# Memory system tests
python -m pytest -m memory

# MCP integration tests
python -m pytest -m mcp

# Agent functionality tests
python -m pytest -m agent
```

### Run Tests with Coverage
```bash
python -m pytest --cov=src --cov-report=html
```

### Run Specific Test Files
```bash
# Test agent configuration
python -m pytest src/agent/agent_config.test.py

# Test core agent
python -m pytest src/agent/core_agent.test.py

# Test MCP client
python -m pytest src/agent/mcp_client.test.py

# Test integration scenarios
python -m pytest tests/integration/test_agent_integration.py
```

## Test Categories and Markers

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests across components
- `@pytest.mark.memory` - Tests related to memory system
- `@pytest.mark.mcp` - Tests related to MCP integration
- `@pytest.mark.agent` - Tests related to core agent functionality
- `@pytest.mark.slow` - Tests that take longer to run

## Test Configuration

The test suite is configured via `pytest.ini` in the project root. Key settings:

- Test discovery patterns: `*.test.py` and `test_*.py`
- Test paths: `src/` and `tests/`
- Markers for categorizing tests
- Warning filters for clean output

## Mock Strategy

The tests use extensive mocking to:

1. **Isolate components** - Each unit test focuses on a single component
2. **Avoid external dependencies** - Mock AWS Bedrock, file I/O, network calls
3. **Simulate error conditions** - Test error handling and resilience
4. **Speed up execution** - Avoid slow operations in unit tests

### Key Mocking Patterns

- **Strands Agent**: Mock the underlying ML model for predictable responses
- **MCP Servers**: Mock server communication for reliable testing
- **Memory System**: Use temporary directories for isolated testing
- **File Operations**: Mock file I/O to avoid filesystem dependencies

## Test Data

Tests use pytest fixtures for consistent test data:

- `temp_memory_dir` - Temporary memory directory for isolation
- `integration_config` - Standard configuration for integration tests
- `mock_mcp_client` - Mocked MCP client with controllable behavior

## Writing New Tests

### Unit Test Guidelines

1. **Test one component at a time**
2. **Mock external dependencies**
3. **Use descriptive test names**
4. **Test both success and failure scenarios**
5. **Verify expected calls to mocked components**

### Integration Test Guidelines

1. **Test realistic workflows**
2. **Use minimal mocking for key components**
3. **Verify end-to-end functionality**
4. **Test error recovery scenarios**

### Example Test Structure

```python
class TestComponentName:
    """Test cases for ComponentName"""
    
    def test_successful_operation(self, fixture):
        """Test successful operation scenario"""
        # Setup
        component = ComponentName(fixture)
        
        # Action
        result = component.operation()
        
        # Assert
        assert result == expected_value
    
    def test_error_handling(self, fixture):
        """Test error handling scenario"""
        # Setup with error condition
        component = ComponentName(fixture)
        
        # Action & Assert
        with pytest.raises(ExpectedError):
            component.operation_that_fails()
```

## Continuous Integration

The test suite is designed to run in CI/CD environments:

- All tests should pass consistently
- Tests are isolated and don't depend on external state
- Mock external services to avoid flaky tests
- Use temporary directories for file-based tests

## Debugging Tests

### Common Issues

1. **Import errors** - Check Python path and module structure
2. **Async test failures** - Ensure proper async/await usage
3. **Mock configuration** - Verify mock setup matches actual usage
4. **File permissions** - Use temporary directories for file tests

### Debug Commands

```bash
# Run with verbose output
python -m pytest -v

# Run with debugging output
python -m pytest -s --log-cli-level=DEBUG

# Run single test with debugging
python -m pytest src/agent/core_agent.test.py::TestAgentInitialization::test_agent_creation -v -s
```

## Contributing

When adding new functionality:

1. **Write tests first** (TDD approach recommended)
2. **Maintain test coverage** above 80%
3. **Add appropriate markers** for test categorization
4. **Update this documentation** for significant changes
5. **Ensure tests pass** before submitting changes 