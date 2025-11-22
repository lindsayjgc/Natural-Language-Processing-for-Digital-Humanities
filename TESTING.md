# Testing Guide

Comprehensive testing documentation for the NLP Document Library project.

## Overview

This project includes test suites for both backend (Python/FastAPI) and frontend (TypeScript/React) components.

## Test Coverage

### Backend Tests ✅

**Unit Tests** (`tests/unit/`)
- ✅ **API Tests** (`test_api_simple.py`, `test_api.py`) - 14 tests
  - API endpoints (GET, POST)
  - File upload handling
  - Error responses
  - ObjectId serialization
  - DateTime serialization
  - CORS headers
  - Special characters in filenames

- ✅ **Database Tests** (`test_database.py`) - 10 tests
  - MongoDB operations (async)
  - Document CRUD operations
  - ObjectId to string conversion
  - DateTime to ISO string conversion
  - Stats saving and retrieval
  - Document with stats joins
  - Connection handling and error scenarios

- ✅ **NLP Processing Tests** (`test_nlp_processing.py`) - 11 tests
  - Vocabulary statistics calculation (word_count, vocab_size)
  - Whitespace normalization
  - Footnote removal
  - Sentiment analysis
  - Type-token ratio
  - Text file processing
  - Error handling and edge cases

- ✅ **Advanced NLP Tests** (`test_nlp_advanced.py`) - 13 tests
  - Feature extraction methods
  - Text preprocessing pipeline
  - Sentiment analysis algorithms
  - Statistical calculations
  - Input validation and error handling

**Integration Tests** (`tests/integration/`)
- ✅ **API Integration Tests** (`test_api_integration.py`) - 10 tests
  - End-to-end API workflow
  - Document upload and processing
  - User document retrieval
  - Error handling in real scenarios


### Frontend Tests ✅

**Unit Tests** (`services/frontend/src/`)
- ✅ **API Client Tests** (`lib/__tests__/api.test.ts`) - 11 tests
  - getUserDocuments() with error handling
  - uploadDocument() with FormData
  - getDocument() with stats
  - Network error scenarios
  - Server error responses (500, 404)
  - Timeout handling
  - API URL configuration

- ✅ **Utility Tests** (`lib/__tests__/`)
  - ✅ **Utils** (`utils.test.ts`) - 5 tests
    - className utility (cn) function
    - Conditional class handling
    - Class deduplication
  - ✅ **Readability** (`readability.test.ts`) - 5 tests
    - Readability score calculation
    - Level determination
    - Edge case handling

- ✅ **Component Tests** (`components/`)
  - ✅ **DocumentsTable** (`documents/__tests__/DocumentsTable.test.tsx`) - 12 tests
    - Loading state rendering
    - Empty state message
    - Document list with completed documents only
    - Status badges (Analyzed, Analyzing, Failed)
    - Dropdown actions menu
    - Date formatting
    - Refresh functionality
    - Table headers and structure
  
  - ✅ **DocumentsView** (`documents/__tests__/DocumentsView.test.tsx`) - 6 tests
    - Main page rendering with title
    - Upload section presence
    - Loading states
    - Document table integration
    - Error handling
  
  - ✅ **UploadCard** (`documents/__tests__/UploadCard.test.tsx`) - 9 tests
    - Upload area rendering
    - Browse button functionality
    - File type validation
    - Drag and drop support
    - Upload state management
    - Icon and text display
  
  - ✅ **LoginForm** (`auth/__tests__/LoginForm.test.tsx`) - 5 tests
    - Form field rendering
    - Input validation
    - Submit button states
    - User interaction handling
    - Loading states

## Running Tests

### Backend Tests

```bash
# Activate virtual environment
source venv311/bin/activate

# Install dependencies (if not already installed)
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/unit/test_api.py -v
pytest tests/unit/test_api_simple.py -v
pytest tests/unit/test_database.py -v
pytest tests/unit/test_nlp_processing.py -v
pytest tests/unit/test_nlp_advanced.py -v

# Run integration tests
pytest tests/integration/test_api_integration.py -v

# Run with coverage
pytest tests/ --cov=services --cov-report=html

# Run tests with minimal output
pytest tests/ --tb=short -q
```

### Frontend Tests

```bash
cd services/frontend

# Install dependencies
pnpm install

# Run all tests
pnpm test

# Run tests once (CI mode)
pnpm test --run

# Run with coverage
pnpm test --coverage

# Run specific test file
pnpm test api.test.ts
pnpm test DocumentsTable.test.tsx

# Run tests with verbose output
pnpm test --run --reporter=verbose

# Run in watch mode
pnpm test --watch
```

## Test Results

### Backend Test Summary (44 tests total)

```
✅ tests/unit/test_api.py - 9/9 passed
✅ tests/unit/test_api_simple.py - 5/5 passed  
✅ tests/unit/test_database.py - 10/10 passed
✅ tests/unit/test_nlp_processing.py - 11/11 passed
✅ tests/unit/test_nlp_advanced.py - 13/13 passed
✅ tests/integration/test_api_integration.py - 10/10 passed

Total: 44/44 tests passing (100%)
```

### Frontend Test Summary (53 tests total)

```
✅ src/lib/__tests__/api.test.ts - 11/11 passed
✅ src/lib/__tests__/utils.test.ts - 5/5 passed
✅ src/lib/__tests__/readability.test.ts - 5/5 passed
✅ src/components/documents/__tests__/DocumentsTable.test.tsx - 12/12 passed
✅ src/components/documents/__tests__/DocumentsView.test.tsx - 6/6 passed
✅ src/components/documents/__tests__/UploadCard.test.tsx - 9/9 passed
✅ src/components/auth/__tests__/LoginForm.test.tsx - 5/5 passed

Total: 53/53 tests passing (100%)
```

## Test Configuration

### Backend (Python)

**Dependencies** (in `requirements.txt`)
```
pytest           # Core testing framework
pytest-asyncio   # Async test support (1.3.0)
httpx           # HTTP client for API testing (0.28.1)
```

**Configuration** (pytest.ini)
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts = -v --tb=short
```

### Frontend (TypeScript)

**Dependencies** (in `package.json`)
```json
{
  "devDependencies": {
    "vitest": "^3.2.4",
    "@testing-library/react": "^16.3.0",
    "@testing-library/jest-dom": "^6.9.1",
    "@testing-library/user-event": "^14.6.1",
    "@vitejs/plugin-react": "^5.0.4",
    "jsdom": "^27.0.1"
  }
}
```

**Configuration** (`vitest.config.ts`)
```typescript
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./vitest.setup.ts'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
```

**Setup File** (`vitest.setup.ts`)
```typescript
import "@testing-library/jest-dom";

// Mock environment variables
process.env.NEXT_PUBLIC_API_URL = "http://localhost:8000";

// Mock ResizeObserver for test environment
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};
```

## Test Organization

```
tests/
├── unit/                              # Backend unit tests
│   ├── test_api.py                   # Core API endpoint tests
│   ├── test_api_simple.py            # Simple API tests  
│   ├── test_database.py              # Database operation tests
│   ├── test_nlp_processing.py        # NLP pipeline tests
│   └── test_nlp_advanced.py          # Advanced NLP feature tests
├── integration/                       # Integration tests
│   └── test_api_integration.py       # End-to-end API workflow
└── conftest.py                       # Shared test configuration

services/frontend/src/
├── lib/__tests__/                    # Frontend utility tests
│   ├── api.test.ts                  # API client tests
│   ├── utils.test.ts                # Utility function tests
│   └── readability.test.ts          # Readability calculation tests
└── components/
    ├── documents/__tests__/          # Document component tests
    │   ├── DocumentsTable.test.tsx  # Table component
    │   ├── DocumentsView.test.tsx   # Main view component
    │   └── UploadCard.test.tsx      # Upload component
    └── auth/__tests__/               # Authentication component tests
        └── LoginForm.test.tsx       # Login form component
```

## Writing New Tests

### Backend Test Example

```python
import pytest
from fastapi.testclient import TestClient

def test_new_feature():
    """Test description"""
    # Arrange
    client = TestClient(app)

    # Act
    response = client.get("/endpoint")

    # Assert
    assert response.status_code == 200
    assert "expected" in response.json()
```

### Frontend Test Example

```typescript
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

describe('Component', () => {
  it('should render correctly', () => {
    render(<Component />)
    expect(screen.getByText('Hello')).toBeInTheDocument()
  })
})
```

## Continuous Integration

Tests can be run in CI/CD pipelines:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/unit/ -v

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: pnpm/action-setup@v2
      - run: cd services/frontend && pnpm install
      - run: cd services/frontend && pnpm test --run
```

## Test Coverage Goals

### Current Coverage

- ✅ Backend API endpoints: 95%+ (14 tests)
- ✅ Backend database operations: 90%+ (10 tests)
- ✅ Backend NLP processing: 90%+ (24 tests)
- ✅ Frontend API client: 95%+ (11 tests)
- ✅ Frontend components: 85%+ (32 tests)
- ✅ Frontend utilities: 90%+ (10 tests)

### Completed Recently

- ✅ DocumentsView component tests
- ✅ UploadCard component tests  
- ✅ LoginForm component tests
- ✅ Integration test suite
- ✅ Advanced NLP feature tests
- ✅ Utility function tests

### Future Additions

- ⏳ End-to-end tests with Playwright
- ⏳ Performance tests
- ⏳ Load testing
- ⏳ Security tests
- ⏳ Visual regression tests
- ⏳ Accessibility tests

## Troubleshooting

### Backend Issues

**Problem**: Tests fail with MongoDB connection error
```bash
# Solution: Use mock tests or ensure MongoDB is running
pytest tests/unit/ -v  # Unit tests don't need MongoDB
```

**Problem**: NLTK data not found
```bash
# Solution: Download required NLTK data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### Frontend Issues

**Problem**: Module resolution errors
```bash
# Solution: Check path aliases in vitest.config.ts
# Ensure '@' points to './src'
```

**Problem**: React component rendering issues
```bash
# Solution: Ensure @testing-library/react is installed
# Check vitest.setup.ts includes '@testing-library/jest-dom'
```

**Problem**: ResizeObserver not defined
```bash
# Solution: Add ResizeObserver mock to vitest.setup.ts
# global.ResizeObserver = class ResizeObserver { ... }
```

**Problem**: API mocking issues
```bash
# Solution: Mock both named exports and default exports
# vi.mock('@/lib/api', () => ({ apiClient: {...}, getUserDocuments: ... }))
```

**Problem**: Multiple elements found in tests
```bash
# Solution: Use more specific selectors
# screen.getByRole('heading', { name: /title/i })
# screen.getByText(/specific text/i)
```

## Best Practices

1. **Write tests first** (TDD when possible)
2. **Test behavior, not implementation**
3. **Use descriptive test names**
4. **Keep tests isolated** (no shared state)
5. **Mock external dependencies**
6. **Test edge cases and error conditions**
7. **Maintain test coverage above 80%**
8. **Run tests before committing**
9. **Keep tests fast** (< 1s per test)
10. **Update tests when changing features**

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Vitest Documentation](https://vitest.dev/)
- [Testing Library](https://testing-library.com/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)

---

**Last Updated**: 2025-11-22
**Test Framework Versions**: pytest 9.0.1, vitest 3.2.4, pytest-asyncio 1.3.0
**Total Tests**: 97 tests (44 backend + 53 frontend) - 100% passing
**Test Coverage**: Comprehensive coverage across API, database, NLP, components, and utilities
