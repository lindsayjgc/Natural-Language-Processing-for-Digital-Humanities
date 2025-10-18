# NLP Document Library API

A FastAPI backend for uploading, processing, and managing documents with NLP analysis.

## Quick Start (Test Version)

### 1. Install Dependencies
The project uses Python 3.11 to avoid compilation issues with scientific computing libraries:

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv venv311
source venv311/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Test API (No MongoDB Required)
```bash
cd services/api
../../venv311/bin/uvicorn test_api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## Full Setup (With MongoDB)

### 1. Install MongoDB
```bash
# macOS with Homebrew
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

### 2. Environment Configuration
Copy the environment template:
```bash
cp env.example .env
```

Edit `.env` if needed (defaults work for local development):
```
MONGODB_URI=mongodb://localhost:27017
DATABASE_NAME=nlp_library
```

### 3. Run the Full API
```bash
cd services/api
../../venv311/bin/uvicorn api:app --reload --port 8000
```

## API Versions

- **`test_api.py`** - Mock version for testing without MongoDB
- **`api.py`** - Full version with MongoDB persistence and real NLP processing

## API Endpoints

### 1. Get User Documents
**GET** `/documents/{user_id}`

Returns all documents for a user.

**Response:**
```json
{
  "user_id": "user123",
  "documents": [
    {
      "_id": "item_id",
      "filename": "document.txt",
      "uploaded_at": "2024-01-01T12:00:00Z",
      "status": "completed",
      "error": null
    }
  ]
}
```

### 2. Upload Document
**POST** `/documents/upload`

Upload and process a document with NLP analysis.

**Form Data:**
- `user_id`: User identifier (string)
- `file`: Document file (PDF, DOCX, TXT, etc.)

**Response:**
```json
{
  "library_item_id": "item_id",
  "filename": "document.txt",
  "processing_status": "completed",
  "stats": {
    "vocab_size": 1234,
    "token_count": 5678,
    "type_token_ratio": 0.217,
    "doc_sentiment": {...},
    "sentiment_method": "transformer"
  }
}
```

### 3. Get Document Details
**GET** `/documents/{user_id}/{item_id}`

Get a specific document with full NLP stats.

**Response:**
```json
{
  "_id": "item_id",
  "user_id": "user123",
  "filename": "document.txt",
  "uploaded_at": "2024-01-01T12:00:00Z",
  "status": "completed",
  "stats": {
    "vocab_size": 1234,
    "token_count": 5678,
    "type_token_ratio": 0.217,
    "doc_sentiment": {...},
    "sentiment_method": "transformer"
  }
}
```

## Data Structure

### Library Items Collection
```python
{
  "user_id": str,
  "filename": str,
  "uploaded_at": datetime,
  "stats_id": ObjectId,  # reference to document_stats
  "status": "completed" | "failed",
  "error": str | None
}
```

### Document Stats Collection
```python
{
  "vocab_size": int,
  "token_count": int,
  "type_token_ratio": float,
  "doc_sentiment": dict,
  "sentiment_method": str,
  "file": str  # original filename
}
```

## Error Handling

- If document processing fails, the item is saved with `status: "failed"` and error details
- All endpoints return appropriate HTTP status codes
- Database connection is tested on startup

## Testing

### Run All Tests
```bash
# From project root
./run_tests.py
```

### Run Specific Test Types
```bash
# Unit tests only
./venv311/bin/python -m pytest tests/unit/ -v

# Integration tests only
./venv311/bin/python -m pytest tests/integration/ -v

# All tests
./venv311/bin/python -m pytest tests/ -v
```

### Test Coverage
- **Unit Tests**: Test individual API endpoints with mocked dependencies
- **Integration Tests**: Test full API workflow with running server
- **Error Handling**: Test validation and error responses

## Development

The API uses:
- **FastAPI** for the web framework
- **Motor** for async MongoDB operations
- **Existing NLP pipeline** from `services/nlp/analyze_texts.py`

No authentication is implemented - user identification is via simple string IDs.
