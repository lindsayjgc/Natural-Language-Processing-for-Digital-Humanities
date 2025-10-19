# API Documentation

Complete reference for the NLP Document Library REST API.

## Base URL

**Development**: `http://localhost:8000`
**Production**: [Your production URL]

## Authentication

Currently, the API does not implement authentication. User identification is done via simple string `user_id` parameters.

**Future**: Authentication will be added using JWT tokens or OAuth2.

## Endpoints

### Health Check

#### `GET /`

Check if the API server is running.

**Response**
```json
{
  "message": "NLP Document Library API",
  "version": "1.0.0"
}
```

**Example**
```bash
curl http://localhost:8000/
```

---

### List User Documents

#### `GET /documents/{user_id}`

Retrieve all documents for a specific user, sorted by upload date (newest first).

**Parameters**
- `user_id` (path, string, required): User identifier

**Response**
```json
{
  "user_id": "demo_user",
  "documents": [
    {
      "_id": "507f1f77bcf86cd799439011",
      "user_id": "demo_user",
      "filename": "example.txt",
      "uploaded_at": "2025-10-19T20:11:24.923000",
      "status": "completed",
      "error": null,
      "stats_id": "507f191e810c19729de860ea"
    }
  ]
}
```

**Status Codes**
- `200 OK`: Successfully retrieved documents
- `500 Internal Server Error`: Database or server error

**Document Status Values**
- `processing`: Document is currently being analyzed
- `completed`: Analysis finished successfully
- `failed`: Processing failed (see `error` field)

**Example**
```bash
curl http://localhost:8000/documents/demo_user
```

---

### Upload Document

#### `POST /documents/upload`

Upload a document for NLP analysis. The document will be processed asynchronously and results stored in the database.

**Content-Type**: `multipart/form-data`

**Form Parameters**
- `user_id` (string, required): User identifier
- `file` (file, required): Document file to upload

**Supported File Types**
- Plain text: `.txt`, `.md`, `.rst`
- Word documents: `.docx` (legacy `.doc` requires textract)
- Rich text: `.rtf`
- PDF: `.pdf` (requires textract for complex PDFs)

**Response**
```json
{
  "library_item_id": "507f1f77bcf86cd799439011",
  "filename": "example.txt",
  "processing_status": "completed",
  "stats": {
    "vocab_size": 234,
    "token_count": 1024,
    "type_token_ratio": 0.2285,
    "doc_sentiment": {
      "neutral": 0.7234,
      "joy": 0.1523,
      "sadness": 0.0843,
      "anger": 0.0234,
      "fear": 0.0098,
      "disgust": 0.0045,
      "surprise": 0.0023
    },
    "sentiment_method": "transformers_emotion"
  }
}
```

**Status Codes**
- `200 OK`: Document uploaded and processed successfully
- `400 Bad Request`: Invalid file or missing parameters
- `500 Internal Server Error`: Processing failed

**Example**
```bash
# Upload a text file
curl -X POST \
  -F "user_id=demo_user" \
  -F "file=@document.txt" \
  http://localhost:8000/documents/upload

# Upload a PDF
curl -X POST \
  -F "user_id=demo_user" \
  -F "file=@research_paper.pdf" \
  http://localhost:8000/documents/upload
```

**Processing Pipeline**
1. File uploaded and saved temporarily
2. Text extracted based on file format
3. Text normalized (whitespace, footnotes removed)
4. NLP analysis performed:
   - Tokenization and lemmatization
   - Vocabulary statistics calculated
   - Sentiment analysis using transformer models
   - POS tagging (if requested)
5. Results saved to database
6. Temporary file cleaned up

**Error Handling**
If processing fails:
- Document status set to `failed`
- Error message stored in `error` field
- Partial results (if any) not saved

---

### Get Document Details

#### `GET /documents/{user_id}/{item_id}`

Retrieve a specific document with full NLP statistics.

**Parameters**
- `user_id` (path, string, required): User identifier
- `item_id` (path, string, required): Document ID (MongoDB ObjectId)

**Response**
```json
{
  "_id": "507f1f77bcf86cd799439011",
  "user_id": "demo_user",
  "filename": "example.txt",
  "uploaded_at": "2025-10-19T20:11:24.923000",
  "status": "completed",
  "error": null,
  "stats_id": "507f191e810c19729de860ea",
  "stats": {
    "_id": "507f191e810c19729de860ea",
    "vocab_size": 234,
    "token_count": 1024,
    "type_token_ratio": 0.2285,
    "doc_sentiment": {
      "neutral": 0.7234,
      "joy": 0.1523,
      "sadness": 0.0843,
      "anger": 0.0234,
      "fear": 0.0098,
      "disgust": 0.0045,
      "surprise": 0.0023
    },
    "sentiment_method": "transformers_emotion",
    "file": "/path/to/original/file"
  }
}
```

**Status Codes**
- `200 OK`: Document found and returned
- `404 Not Found`: Document doesn't exist or belongs to different user
- `500 Internal Server Error`: Database or server error

**Example**
```bash
curl http://localhost:8000/documents/demo_user/507f1f77bcf86cd799439011
```

---

## Data Models

### Document (Library Item)

Represents an uploaded document.

```typescript
interface Document {
  _id: string;              // Unique document ID
  user_id: string;          // Owner user ID
  filename: string;         // Original filename
  uploaded_at: string;      // ISO 8601 timestamp
  status: string;           // "processing" | "completed" | "failed"
  error: string | null;     // Error message if failed
  stats_id?: string;        // Reference to statistics
  stats?: DocumentStats;    // Full statistics (if included)
}
```

### Document Statistics

NLP analysis results for a document.

```typescript
interface DocumentStats {
  _id: string;                    // Unique stats ID
  vocab_size: number;             // Number of unique words
  token_count: number;            // Total number of tokens
  type_token_ratio: number;       // Lexical diversity (0-1)
  doc_sentiment: {                // Emotion scores (0-1)
    neutral: number;
    joy: number;
    sadness: number;
    anger: number;
    fear: number;
    disgust: number;
    surprise: number;
  };
  sentiment_method: string;       // Analysis method used
  file: string;                   // Original file path
}
```

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

- `400 Bad Request`: Invalid input parameters
- `404 Not Found`: Resource doesn't exist
- `500 Internal Server Error`: Server-side error
- `503 Service Unavailable`: Database connection failed

## Rate Limiting

**Current**: No rate limiting implemented
**Future**: Will implement rate limiting for production deployment

## CORS

The API allows requests from all origins (`*`) for development.

**Production**: Configure specific allowed origins in `api.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## NLP Processing Details

### Sentiment Analysis

**Primary Method**: Transformer-based emotion classification
- Model: `SamLowe/roberta-base-go_emotions`
- 7 emotion categories: neutral, joy, sadness, anger, fear, disgust, surprise
- Confidence scores for each emotion (sum to 1.0)

**Fallback Method**: VADER sentiment analysis
- Used if transformer model unavailable
- Provides compound, positive, negative, neutral scores

### Vocabulary Statistics

- **Vocab Size**: Count of unique lemmatized words
- **Token Count**: Total number of tokens after preprocessing
- **Type-Token Ratio**: Vocab size / token count (lexical diversity measure)
  - Higher ratio = more diverse vocabulary
  - Lower ratio = more repetitive text

### Text Preprocessing

1. **Format Detection**: Auto-detect file type
2. **Text Extraction**:
   - PDF: Extract text (or OCR if needed)
   - DOCX: Parse document XML
   - RTF: Decode rich text
   - TXT: Direct read with encoding detection
3. **Normalization**:
   - Remove Project Gutenberg boilerplate
   - Remove footnotes
   - Normalize whitespace
   - Unwrap soft line breaks
4. **Tokenization**: spaCy or NLTK tokenizer
5. **Lemmatization**: Reduce words to base forms
6. **Stop word removal**: Filter common words (optional)

## Testing

### Test API

A test version of the API is available that doesn't require MongoDB:

```bash
cd services/api
uvicorn test_api:app --reload --port 8001
```

The test API returns mock data and is useful for:
- Frontend development without backend
- Testing API integration
- Demos without database setup

### Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Use these interfaces to:
- Explore all endpoints
- Test API calls interactively
- View request/response schemas
- Download OpenAPI specification

## Performance Considerations

### Upload Processing Time

Processing time depends on:
- Document size (longer texts take more time)
- File format (PDFs slower than TXT)
- NLP model loading (first request is slower)

**Typical Times**:
- Small text (< 1000 words): 2-5 seconds
- Medium text (1000-5000 words): 5-15 seconds
- Large text (> 5000 words): 15-60 seconds

### Optimization Tips

1. **Batch Processing**: For multiple documents, process in parallel
2. **Caching**: Results are cached in database (no re-processing)
3. **Model Loading**: Keep transformer model loaded in memory
4. **Max Length**: Very long documents (>10,000 words) may timeout

### Database Queries

- Document lists use indexed queries on `user_id`
- Uploads create two database writes (item + stats)
- Detail queries join item with stats collection

## Future Enhancements

Planned features for future versions:

- [ ] User authentication (JWT/OAuth2)
- [ ] Batch document upload
- [ ] Document comparison/similarity
- [ ] Custom analysis parameters
- [ ] Webhook notifications
- [ ] Rate limiting
- [ ] API versioning
- [ ] GraphQL endpoint
- [ ] WebSocket for real-time updates
- [ ] Export results (CSV/JSON)
- [ ] Document tagging/categorization

## Support

For issues or questions:
- Check [GitHub Issues](your-repo/issues)
- Review [Setup Guide](../../SETUP.md)
- Contact team members

---

**API Version**: 1.0.0
**Last Updated**: 2025-10-19
**Maintained By**: NLP Digital Humanities Team
