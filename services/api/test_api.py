# Test API without MongoDB dependency
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import json

app = FastAPI(title="NLP Document Library API - Test Version", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "NLP Document Library API - Test Version", "version": "1.0.0"}


@app.get("/documents/{user_id}")
async def get_documents(user_id: str):
    """Get all documents for a user (mock data)"""
    return {
        "user_id": user_id,
        "documents": [
            {
                "_id": "mock_id_1",
                "filename": "sample_document.txt",
                "uploaded_at": "2024-01-01T12:00:00Z",
                "status": "completed",
                "error": None,
            }
        ],
    }


@app.post("/documents/upload")
async def upload_document(user_id: str = Form(...), file: UploadFile = File(...)):
    """Upload and process a document (mock processing)"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Mock processing results
    mock_stats = {
        "vocab_size": 1234,
        "token_count": 5678,
        "type_token_ratio": 0.217,
        "doc_sentiment": {"joy": 0.3, "sadness": 0.1, "anger": 0.05},
        "sentiment_method": "mock_transformer",
    }

    return {
        "library_item_id": "mock_item_id",
        "filename": file.filename,
        "processing_status": "completed",
        "stats": mock_stats,
    }


@app.get("/documents/{user_id}/{item_id}")
async def get_document(user_id: str, item_id: str):
    """Get a specific document with its stats (mock data)"""
    return {
        "_id": item_id,
        "user_id": user_id,
        "filename": "sample_document.txt",
        "uploaded_at": "2024-01-01T12:00:00Z",
        "status": "completed",
        "stats": {
            "vocab_size": 1234,
            "token_count": 5678,
            "type_token_ratio": 0.217,
            "doc_sentiment": {"joy": 0.3, "sadness": 0.1, "anger": 0.05},
            "sentiment_method": "mock_transformer",
        },
    }
