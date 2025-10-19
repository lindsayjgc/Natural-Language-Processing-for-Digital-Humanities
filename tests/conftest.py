"""
Pytest configuration and fixtures for the NLP Document Library tests
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from bson import ObjectId
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.api.database import (
    get_user_documents,
    create_document,
    update_document,
    save_document_stats,
    get_document,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_documents():
    """Mock documents data for testing"""
    return [
        {
            "_id": "mock_id_1",
            "user_id": "user123",
            "filename": "sample_document.txt",
            "uploaded_at": datetime(2024, 1, 1, 12, 0, 0),
            "status": "completed",
            "stats_id": "stats_id_1",
            "error": None,
        }
    ]


@pytest.fixture
def mock_document_stats():
    """Mock document stats data for testing"""
    return {
        "_id": "stats_id_1",
        "vocab_size": 1234,
        "token_count": 5678,
        "type_token_ratio": 0.217,
        "sentiment_score": 0.5,
        "readability_score": 65.2,
    }


@pytest.fixture
def mock_empty_documents():
    """Mock empty documents list for testing"""
    return []


@pytest.fixture
def mock_database_operations(mock_documents, mock_document_stats, mock_empty_documents):
    """Mock all database operations"""

    async def mock_get_user_documents(user_id: str):
        if user_id == "user_with_no_docs":
            return mock_empty_documents
        return mock_documents

    async def mock_create_document(user_id: str, filename: str):
        return "mock_item_id"

    async def mock_update_document(
        document_id: str,
        stats_id: str = None,
        status: str = "completed",
        error: str = None,
    ):
        return True

    async def mock_save_document_stats(stats_data: dict):
        return "mock_stats_id"

    async def mock_get_document(document_id: str, user_id: str):
        if document_id == "mock_id_1" and user_id == "user123":
            doc = mock_documents[0].copy()
            doc["stats"] = mock_document_stats
            return doc
        return None

    # Patch the database functions
    with (
        patch(
            "services.api.database.get_user_documents",
            side_effect=mock_get_user_documents,
        ),
        patch(
            "services.api.database.create_document", side_effect=mock_create_document
        ),
        patch(
            "services.api.database.update_document", side_effect=mock_update_document
        ),
        patch(
            "services.api.database.save_document_stats",
            side_effect=mock_save_document_stats,
        ),
        patch("services.api.database.get_document", side_effect=mock_get_document),
    ):
        yield


@pytest.fixture
def mock_nlp_processing():
    """Mock NLP processing functions"""

    def mock_process_path(ipath, outdir, from_raw=False):
        return {
            "vocab_size": 1234,
            "token_count": 5678,
            "type_token_ratio": 0.217,
            "sentiment_score": 0.5,
            "readability_score": 65.2,
            "top_keywords": ["test", "document", "nlp"],
            "pos_counts": {"NOUN": 100, "VERB": 50, "ADJ": 30},
        }

    with patch("services.api.api.process_path", side_effect=mock_process_path):
        yield


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing"""

    class MockUploadFile:
        def __init__(self, filename: str, content: bytes):
            self.filename = filename
            self.content = content

        async def read(self):
            return self.content

    return MockUploadFile
