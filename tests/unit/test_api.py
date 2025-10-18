"""
Unit tests for the NLP Document Library API
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.api.test_api import app

client = TestClient(app)


class TestAPI:
    """Test cases for the API endpoints"""

    def test_root_endpoint(self):
        """Test the root endpoint returns correct response"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "NLP Document Library API - Test Version"
        assert data["version"] == "1.0.0"

    def test_get_documents(self):
        """Test getting documents for a user"""
        response = client.get("/documents/user123")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user123"
        assert "documents" in data
        assert len(data["documents"]) == 1
        assert data["documents"][0]["_id"] == "mock_id_1"
        assert data["documents"][0]["filename"] == "sample_document.txt"
        assert data["documents"][0]["status"] == "completed"

    def test_get_specific_document(self):
        """Test getting a specific document"""
        response = client.get("/documents/user123/mock_id_1")
        assert response.status_code == 200
        data = response.json()
        assert data["_id"] == "mock_id_1"
        assert data["user_id"] == "user123"
        assert data["filename"] == "sample_document.txt"
        assert data["status"] == "completed"
        assert "stats" in data
        assert data["stats"]["vocab_size"] == 1234
        assert data["stats"]["token_count"] == 5678
        assert data["stats"]["type_token_ratio"] == 0.217

    def test_upload_document(self):
        """Test uploading a document"""
        # Create a test file
        test_content = "This is a test document for NLP processing."

        response = client.post(
            "/documents/upload",
            data={"user_id": "user123"},
            files={"file": ("test.txt", test_content, "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["library_item_id"] == "mock_item_id"
        assert data["filename"] == "test.txt"
        assert data["processing_status"] == "completed"
        assert "stats" in data
        assert data["stats"]["vocab_size"] == 1234

    def test_upload_document_no_file(self):
        """Test uploading without a file should return 422 (validation error)"""
        response = client.post(
            "/documents/upload",
            data={"user_id": "user123"},
            files={"file": ("", "", "text/plain")},  # Empty filename
        )

        assert response.status_code == 422  # FastAPI validation error

    def test_upload_document_no_user_id(self):
        """Test uploading without user_id should return 422"""
        test_content = "This is a test document."

        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", test_content, "text/plain")},
        )

        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__])
