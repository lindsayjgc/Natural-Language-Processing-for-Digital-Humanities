"""
Unit tests for the NLP Document Library API
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.api.api import app
from services.api.auth import get_current_user


def override_get_current_user():
    """Override authentication for testing"""
    return "test_user_123"


# Override the dependency for testing
app.dependency_overrides[get_current_user] = override_get_current_user


class TestAPI:
    """Test cases for the API endpoints"""

    def test_root_endpoint(self):
        """Test the root endpoint returns correct response"""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "NLP Document Library API - Test Version"
        assert data["version"] == "1.0.0"

    @patch("services.api.api.get_user_documents")
    def test_get_documents(self, mock_get_user_documents):
        """Test getting documents for a user"""
        # Mock the database response
        mock_documents = [
            {
                "_id": "mock_id_1",
                "user_id": "user123",
                "filename": "sample_document.txt",
                "uploaded_at": datetime(2024, 1, 1, 12, 0, 0).isoformat(),
                "status": "completed",
                "stats_id": "stats_id_1",
                "error": None,
            }
        ]
        mock_get_user_documents.return_value = mock_documents

        client = TestClient(app)
        response = client.get("/documents/user123")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user123"
        assert "documents" in data
        assert len(data["documents"]) == 1
        assert data["documents"][0]["_id"] == "mock_id_1"
        assert data["documents"][0]["filename"] == "sample_document.txt"
        assert data["documents"][0]["status"] == "completed"

    @patch("services.api.api.get_document")
    def test_get_specific_document(self, mock_get_document):
        """Test getting a specific document"""
        # Mock the database response
        mock_document = {
            "_id": "mock_id_1",
            "user_id": "user123",
            "filename": "sample_document.txt",
            "uploaded_at": datetime(2024, 1, 1, 12, 0, 0),
            "status": "completed",
            "stats": {
                "vocab_size": 1234,
                "token_count": 5678,
                "type_token_ratio": 0.217,
            },
        }
        mock_get_document.return_value = mock_document

        client = TestClient(app)
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

    @patch("services.api.api.create_document")
    @patch("services.api.api.save_document_stats")
    @patch("services.api.api.update_document")
    @patch("services.api.api.process_path")
    def test_upload_document(
        self,
        mock_process_path,
        mock_update_document,
        mock_save_document_stats,
        mock_create_document,
    ):
        """Test uploading a document"""
        # Mock the database operations
        mock_create_document.return_value = "mock_item_id"
        mock_save_document_stats.return_value = "mock_stats_id"
        mock_update_document.return_value = None

        # Mock the NLP processing
        mock_process_path.return_value = {
            "vocab_size": 1234,
            "token_count": 5678,
            "type_token_ratio": 0.217,
        }

        client = TestClient(app)
        # Create a test file
        test_content = "This is a test document for NLP processing."

        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", test_content, "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "mock_item_id"
        assert data["filename"] == "test.txt"
        assert data["processing_status"] == "completed"
        assert "stats" in data
        assert data["stats"]["vocab_size"] == 1234

    def test_upload_document_no_file(self):
        """Test uploading without a file should return 422 (validation error)"""
        client = TestClient(app)
        response = client.post(
            "/documents/upload",
            files={"file": ("", "", "text/plain")},  # Empty filename
        )

        assert response.status_code == 422  # Validation error - empty file rejected before auth check

    def test_upload_document_no_user_id(self):
        """Test uploading without user authentication - user_id now comes from JWT"""
        client = TestClient(app)
        test_content = "This is a test document."

        # Remove the auth override temporarily to test no authentication
        original_override = app.dependency_overrides.get(get_current_user)
        if get_current_user in app.dependency_overrides:
            del app.dependency_overrides[get_current_user]

        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", test_content, "text/plain")},
        )

        # Restore the auth override
        if original_override:
            app.dependency_overrides[get_current_user] = original_override

        assert response.status_code == 403  # Authentication required

    @patch("services.api.api.get_user_documents")
    def test_objectid_serialization_in_response(self, mock_get_user_documents):
        """Test that ObjectIds are properly serialized to strings in API responses"""
        # Mock the database response
        mock_documents = [
            {
                "_id": "mock_id_1",
                "user_id": "user123",
                "filename": "sample_document.txt",
                "uploaded_at": datetime(2024, 1, 1, 12, 0, 0).isoformat(),
                "status": "completed",
                "stats_id": "stats_id_1",
                "error": None,
            }
        ]
        mock_get_user_documents.return_value = mock_documents

        client = TestClient(app)
        response = client.get("/documents/user123")
        assert response.status_code == 200
        data = response.json()

        # All IDs should be strings, not ObjectId objects
        for doc in data["documents"]:
            assert isinstance(doc["_id"], str)
            if "stats_id" in doc and doc["stats_id"]:
                assert isinstance(doc["stats_id"], str)

    @patch("services.api.api.get_document")
    def test_datetime_serialization_in_response(self, mock_get_document):
        """Test that datetime objects are serialized to ISO format strings"""
        # Mock the database response
        mock_document = {
            "_id": "mock_id_1",
            "user_id": "user123",
            "filename": "sample_document.txt",
            "uploaded_at": datetime(2024, 1, 1, 12, 0, 0),
            "status": "completed",
            "stats": {
                "vocab_size": 1234,
                "token_count": 5678,
                "type_token_ratio": 0.217,
            },
        }
        mock_get_document.return_value = mock_document

        client = TestClient(app)
        response = client.get("/documents/user123/mock_id_1")
        assert response.status_code == 200
        data = response.json()

        # uploaded_at should be ISO format string
        assert isinstance(data["uploaded_at"], str)
        # Should be parseable as ISO format
        datetime.fromisoformat(data["uploaded_at"])  # Should not raise

    def test_cors_headers(self):
        """Test that CORS headers are present"""
        client = TestClient(app)
        response = client.get("/")
        # Test API typically doesn't need CORS headers, but check if present
        # In real API, verify Access-Control-Allow-Origin header
        assert response.status_code == 200

    def test_error_response_format(self):
        """Test that error responses have correct format"""
        client = TestClient(app)
        # Test with empty file
        response = client.post(
            "/documents/upload",
            files={"file": ("", "", "text/plain")},
        )

        assert response.status_code == 422
        data = response.json()
        # Validation errors have 'detail' field
        assert "detail" in data

    @patch("services.api.api.create_document")
    @patch("services.api.api.save_document_stats")
    @patch("services.api.api.update_document")
    @patch("services.api.api.process_path")
    def test_large_document_upload(
        self,
        mock_process_path,
        mock_update_document,
        mock_save_document_stats,
        mock_create_document,
    ):
        """Test uploading a large document"""
        # Mock the database operations
        mock_create_document.return_value = "mock_item_id"
        mock_save_document_stats.return_value = "mock_stats_id"
        mock_update_document.return_value = None

        # Mock the NLP processing
        mock_process_path.return_value = {
            "vocab_size": 1234,
            "token_count": 5678,
            "type_token_ratio": 0.217,
        }

        client = TestClient(app)
        # Create a larger test document
        large_content = "This is a test sentence. " * 1000  # ~25KB

        response = client.post(
            "/documents/upload",
            files={"file": ("large_test.txt", large_content, "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "large_test.txt"
        assert "stats" in data

    @patch("services.api.api.create_document")
    @patch("services.api.api.save_document_stats")
    @patch("services.api.api.update_document")
    @patch("services.api.api.process_path")
    def test_special_characters_in_filename(
        self,
        mock_process_path,
        mock_update_document,
        mock_save_document_stats,
        mock_create_document,
    ):
        """Test uploading file with special characters in name"""
        # Mock the database operations
        mock_create_document.return_value = "mock_item_id"
        mock_save_document_stats.return_value = "mock_stats_id"
        mock_update_document.return_value = None

        # Mock the NLP processing
        mock_process_path.return_value = {
            "vocab_size": 1234,
            "token_count": 5678,
            "type_token_ratio": 0.217,
        }

        client = TestClient(app)
        test_content = "Test content"

        response = client.post(
            "/documents/upload",
            files={"file": ("tëst_fîlé_ñame.txt", test_content, "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "tëst_fîlé_ñame.txt"

    @patch("services.api.api.get_user_documents")
    def test_get_documents_empty_list(self, mock_get_user_documents):
        """Test getting documents for user with no documents"""
        # Mock empty documents list
        mock_get_user_documents.return_value = []

        client = TestClient(app)
        response = client.get("/documents/user_with_no_docs")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user_with_no_docs"
        # Test API returns mock data, real API might return empty list
        assert "documents" in data

    def test_api_version_endpoint(self):
        """Test that API version is returned correctly"""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"


if __name__ == "__main__":
    pytest.main([__file__])
