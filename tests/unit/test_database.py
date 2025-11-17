"""
Unit tests for database operations
Tests MongoDB interaction with mocked database
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from bson import ObjectId
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestDatabaseOperations:
    """Test MongoDB database operations"""

    @pytest.mark.asyncio
    async def test_get_user_documents(self):
        """Test retrieving documents for a user"""
        from services.api.database import get_user_documents

        # Mock cursor and documents
        mock_doc1 = {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "user_id": "test_user",
            "filename": "test.txt",
            "uploaded_at": datetime(2025, 1, 15, 10, 30, 0),
            "status": "completed",
            "error": None,
            "stats_id": ObjectId("507f191e810c19729de860ea"),
        }

        mock_doc2 = {
            "_id": ObjectId("507f1f77bcf86cd799439012"),
            "user_id": "test_user",
            "filename": "test2.txt",
            "uploaded_at": datetime(2025, 1, 16, 11, 30, 0),
            "status": "processing",
            "error": None,
        }

        # Create async iterator mock
        async def mock_cursor():
            for doc in [mock_doc2, mock_doc1]:  # Reverse order (sorted by date)
                yield doc

        mock_collection = MagicMock()
        mock_find = MagicMock()
        mock_find.sort.return_value = mock_cursor()
        mock_collection.find.return_value = mock_find

        with patch("services.api.database.get_documents", return_value=mock_collection):
            result = await get_user_documents("test_user")

        # Verify results
        assert len(result) == 2
        assert result[0]["_id"] == "507f1f77bcf86cd799439012"  # Converted to string
        assert result[0]["uploaded_at"] == "2025-01-16T11:30:00"  # ISO format
        assert (
            result[1]["stats_id"] == "507f191e810c19729de860ea"
        )  # Converted to string

    @pytest.mark.asyncio
    async def test_objectid_serialization(self):
        """Test that ObjectIds are converted to strings"""
        from services.api.database import get_user_documents

        mock_doc = {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "user_id": "test_user",
            "filename": "test.txt",
            "uploaded_at": datetime(2025, 1, 15, 10, 30, 0),
            "status": "completed",
            "stats_id": ObjectId("507f191e810c19729de860ea"),
        }

        async def mock_cursor():
            yield mock_doc

        mock_collection = MagicMock()
        mock_find = MagicMock()
        mock_find.sort.return_value = mock_cursor()
        mock_collection.find.return_value = mock_find

        with patch("services.api.database.get_documents", return_value=mock_collection):
            result = await get_user_documents("test_user")

        # All ObjectIds should be strings
        assert isinstance(result[0]["_id"], str)
        assert isinstance(result[0]["stats_id"], str)
        assert result[0]["_id"] == "507f1f77bcf86cd799439011"

    @pytest.mark.asyncio
    async def test_datetime_serialization(self):
        """Test that datetime objects are converted to ISO format strings"""
        from services.api.database import get_user_documents

        test_datetime = datetime(2025, 1, 15, 10, 30, 45)
        mock_doc = {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "user_id": "test_user",
            "filename": "test.txt",
            "uploaded_at": test_datetime,
            "status": "completed",
        }

        async def mock_cursor():
            yield mock_doc

        mock_collection = MagicMock()
        mock_find = MagicMock()
        mock_find.sort.return_value = mock_cursor()
        mock_collection.find.return_value = mock_find

        with patch("services.api.database.get_documents", return_value=mock_collection):
            result = await get_user_documents("test_user")

        # DateTime should be ISO format string
        assert isinstance(result[0]["uploaded_at"], str)
        assert result[0]["uploaded_at"] == "2025-01-15T10:30:45"

    @pytest.mark.asyncio
    async def test_create_document(self):
        """Test creating a new document"""
        from services.api.database import create_document

        mock_result = MagicMock()
        mock_result.inserted_id = ObjectId("507f1f77bcf86cd799439011")

        mock_collection = AsyncMock()
        mock_collection.insert_one.return_value = mock_result

        with patch("services.api.database.get_documents", return_value=mock_collection):
            result = await create_document("test_user", "test.txt")

        # Should return string ID
        assert isinstance(result, str)
        assert result == "507f1f77bcf86cd799439011"

        # Verify insert was called with correct data
        mock_collection.insert_one.assert_called_once()
        call_args = mock_collection.insert_one.call_args[0][0]
        assert call_args["user_id"] == "test_user"
        assert call_args["filename"] == "test.txt"
        assert call_args["status"] == "processing"
        assert "uploaded_at" in call_args

    @pytest.mark.asyncio
    async def test_update_document_with_stats(self):
        """Test updating document with stats"""
        from services.api.database import update_document

        mock_collection = AsyncMock()

        with patch("services.api.database.get_documents", return_value=mock_collection):
            await update_document(
                "507f1f77bcf86cd799439011",
                stats_id="507f191e810c19729de860ea",
                status="completed",
            )

        # Verify update was called correctly
        mock_collection.update_one.assert_called_once()
        call_args = mock_collection.update_one.call_args[0]

        # Check filter
        assert call_args[0]["_id"] == ObjectId("507f1f77bcf86cd799439011")

        # Check update data
        update_data = call_args[1]["$set"]
        assert update_data["status"] == "completed"
        assert update_data["stats_id"] == ObjectId("507f191e810c19729de860ea")

    @pytest.mark.asyncio
    async def test_update_document_with_error(self):
        """Test updating document with error status"""
        from services.api.database import update_document

        mock_collection = AsyncMock()

        with patch("services.api.database.get_documents", return_value=mock_collection):
            await update_document(
                "507f1f77bcf86cd799439011", status="failed", error="Processing failed"
            )

        # Verify error was included
        call_args = mock_collection.update_one.call_args[0]
        update_data = call_args[1]["$set"]
        assert update_data["status"] == "failed"
        assert update_data["error"] == "Processing failed"

    @pytest.mark.asyncio
    async def test_save_document_stats(self):
        """Test saving document statistics"""
        from services.api.database import save_document_stats

        mock_result = MagicMock()
        mock_result.inserted_id = ObjectId("507f191e810c19729de860ea")

        mock_collection = AsyncMock()
        mock_collection.insert_one.return_value = mock_result

        stats_data = {
            "vocab_size": 234,
            "token_count": 1024,
            "type_token_ratio": 0.2285,
            "doc_sentiment": {"neutral": 0.7, "joy": 0.2, "sadness": 0.1},
        }

        with patch(
            "services.api.database.get_document_stats", return_value=mock_collection
        ):
            result = await save_document_stats(stats_data)

        # Should return string ID
        assert isinstance(result, str)
        assert result == "507f191e810c19729de860ea"

        # Verify stats were saved
        mock_collection.insert_one.assert_called_once_with(stats_data)

    @pytest.mark.asyncio
    async def test_get_document_with_stats(self):
        """Test retrieving a document with its stats"""
        from services.api.database import get_document

        mock_doc = {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "user_id": "test_user",
            "filename": "test.txt",
            "uploaded_at": datetime(2025, 1, 15, 10, 30, 0),
            "status": "completed",
            "stats_id": ObjectId("507f191e810c19729de860ea"),
        }

        mock_stats = {
            "_id": ObjectId("507f191e810c19729de860ea"),
            "vocab_size": 234,
            "token_count": 1024,
            "type_token_ratio": 0.2285,
        }

        mock_docs_collection = AsyncMock()
        mock_docs_collection.find_one.return_value = mock_doc

        mock_stats_collection = AsyncMock()
        mock_stats_collection.find_one.return_value = mock_stats

        with (
            patch(
                "services.api.database.get_documents", return_value=mock_docs_collection
            ),
            patch(
                "services.api.database.get_document_stats",
                return_value=mock_stats_collection,
            ),
        ):
            result = await get_document("507f1f77bcf86cd799439011", "test_user")

        # Verify document data
        assert result is not None
        assert result["_id"] == "507f1f77bcf86cd799439011"
        assert result["filename"] == "test.txt"

        # Verify stats were included and serialized
        assert "stats" in result
        assert result["stats"]["_id"] == "507f191e810c19729de860ea"
        assert result["stats"]["vocab_size"] == 234

    @pytest.mark.asyncio
    async def test_get_document_not_found(self):
        """Test retrieving non-existent document"""
        from services.api.database import get_document

        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None

        with patch("services.api.database.get_documents", return_value=mock_collection):
            result = await get_document("507f1f77bcf86cd799439011", "test_user")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_document_wrong_user(self):
        """Test retrieving document with wrong user_id"""
        from services.api.database import get_document

        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None

        with patch("services.api.database.get_documents", return_value=mock_collection):
            result = await get_document("507f1f77bcf86cd799439011", "wrong_user")

        # Verify query included user_id check
        call_args = mock_collection.find_one.call_args[0][0]
        assert call_args["user_id"] == "wrong_user"
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
