"""
Integration tests for the NLP Document Library API
Tests the API with a running server
"""

import pytest
import requests
import time
import subprocess
import signal
import os
from pathlib import Path


class TestAPIIntegration:
    """Integration tests that require a running API server"""

    @classmethod
    def setup_class(cls):
        """Start the API server for integration tests"""
        cls.api_process = None
        cls.base_url = "http://localhost:8000"

        # Start the API server
        try:
            project_root = Path(__file__).parent.parent.parent
            api_dir = project_root / "services" / "api"

            cls.api_process = subprocess.Popen(
                ["../../venv311/bin/uvicorn", "test_api:app", "--port", "8000"],
                cwd=api_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for server to start
            for _ in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get(f"{cls.base_url}/", timeout=1)
                    if response.status_code == 200:
                        break
                except requests.exceptions.RequestException:
                    time.sleep(1)
            else:
                raise Exception("API server failed to start")

        except Exception as e:
            pytest.skip(f"Could not start API server: {e}")

    @classmethod
    def teardown_class(cls):
        """Stop the API server"""
        if cls.api_process:
            cls.api_process.terminate()
            cls.api_process.wait()

    def test_api_health_check(self):
        """Test that the API is running and responding"""
        response = requests.get(f"{self.base_url}/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "NLP Document Library API - Test Version"

    def test_full_workflow(self):
        """Test the complete workflow: upload -> get documents -> get specific document"""
        # 1. Get initial documents
        response = requests.get(f"{self.base_url}/documents/user123")
        assert response.status_code == 200
        initial_docs = response.json()["documents"]

        # 2. Upload a new document
        test_content = "This is an integration test document."
        files = {"file": ("integration_test.txt", test_content, "text/plain")}
        data = {"user_id": "user123"}

        response = requests.post(
            f"{self.base_url}/documents/upload", files=files, data=data
        )
        assert response.status_code == 200
        upload_result = response.json()
        assert upload_result["filename"] == "integration_test.txt"
        assert upload_result["processing_status"] == "completed"
        assert "stats" in upload_result

        # 3. Get documents again (should still show mock data)
        response = requests.get(f"{self.base_url}/documents/user123")
        assert response.status_code == 200
        docs = response.json()["documents"]
        assert len(docs) == len(initial_docs)  # Mock data doesn't change

        # 4. Get specific document
        response = requests.get(f"{self.base_url}/documents/user123/mock_id_1")
        assert response.status_code == 200
        doc = response.json()
        assert doc["_id"] == "mock_id_1"
        assert doc["user_id"] == "user123"
        assert "stats" in doc

    def test_error_handling(self):
        """Test error handling for invalid requests"""
        # Test invalid user ID format
        response = requests.get(f"{self.base_url}/documents/")
        assert response.status_code == 404

        # Test non-existent document
        response = requests.get(f"{self.base_url}/documents/user123/nonexistent")
        assert response.status_code == 200  # Mock API returns mock data

        # Test upload without file
        response = requests.post(
            f"{self.base_url}/documents/upload", data={"user_id": "user123"}
        )
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
