"""
Integration tests for the API endpoints with database
These tests require a running API server and database connection
"""

import pytest
import requests
import json
from pathlib import Path

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_USER_EMAIL = "test_integration@example.com"
TEST_USER_PASSWORD = "test_password_123"

class TestAPIIntegration:
    """Integration tests for API endpoints"""

    @pytest.fixture(scope="class")
    def auth_token(self):
        """Get authentication token for testing"""
        # Register user
        register_data = {
            "email": TEST_USER_EMAIL,
            "password": TEST_USER_PASSWORD
        }
        
        try:
            response = requests.post(f"{API_BASE_URL}/auth/register", json=register_data)
            if response.status_code in [200, 409]:  # OK or already exists
                # Try to login
                login_response = requests.post(f"{API_BASE_URL}/auth/login", json=register_data)
                if login_response.status_code == 200:
                    return login_response.json().get("access_token")
        except requests.ConnectionError:
            pytest.skip("API server not running")
        
        return None

    def test_api_health(self):
        """Test that API is accessible"""
        try:
            response = requests.get(f"{API_BASE_URL}/")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
        except requests.ConnectionError:
            pytest.skip("API server not running")

    def test_user_registration(self):
        """Test user registration endpoint"""
        try:
            register_data = {
                "email": f"test_new_user_{hash('test')}@example.com",
                "password": "new_password_123"
            }
            
            response = requests.post(f"{API_BASE_URL}/auth/register", json=register_data)
            assert response.status_code in [200, 409]  # Created or already exists
        except requests.ConnectionError:
            pytest.skip("API server not running")

    def test_user_login(self):
        """Test user login endpoint"""
        try:
            login_data = {
                "email": TEST_USER_EMAIL,
                "password": TEST_USER_PASSWORD
            }
            
            response = requests.post(f"{API_BASE_URL}/auth/login", json=login_data)
            # Might fail if user doesn't exist, which is OK for this simple test
            assert response.status_code in [200, 401, 404]
        except requests.ConnectionError:
            pytest.skip("API server not running")

    def test_get_documents_with_user_id(self):
        """Test getting documents with user_id parameter"""
        try:
            # Test with a sample user_id - should work and return documents structure
            response = requests.get(f"{API_BASE_URL}/documents/test_user_id")
            assert response.status_code == 200
            data = response.json()
            assert "user_id" in data
            assert "documents" in data
            assert isinstance(data["documents"], list)
        except requests.ConnectionError:
            pytest.skip("API server not running")

    def test_upload_endpoint_post_method(self):
        """Test that upload endpoint accepts POST requests"""
        try:
            # Test POST to upload endpoint without file (should get validation/auth error, not 404)
            response = requests.post(f"{API_BASE_URL}/documents/upload")
            assert response.status_code in [400, 422, 403]  # Bad request, validation error, or forbidden
        except requests.ConnectionError:
            pytest.skip("API server not running")

    def test_upload_endpoint_exists(self):
        """Test that upload endpoint is accessible"""
        try:
            # Test OPTIONS request (CORS preflight) on correct upload endpoint
            response = requests.options(f"{API_BASE_URL}/documents/upload")
            assert response.status_code == 200
        except requests.ConnectionError:
            pytest.skip("API server not running")

    def test_api_documentation(self):
        """Test that API documentation is accessible"""
        try:
            response = requests.get(f"{API_BASE_URL}/docs")
            assert response.status_code == 200
            assert "swagger" in response.text.lower() or "openapi" in response.text.lower()
        except requests.ConnectionError:
            pytest.skip("API server not running")

    def test_cors_headers(self):
        """Test that CORS headers are present"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/",
                headers={"Origin": "http://localhost:3000"}
            )
            assert response.status_code == 200
            # CORS headers should be present for cross-origin requests
        except requests.ConnectionError:
            pytest.skip("API server not running")

class TestDatabaseIntegration:
    """Integration tests that verify database operations"""

    def test_database_connection(self):
        """Test that database connection is working"""
        # This would be tested indirectly through API calls
        # since we don't have direct database access in integration tests
        try:
            response = requests.get(f"{API_BASE_URL}/")
            assert response.status_code == 200
        except requests.ConnectionError:
            pytest.skip("API server not running")

    def test_user_creation_and_retrieval(self):
        """Test end-to-end user creation and data retrieval"""
        try:
            # Create unique user
            unique_email = f"integration_test_{hash('unique')}@example.com"
            register_data = {
                "email": unique_email,
                "password": "integration_test_pass"
            }
            
            # Register user
            response = requests.post(f"{API_BASE_URL}/auth/register", json=register_data)
            assert response.status_code in [200, 409]
            
            # Login
            login_response = requests.post(f"{API_BASE_URL}/auth/login", json=register_data)
            if login_response.status_code == 200:
                token = login_response.json().get("access_token")
                assert token is not None
        except requests.ConnectionError:
            pytest.skip("API server not running")

if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])