import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from main import app
from app.api.deps import get_current_user

# Setup client
client = TestClient(app)

class MockUser:
    id = 123
    username = "testuser"

@pytest.fixture(autouse=True)
def setup_dependencies():
    # Override authentication dependency
    app.dependency_overrides[get_current_user] = lambda: MockUser()
    
    # Mock media registry on app.state
    mock_registry = MagicMock()
    app.state.media_registry = mock_registry
    
    yield
    
    app.dependency_overrides.clear()
    if hasattr(app.state, "media_registry"):
        delattr(app.state, "media_registry")

@pytest.mark.asyncio
@patch('app.api.upload.ClientSecretCredential')
@patch('app.api.upload.BlobServiceClient')
@patch('app.api.upload.generate_blob_sas')
async def test_upload_image_success(mock_generate_sas, mock_blob_service_client_class, mock_credential_class, monkeypatch):
    # Setup mock env vars
    monkeypatch.setenv("AZURE_TENANT_ID", "dummy-tenant")
    monkeypatch.setenv("AZURE_CLIENT_ID", "dummy-client")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "dummy-secret")
    
    # Mock Azure Blob Service Client hierarchy
    mock_blob_service_client = MagicMock()
    mock_blob_service_client_class.return_value = mock_blob_service_client
    mock_blob_service_client.account_name = "testaccount"
    
    mock_container_client = MagicMock()
    mock_blob_service_client.get_container_client.return_value = mock_container_client
    
    mock_blob_client = MagicMock()
    mock_blob_client.url = "https://testaccount.blob.core.windows.net/container/blob"
    mock_container_client.get_blob_client.return_value = mock_blob_client
    
    mock_generate_sas.return_value = "dummy-sas-token"
    
    # Make call for image
    file_content = b"fake image content"
    files = {"file": ("test.jpg", file_content, "image/jpeg")}
    response = client.post("/upload?thread_id=test-thread-uuid", files=files)
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "url" in data
    assert data["url"] == "https://testaccount.blob.core.windows.net/container/blob?dummy-sas-token"
    
    # Verify mock calls
    mock_blob_client.upload_blob.assert_called_once()
    app.state.media_registry.add_new_image.assert_called_once()


@pytest.mark.asyncio
@patch('app.api.upload.ClientSecretCredential')
@patch('app.api.upload.BlobServiceClient')
@patch('app.api.upload.generate_blob_sas')
async def test_upload_video_success(mock_generate_sas, mock_blob_service_client_class, mock_credential_class, monkeypatch):
    # Setup mock env vars
    monkeypatch.setenv("AZURE_TENANT_ID", "dummy-tenant")
    monkeypatch.setenv("AZURE_CLIENT_ID", "dummy-client")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "dummy-secret")
    
    # Mock Azure Blob Service Client hierarchy
    mock_blob_service_client = MagicMock()
    mock_blob_service_client_class.return_value = mock_blob_service_client
    mock_blob_service_client.account_name = "testaccount"
    
    mock_container_client = MagicMock()
    mock_blob_service_client.get_container_client.return_value = mock_container_client
    
    mock_blob_client = MagicMock()
    mock_blob_client.url = "https://testaccount.blob.core.windows.net/container/blob"
    mock_container_client.get_blob_client.return_value = mock_blob_client
    
    mock_generate_sas.return_value = "dummy-sas-token"
    
    # Make call for video
    file_content = b"fake video content"
    files = {"file": ("test.mp4", file_content, "video/mp4")}
    response = client.post("/upload?thread_id=test-thread-uuid", files=files)
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "url" in data
    assert data["url"] == "https://testaccount.blob.core.windows.net/container/blob?dummy-sas-token"
    
    # Verify mock calls
    mock_blob_client.upload_blob.assert_called_once()
    app.state.media_registry.add_new_image.assert_called_once()


@pytest.mark.asyncio
async def test_upload_file_too_large():
    # File content larger than 25MB (25MB + 1 byte)
    large_content = b"a" * (25 * 1024 * 1024 + 1)
    files = {"file": ("test.jpg", large_content, "image/jpeg")}
    response = client.post("/upload?thread_id=test-thread-uuid", files=files)
    
    assert response.status_code == 400
    assert "Kích thước tệp vượt quá giới hạn" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_unsupported_type():
    files = {"file": ("test.txt", b"plain text", "text/plain")}
    response = client.post("/upload?thread_id=test-thread-uuid", files=files)
    
    assert response.status_code == 400
    assert "Chỉ cho phép tải hình ảnh hoặc video" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_missing_azure_config(monkeypatch):
    # Clear Azure credentials env vars
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_ID", raising=False)
    monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)
    
    files = {"file": ("test.jpg", b"fake content", "image/jpeg")}
    response = client.post("/upload?thread_id=test-thread-uuid", files=files)
    
    assert response.status_code == 500
    assert "Lỗi cấu hình Azure" in response.json()["detail"]
