from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Request
from app.api.deps import get_current_user
from app.config.settings import settings
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas, ContentSettings
import datetime
import uuid
import os
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload")
async def upload_media(
    request: Request,
    file: UploadFile = File(...),
    thread_id: str = Query(..., description="ID của cuộc hội thoại CopilotKit"),
    current_user = Depends(get_current_user)  # Yêu cầu xác thực người dùng qua JWT
):
    # Kiểm tra định dạng (ảnh hoặc video)
    if not (file.content_type.startswith("image/") or file.content_type.startswith("video/")):
        raise HTTPException(status_code=400, detail="Chỉ cho phép tải hình ảnh hoặc video!")

    # Đọc tệp và kiểm tra kích thước (tối đa 25MB)
    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Kích thước tệp vượt quá giới hạn 25MB!")
    await file.seek(0)

    try:
        # 1. Khởi tạo kết nối Azure
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")

        if not all([tenant_id, client_id, client_secret]):
            raise HTTPException(
                status_code=500, 
                detail="Lỗi cấu hình Azure: Thiếu AZURE_TENANT_ID, AZURE_CLIENT_ID, hoặc AZURE_CLIENT_SECRET trong .env"
            )

        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        blob_service_client = BlobServiceClient(settings.AZURE_STORAGE_ACCOUNT_URL, credential=credential)
        container_client = blob_service_client.get_container_client(settings.AZURE_CONTAINER_NAME)

        # Đảm bảo container tồn tại (tạo nếu chưa có)
        try:
            container_client.create_container()
        except Exception as e:
            # Bỏ qua nếu container đã tồn tại
            if "ContainerAlreadyExists" not in str(e):
                pass

        # 2. Tạo đường dẫn tệp sessions/{thread_id}/{uuid}.extension
        file_extension = os.path.splitext(file.filename)[1] or ".jpg"
        blob_filename = f"{uuid.uuid4()}{file_extension}"
        blob_name = f"sessions/{thread_id}/{blob_filename}"

        # 3. Tải tệp lên Blob Storage
        blob_client = container_client.get_blob_client(blob_name)
        content = await file.read()
        blob_client.upload_blob(content, 
                                overwrite=True,
                                content_settings=ContentSettings(content_type=file.content_type))
        logger.info(f"Uploaded file {file.filename} to Azure Blob Storage at {blob_name}")
        
        # Đăng ký hình ảnh vào MediaRegistryService
        if hasattr(request.app.state, "media_registry"):
            media_registry = request.app.state.media_registry
            media_registry.add_new_image(
                user_id=current_user.id,
                thread_id=thread_id,
                media_uuid=blob_filename.split('.')[0],
                type="remote",
                path=blob_name,
                temporary=False
            )
        logger.info(f"Registered media in MediaRegistryService: user_id={current_user.id}, thread_id={thread_id}")

        # 4. Sinh User Delegation SAS URL có thời hạn 30 phút
        delegation_start_time = datetime.datetime.now(datetime.timezone.utc)
        delegation_expiry_time = delegation_start_time + datetime.timedelta(hours=1) # Delegation Key sống 1 giờ
        
        user_delegation_key = blob_service_client.get_user_delegation_key(
            key_start_time=delegation_start_time,
            key_expiry_time=delegation_expiry_time
        )

        sas_expiry_time = delegation_start_time + datetime.timedelta(minutes=30) # SAS sống 30 phút theo yêu cầu
        
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=settings.AZURE_CONTAINER_NAME,
            blob_name=blob_name,
            user_delegation_key=user_delegation_key,
            permission=BlobSasPermissions(read=True),
            expiry=sas_expiry_time,
            start=delegation_start_time,
            protocol="https"
        )

        sas_url = f"{blob_client.url}?{sas_token}"
        logger.info(f"Generated SAS URL for {blob_name}: {sas_url}")

        return {"url": sas_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tải ảnh lên Azure: {str(e)}")
