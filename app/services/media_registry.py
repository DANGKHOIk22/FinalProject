import redis
import json
import base64
import os
import datetime
import io
import logging
from PIL import Image
from typing import Optional
from functools import lru_cache
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

logger = logging.getLogger(__name__)

class MediaRegistryService:
    def __init__(self, redis_url: str, azure_storage_url: str, azure_container_name: str):
        # Redis Connection
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.azure_container_name = azure_container_name
        self.azure_storage_url = azure_storage_url
        
        # Azure BlobServiceClient Initialization
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        
        if all([tenant_id, client_id, client_secret]):
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )
            self.blob_service_client = BlobServiceClient(azure_storage_url, credential=credential)
        else:
            self.blob_service_client = None

        logger.info("MediaRegistryService initialized with Redis and Azure Blob Storage configurations.")
        
        # Thread-local RAM cache via dict or lru_cache for PIL objects
        # We use a simple dict as LRU cache for current process
        self.ram_cache = {}
        self.cache_capacity = 100

    def _get_redis_key(self, user_id: str, thread_id: str, media_uuid: str) -> str:
        return f"media:{user_id}:{thread_id}:{media_uuid}"

    def add_new_image(self, user_id: str, thread_id: str, media_uuid: str, type: str, path: str, temporary: bool = True, ttl_seconds: int = 3600):
        key = self._get_redis_key(user_id, thread_id, media_uuid)
        data = {
            "type": type,
            "temporary": "true" if temporary else "false",
            "path": path,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        self.redis_client.hset(key, mapping=data)
        if temporary:
            self.redis_client.expire(key, ttl_seconds)
            # TODO: ZADD media:cleanup_queue ...

    def get_sas_url(self, user_id: str, thread_id: str, media_uuid: str) -> Optional[str]:
        key = self._get_redis_key(user_id, thread_id, media_uuid)
        media_data = self.redis_client.hgetall(key)
        
        if not media_data:
            logger.warning(f"Media not found for user_id: {user_id}, thread_id: {thread_id}, media_uuid: {media_uuid}")
            return None
            
        if media_data.get("type") == "local":
            return None
            
        if media_data.get("type") == "remote":
            sas_url = media_data.get("sas_url")
            sas_expires_at = media_data.get("sas_expires_at")
            
            current_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
            if sas_url and sas_expires_at and (current_time + 300 < float(sas_expires_at)):
                return sas_url
                
            # Need to generate new SAS URL
            if not self.blob_service_client:
                return None
                
            blob_name = media_data.get("path")
            
            delegation_start_time = datetime.datetime.now(datetime.timezone.utc)
            delegation_expiry_time = delegation_start_time + datetime.timedelta(hours=1)
            
            user_delegation_key = self.blob_service_client.get_user_delegation_key(
                key_start_time=delegation_start_time,
                key_expiry_time=delegation_expiry_time
            )
            
            sas_expiry_time = delegation_start_time + datetime.timedelta(minutes=30)
            
            sas_token = generate_blob_sas(
                account_name=self.blob_service_client.account_name,
                container_name=self.azure_container_name,
                blob_name=blob_name,
                user_delegation_key=user_delegation_key,
                permission=BlobSasPermissions(read=True),
                expiry=sas_expiry_time,
                start=delegation_start_time,
                protocol="https"
            )
            
            blob_client = self.blob_service_client.get_container_client(self.azure_container_name).get_blob_client(blob_name)
            new_sas_url = f"{blob_client.url}?{sas_token}"
            
            # Update Redis
            self.redis_client.hset(key, mapping={
                "sas_url": new_sas_url,
                "sas_expires_at": str(sas_expiry_time.timestamp())
            })
            
            return new_sas_url
            
        return None

    def get_pil_image(self, user_id: str, thread_id: str, media_uuid: str) -> Optional[Image.Image]:
        cache_key = f"{user_id}:{thread_id}:{media_uuid}"
        if cache_key in self.ram_cache:
            return self.ram_cache[cache_key]

        key = self._get_redis_key(user_id, thread_id, media_uuid)
        media_data = self.redis_client.hgetall(key)
        
        if not media_data:
            return None
            
        img = None
        if media_data.get("type") == "local":
            path = media_data.get("path")
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
        elif media_data.get("type") == "remote":
            sas_url = self.get_sas_url(user_id, thread_id, media_uuid)
            if sas_url:
                import requests
                response = requests.get(sas_url)
                if response.status_code == 200:
                    img = Image.open(io.BytesIO(response.content)).convert("RGB")
                    
        if img:
            # Manage LRU manually or just simplistic cache limit
            if len(self.ram_cache) >= self.cache_capacity:
                self.ram_cache.pop(next(iter(self.ram_cache)))
            self.ram_cache[cache_key] = img
            return img
            
        return None

    def get_base_64(self, user_id: str, thread_id: str, media_uuid: str) -> Optional[str]:
        img = self.get_pil_image(user_id, thread_id, media_uuid)
        if not img:
            return None
            
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
