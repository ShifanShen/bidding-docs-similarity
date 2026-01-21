import logging
import os
from typing import Optional

import urllib3
from minio import Minio
from minio.error import S3Error

from app.config.oss_config import minio_config

logger = logging.getLogger(__name__)

class OSSService:
    """MinIO服务类"""
    def __init__(self):
        """初始化MinIO客户端（不在此处做网络探测，避免服务启动时阻塞/报错）"""
        self.config = minio_config
        self._client: Optional[Minio] = None
        self._available: Optional[bool] = None  # None=未探测

    def _get_client(self) -> Minio:
        """延迟创建 MinIO client，并设置较短超时/较低重试，避免刷屏与卡顿"""
        if self._client is None:
            # 短超时 + 少重试（否则 WinError 10061 会刷大量 Retrying 日志）
            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=2.0, read=10.0),
                retries=urllib3.Retry(total=1, connect=1, read=0, redirect=0, status=0),
            )
            self._client = Minio(
                self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure,
                http_client=http_client,
            )
        return self._client

    def is_available(self) -> bool:
        """探测 MinIO 是否可用（失败则返回 False，不抛异常）"""
        if self._available is not None:
            return self._available
        try:
            client = self._get_client()
            # 轻量探测：bucket_exists 会触发一次请求
            _ = client.bucket_exists(self.config.bucket_name)
            self._available = True
        except Exception as e:
            self._available = False
            logger.warning(f"MinIO不可用: endpoint={self.config.endpoint} bucket={self.config.bucket_name} err={e}")
        return self._available

    def ensure_bucket(self) -> None:
        """确保存储桶存在（仅在 MinIO 可用时执行）"""
        client = self._get_client()
        if not client.bucket_exists(self.config.bucket_name):
            client.make_bucket(self.config.bucket_name)
    
    def upload_file(self, file_path, object_name=None):
        """上传文件到MinIO"""
        if not object_name: object_name = os.path.basename(file_path)
        if not self.is_available():
            raise ConnectionError(f"MinIO不可用: {self.config.endpoint}")
        self.ensure_bucket()
        self._get_client().fput_object(
            self.config.bucket_name,
            object_name,
            file_path
        )
        return self.get_file_url(object_name)
    
    def download_file(self, object_name, file_path):
        """下载文件到本地"""
        if not self.is_available():
            raise ConnectionError(f"MinIO不可用: {self.config.endpoint}")
        self._get_client().fget_object(
            self.config.bucket_name,
            object_name,
            file_path
        )
        return True
    
    def delete_file(self, object_name):
        """删除MinIO中的文件"""
        if not self.is_available():
            raise ConnectionError(f"MinIO不可用: {self.config.endpoint}")
        self._get_client().remove_object(self.config.bucket_name, object_name)
        return True
    
    def get_file_url(self, object_name):
        """自动生成文件URL"""
        protocol = "https" if self.config.secure else "http"
        return f"{protocol}://{self.config.endpoint}/{self.config.bucket_name}/{object_name}"
    
    def list_files(self, prefix: str = ""):
        """获取MinIO的文件列表"""
        if not self.is_available():
            raise ConnectionError(f"MinIO不可用: {self.config.endpoint}")
        objects = self._get_client().list_objects(
            self.config.bucket_name,
            prefix=prefix,
            recursive=True
        )
        return [obj.object_name for obj in objects]
    
    def file_exists(self, object_name: str) -> bool:
        """检查文件是否存在"""
        try:
            if not self.is_available():
                return False
            self._get_client().stat_object(self.config.bucket_name, object_name)
            return True
        except Exception as e:
            return False