import os
from minio import Minio
from app.config.oss_config import minio_config

class OSSService:
    """MinIO服务类"""
    def __init__(self):
        """初始化MinIO客户端"""
        self.config = minio_config
        self.client = Minio(
            self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure
        )
        # 确保存储桶存在
        if not self.client.bucket_exists(self.config.bucket_name):
            self.client.make_bucket(self.config.bucket_name)
    
    def upload_file(self, file_path, object_name=None):
        """上传文件到MinIO"""
        if not object_name: object_name = os.path.basename(file_path)
        self.client.fput_object(
            self.config.bucket_name,
            object_name,
            file_path
        )
        return self.get_file_url(object_name)
    
    def download_file(self, object_name, file_path):
        """下载文件到本地"""
        self.client.fget_object(
            self.config.bucket_name,
            object_name,
            file_path
        )
        return True
    
    def delete_file(self, object_name):
        """删除MinIO中的文件"""
        self.client.remove_object(self.config.bucket_name, object_name)
        return True
    
    def get_file_url(self, object_name):
        """自动生成文件URL"""
        protocol = "https" if self.config.secure else "http"
        return f"{protocol}://{self.config.endpoint}/{self.config.bucket_name}/{object_name}"
    
    def list_files(self, prefix: str = ""):
        """获取MinIO的文件列表"""
        objects = self.client.list_objects(
            self.config.bucket_name,
            prefix=prefix,
            recursive=True
        )
        return [obj.object_name for obj in objects]
    
    def file_exists(self, object_name: str) -> bool:
        """检查文件是否存在"""
        try:
            self.client.stat_object(self.config.bucket_name, object_name)
            return True
        except Exception as e:
            return False

oss_service = OSSService()