import os

class MinioConfig:
    """MinIO配置类"""
    def __init__(self):
        #端口号9000，控制台9001
        self.endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        #访问密钥和密码密钥
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        #是否使用安全连接
        self.secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        #存储桶名称
        self.bucket_name = os.getenv("MINIO_BUCKET", "bidding-docs")

minio_config = MinioConfig()