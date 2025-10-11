"""
统一配置管理器
"""
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from app.config.similarity_config import SimilarityConfig
from app.config.paddle_ocr_config import PaddleOCRConfig


@dataclass
class AppConfig:
    """应用配置"""
    # 基础配置
    app_name: str = "Bidding Docs Similarity System"
    debug: bool = False
    log_level: str = "INFO"
    
    # 存储配置
    storage_dir: str = field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), '../../upload_files')))
    extracted_texts_dir: str = field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), '../../extracted_texts')))
    
    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 子配置
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    ocr: PaddleOCRConfig = field(default_factory=PaddleOCRConfig)
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保目录存在
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.extracted_texts_dir, exist_ok=True)


class ConfigManager:
    """配置管理器"""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None
    
    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = AppConfig()
    
    @property
    def config(self) -> AppConfig:
        """获取配置"""
        return self._config
    
    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """更新配置"""
        for key, value in config_dict.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
    
    def get_similarity_config(self) -> SimilarityConfig:
        """获取相似度配置"""
        return self._config.similarity
    
    def get_ocr_config(self) -> PaddleOCRConfig:
        """获取OCR配置"""
        return self._config.ocr
    
    def reload_config(self) -> None:
        """重新加载配置"""
        self._config = AppConfig()


# 全局配置管理器实例
config_manager = ConfigManager()
