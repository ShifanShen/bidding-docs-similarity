class TesseractOCRConfig:
    # OCR 阈值配置
    OCR_THRESHOLD = 25  # 当页面文本长度低于此值时，触发OCR识别
    
    # 预处理配置
    DPI = 300  # 图像分辨率
    
    # Tesseract特定配置
    TESSERACT_CONFIG = r'--oem 3 --psm 6 -l chi_sim'  # 识别引擎模式和语言配置
    
    # 后处理配置
    MIN_CONFIDENCE = 0.5  # 最小置信度
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典加载配置"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

# 默认配置实例
default_tesseract_config = TesseractOCRConfig()