class PaddleOCRConfig:
    """PaddleOCR配置类"""
    
    # OCR 阈值配置
    OCR_THRESHOLD = 25  # 当页面文本长度低于此值时，触发OCR识别
    
    # 预处理配置
    DPI = 300  # 图像分辨率
    
    # PaddleOCR模型配置
    TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"  # 文本检测模型
    TEXT_RECOGNITION_MODEL_NAME = "PP-OCRv5_mobile_rec"  # 文本识别模型
    
    # 文档处理配置
    USE_DOC_ORIENTATION_CLASSIFY = False  # 是否使用文档方向分类
    USE_DOC_UNWARPING = False  # 是否使用文档矫正
    USE_TEXTLINE_ORIENTATION = False  # 是否使用文本行方向分类
    
    # 语言配置
    LANG = "ch"  # 语言设置，ch为中文，en为英文
    
    # 设备配置
    DEVICE = "cpu"  # 设备类型，cpu或gpu
    
    # 后处理配置
    MIN_CONFIDENCE = 0.5  # 最小置信度
    TEXT_REC_SCORE_THRESH = 0.0  # 文本识别分数阈值
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典加载配置"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

# 默认配置实例
default_paddle_ocr_config = PaddleOCRConfig()
