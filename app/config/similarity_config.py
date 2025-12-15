"""
相似度分析服务配置
"""
import os

class SimilarityConfig:
    """相似度分析服务配置类"""
    
    # 存储配置
    STORAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../upload_files'))
    
    # 并发控制
    MAX_CONCURRENT_TASKS = 3
    MAX_TASK_TIMEOUT = 0  # 秒，0表示无超时限制，支持千页级文档分析
    
    # 文本处理配置
    MIN_TEXT_LENGTH = 100  # 最小文本长度阈值
    MIN_SEGMENT_LENGTH = 100  # 最小段落长度阈值（同一页内上下文感知）- 进一步增加以保持更大段落
    MAX_SEGMENT_LENGTH = 1200  # 最大段落长度阈值（同一页内上下文感知）- 支持大段落检测
    BATCH_SIZE = 1000  # 批量处理大小
    
    # OCR配置
    ENABLE_OCR = False  # 是否启用OCR提取（True使用PaddleOCR，False使用pdfplumber）
    
    # 检测模式配置
    DETECTION_MODE = "paragraph"  # 检测模式: "paragraph"(段落), "page"(整页), "sentence"(句子)
    PAGE_LEVEL_DETECTION = False  # 是否启用整页级别检测

    # 表格检测配置（MVP）
    ENABLE_TABLE_DETECTION = False  # 是否启用表格抄袭检测（基于文本的KV抽取）
    TABLE_MIN_ROWS = 3  # 判定为表格块的最少连续KV行数
    TABLE_VALUE_TOLERANCE = 0.02  # 数值相对误差容忍（2%）
    TABLE_TEXT_WEIGHT = 0.4  # 表格文本embedding相似度权重
    TABLE_VALUE_WEIGHT = 0.6  # 表格数值/参数匹配权重
    
    # 相似度阈值配置
    TENDER_SIMILARITY_THRESHOLD = 0.6  # 与招标文件相似度的剔除阈值
    BID_SIMILARITY_THRESHOLD = 0.9  # 投标文件间相似度的检测阈值 - 提高以检测几乎相同的页面
    NEAR_IDENTICAL_THRESHOLD = 0.95  # 几乎相同页面的检测阈值
    HIGH_SIMILARITY_THRESHOLD = 0.99  # 高相似度阈值
    VERY_HIGH_SIMILARITY_THRESHOLD = 0.995  # 极高相似度阈值
    
    # 规避行为检测配置
    SEMANTIC_EVADE_LOWER_THRESHOLD = 0.85  # 语义规避检测的下限阈值
    SEMANTIC_EVADE_UPPER_THRESHOLD = 0.92  # 语义规避检测的上限阈值
    SYNONYM_TOKEN_COUNT_DIFF_THRESHOLD = 0.2  # 同义词检测的词数差异阈值
    COMMON_TERM_COUNT_THRESHOLD = 2  # 通用模板文本检测的术语数量阈值
    
    # 性能配置
    ENABLE_GPU = True  # 是否启用GPU加速
    MEMORY_CLEANUP_INTERVAL = 500  # 内存清理间隔
    SIMILARITY_TOP_K = 3  # 返回每个查询的前K个相似结果
    
    # 日志配置
    LOG_LEVEL = "INFO"
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典加载配置"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

# 默认配置实例
default_config = SimilarityConfig()