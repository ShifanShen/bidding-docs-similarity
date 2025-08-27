# 相似度分析服务配置
import os

class SimilarityConfig:
    # 存储配置
    STORAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../upload_files'))
    
    # 并发控制
    MAX_CONCURRENT_TASKS = 3
    MAX_TASK_TIMEOUT = 1200  # 秒
    
    # 文本处理
    MIN_TEXT_LENGTH = 100  # 最小文本长度阈值
    MIN_SEGMENT_LENGTH = 100  # 最小段落长度阈值
    MAX_SEGMENT_LENGTH = 500  # 最大段落长度阈值
    TABLE_PROCESSING_MODE = "row"  # 表格处理方式: "cell"(按单元格), "row"(按行)
    BATCH_SIZE = 1000  # 批量处理大小
    
    # 相似度阈值
    TENDER_SIMILARITY_THRESHOLD = 0.6  # 与招标文件相似度的剔除阈值
    BID_SIMILARITY_THRESHOLD = 0.9  # 投标文件间相似度的检测阈值
    TABLE_ROW_THRESHOLD_OFFSET = 0.05  # 表格行相似度阈值偏移量
    TABLE_CELL_THRESHOLD_OFFSET = 0.05  # 表格单元格相似度阈值偏移量
    HIGH_SIMILARITY_THRESHOLD = 0.99  # 高相似度阈值
    VERY_HIGH_SIMILARITY_THRESHOLD = 0.995  # 极高相似度阈值
    
    # 规避行为检测配置
    SEMANTIC_EVADE_LOWER_THRESHOLD = 0.85  # 语义规避检测的下限阈值
    SEMANTIC_EVADE_UPPER_THRESHOLD = 0.92  # 语义规避检测的上限阈值
    SYNONYM_TOKEN_COUNT_DIFF_THRESHOLD = 0.2  # 同义词检测的词数差异阈值
    COMMON_TERM_COUNT_THRESHOLD = 2  # 通用模板文本检测的术语数量阈值
    
    # 高级配置
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