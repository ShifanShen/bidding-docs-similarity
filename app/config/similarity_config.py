# 相似度分析服务配置

class SimilarityConfig:
    # 存储配置
    STORAGE_DIR = "tmp_files"
    
    # 并发控制
    MAX_CONCURRENT_TASKS = 3
    MAX_TASK_TIMEOUT = 1200  # 秒
    
    # 文本处理
    MIN_TEXT_LENGTH = 100  # 最小文本长度阈值
    TABLE_PROCESSING_MODE = "row"  # 表格处理方式: "cell"(按单元格), "row"(按行)
    BATCH_SIZE = 1000  # 批量处理大小
    
    # 相似度阈值
    TENDER_SIMILARITY_THRESHOLD = 0.7  # 与招标文件相似度的剔除阈值
    BID_SIMILARITY_THRESHOLD = 0.9  # 投标文件间相似度的检测阈值
    
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