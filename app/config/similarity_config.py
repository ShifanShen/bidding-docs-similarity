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
    MIN_SEGMENT_LENGTH = 200  # 最小段落长度阈值（提高以保持语义完整性）
    MAX_SEGMENT_LENGTH = 1000  # 最大段落长度阈值（提高以容纳完整条款）
    TABLE_PROCESSING_MODE = "row"  # 表格处理方式: "cell"(按单元格), "row"(按行)
    BATCH_SIZE = 1000  # 批量处理大小
    
    # 智能分段配置
    ENABLE_SMART_SEGMENTATION = True  # 启用智能分段
    SENTENCE_BOUNDARY_MARKERS = ['。', '！', '？', '；', '：']  # 句子边界标记
    PARAGRAPH_BOUNDARY_MARKERS = ['\n\n', '\n\r\n', '\r\n\r\n']  # 段落边界标记
    MIN_SENTENCE_LENGTH = 20  # 最小句子长度
    MAX_SENTENCE_LENGTH = 200  # 最大句子长度
    
    # 上下文感知检测配置
    ENABLE_PAGE_LEVEL_DETECTION = True  # 启用页面级别检测
    PAGE_SIMILARITY_THRESHOLD = 0.7  # 页面相似度阈值
    MIN_PAGE_TEXT_LENGTH = 500  # 页面最小文本长度
    CONTEXT_WINDOW_SIZE = 3  # 上下文窗口大小（前后页数）
    ENABLE_ADJACENT_MERGE = True  # 启用相邻相似片段合并
    MERGE_THRESHOLD = 0.8  # 合并阈值
    
    # 相似度阈值
    TENDER_SIMILARITY_THRESHOLD = 0.6  # 与招标文件相似度的剔除阈值（降低以更严格剔除）
    BID_SIMILARITY_THRESHOLD = 0.75  # 投标文件间相似度的检测阈值
    
    # 招标文件过滤增强配置
    ENABLE_ENHANCED_TENDER_FILTERING = True  # 启用增强的招标文件过滤
    TENDER_FILTER_MULTI_THRESHOLD = True  # 使用多阈值过滤
    TENDER_HIGH_SIMILARITY_THRESHOLD = 0.8  # 高相似度阈值
    TENDER_MEDIUM_SIMILARITY_THRESHOLD = 0.6  # 中等相似度阈值
    TENDER_LOW_SIMILARITY_THRESHOLD = 0.4  # 低相似度阈值
    TABLE_ROW_THRESHOLD_OFFSET = 0.025  # 表格行相似度阈值偏移量
    TABLE_CELL_THRESHOLD_OFFSET = 0.025  # 表格单元格相似度阈值偏移量
    HIGH_SIMILARITY_THRESHOLD = 0.99  # 高相似度阈值
    VERY_HIGH_SIMILARITY_THRESHOLD = 0.995  # 极高相似度阈值
    
    # 规避行为检测配置
    SEMANTIC_EVADE_LOWER_THRESHOLD = 0.85  # 语义规避检测的下限阈值
    SEMANTIC_EVADE_UPPER_THRESHOLD = 0.92  # 语义规避检测的上限阈值
    SYNONYM_TOKEN_COUNT_DIFF_THRESHOLD = 0.2  # 同义词检测的词数差异阈值
    COMMON_TERM_COUNT_THRESHOLD = 2  # 通用模板文本检测的术语数量阈值
    
    # 公章和印章检测配置
    ENABLE_SEAL_DETECTION = True  # 启用公章检测
    SEAL_PATTERNS = ['公章', '印章', '印鉴', '公司章', '单位章', '合同专用章', '财务专用章']  # 公章关键词
    SEAL_SIMILARITY_THRESHOLD = 0.95  # 公章相似度阈值（忽略公章差异）
    
    # 模板文本增强过滤
    ENABLE_ENHANCED_TEMPLATE_FILTERING = True  # 启用增强模板过滤
    TEMPLATE_PATTERNS = [
        '承诺函', '声明函', '保证函', '确认函',
        '投标函', '投标书', '投标文件',
        '法定代表人', '授权委托书', '营业执照',
        '资质证书', '安全生产许可证',
        '项目经理', '技术负责人', '质量保证期',
        '履约保证金', '投标保证金'
    ]
    TEMPLATE_SIMILARITY_THRESHOLD = 0.9  # 模板文本相似度阈值
    
    # 内容类型过滤配置
    ENABLE_CONTENT_TYPE_FILTERING = True  # 启用内容类型过滤
    IGNORE_CONTENT_TYPES = ['header', 'footer', 'page_number', 'watermark']  # 忽略的内容类型
    CONTENT_TYPE_SIMILARITY_THRESHOLD = 0.95  # 内容类型相似度阈值
    
    # 相似度计算优化配置
    ENABLE_MULTI_DIMENSIONAL_SIMILARITY = True  # 启用多维度相似度计算
    SIMILARITY_WEIGHTS = {
        'semantic': 0.6,      # 语义相似度权重
        'structural': 0.2,    # 结构相似度权重
        'lexical': 0.2        # 词汇相似度权重
    }
    
    # 文本提取策略配置
    ENABLE_SMART_EXTRACTION_STRATEGY = True  # 启用智能提取策略
    EXTRACTION_MODE = "hybrid"  # 提取模式: "pdfplumber", "ocr", "hybrid"
    OCR_FALLBACK_THRESHOLD = 50  # OCR后备阈值（降低以更频繁使用OCR）
    ENABLE_OCR_FOR_SCANNED_PDF = True  # 对扫描PDF启用OCR
    ENABLE_OCR_FOR_COMPLEX_LAYOUT = True  # 对复杂布局启用OCR
    
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