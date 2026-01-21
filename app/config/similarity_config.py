"""
相似度分析服务配置（SimilarityConfig）

- **USER_TUNABLE_FIELDS**：建议提供给用户（/api/config/get|update）进行调整的字段
- **ADVANCED_FIELDS**：高级参数，默认值通常足够；只有明确知道影响时才建议调整
"""
import os

class SimilarityConfig:
    """相似度分析服务配置类"""
    
    # 存储配置
    STORAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../upload_files'))
    
    # ==================== 并发 / 稳定性（高级） ====================
    # MAX_CONCURRENT_TASKS：
    # - 含义：同一时间允许并行跑几个分析任务
    # - 调大：吞吐更高，但更吃 CPU/RAM/GPU，可能更容易 OOM 或触发系统换页导致变慢
    # - 调小：更稳，但排队更久
    MAX_CONCURRENT_TASKS = 3
    # MAX_TASK_TIMEOUT（秒）：
    # - 含义：单个任务的超时保护；0=不限制（适合千页级文档，但要自己确保机器资源足够）
    # - 建议：生产环境若要防止“极端大文件卡死”，可设置 1800/3600
    MAX_TASK_TIMEOUT = 0
    
    # ==================== 切分 / 预处理（用户常调） ====================
    # MIN_TEXT_LENGTH：
    # - 含义：参与相似度计算的最小片段长度（太短的句子/碎片会被过滤）
    # - 调大：减少噪声/误报，但可能漏掉短条款（例如“交货期：30天”）
    # - 调小：更敏感，但更容易把模板短语也当成相似
    MIN_TEXT_LENGTH = 50
    # MIN_SEGMENT_LENGTH / MAX_SEGMENT_LENGTH：
    # - 含义：切分后的目标段落长度范围（不是硬限制，但会影响合并/拆分策略）
    # - 调大：段落更长，语义更完整，但可能把多个条款揉在一起；向量化更慢
    # - 调小：段落更碎，召回更高但误报更多，且更受 OCR/换行噪声影响
    MIN_SEGMENT_LENGTH = 50
    MAX_SEGMENT_LENGTH = 1200
    # DETECTION_MODE：
    # - paragraph：按自然段（空行）切分（适合有明显空行的 Word/纯文本）
    # - sentence：按句子切分（更碎，误报更高，一般不推荐用于招标文件）
    # - page：整页作为一个段（适合“整页抄袭”的粗粒度检测，但定位不够精细）
    # - chapter_paragraph：按“章/节/条款编号”切分（推荐用于招标/投标 PDF）
    # 什么时候调：
    # - PDF 换行多、几乎没空行：优先用 chapter_paragraph
    # - 文本本身有清晰空行：可用 paragraph
    DETECTION_MODE = "chapter_paragraph"
    
    # OCR配置
    # ENABLE_OCR：
    # - True：强依赖 OCR（适合扫描版 PDF）；速度慢、资源占用高
    # - False：优先 pdfplumber（适合可复制文本 PDF）；速度快更稳定
    ENABLE_OCR = False

    # 表格检测配置（MVP）
    # （高级）默认不建议用户调整，除非你明确要做“表格/参数抄袭”检测
    ENABLE_TABLE_DETECTION = False
    TABLE_MIN_ROWS = 3
    TABLE_VALUE_TOLERANCE = 0.02
    TABLE_TEXT_WEIGHT = 0.4
    TABLE_VALUE_WEIGHT = 0.6
    
    # ==================== 相似度阈值（用户常调） ====================
    # 重要：这些阈值的含义是“向量相似度/余弦相似度”，范围 [0,1]，越大越相似。
    #
    # TENDER_SIMILARITY_THRESHOLD（招标剔除阈值）：
    # - 含义：投标片段若与“招标文件”最大相似度 >= 此值，则认为是“招标复述/引用”，从投标互比中剔除
    # - 调大（例如 0.75）：剔除更少，更容易把“引用招标条款”的内容也算成投标互抄（误报↑）
    # - 调小（例如 0.45）：剔除更多，更容易把真实互抄也误剔除（漏报↑）
    # - 建议：一般 0.55~0.70；招标文件非常模板化时可适当调低
    TENDER_SIMILARITY_THRESHOLD = 0.6
    # BID_SIMILARITY_THRESHOLD（投标互相比对阈值）：
    # - 含义：投标 A 片段与投标 B 片段相似度 >= 此值，则认为“命中相似片段”
    # - 调大：更严格，命中更少（误报↓，漏报↑）
    # - 调小：更宽松，命中更多（误报↑，漏报↓）
    # - 建议：
    #   - 想只抓几乎一致：0.90~0.95
    #   - 想抓“改写后仍相似”：0.80~0.90（会带来更多模板误报，需要配合切分更好）
    BID_SIMILARITY_THRESHOLD = 0.9
    # NEAR_IDENTICAL_THRESHOLD（几乎相同页面阈值）：
    # - 含义：用于标记“几乎相同页面”的辅助判断（影响 is_near_identical）
    # - 建议：通常略高于 BID_SIMILARITY_THRESHOLD，例如 0.95
    NEAR_IDENTICAL_THRESHOLD = 0.95
    # HIGH_SIMILARITY_THRESHOLD / VERY_HIGH_SIMILARITY_THRESHOLD：
    # - 含义：用于“高相似模板文本”的二次过滤策略（非常高相似但可能是通用条款/术语堆砌）
    # - 一般不用改
    HIGH_SIMILARITY_THRESHOLD = 0.99
    VERY_HIGH_SIMILARITY_THRESHOLD = 0.995
    
    # ==================== 规避行为检测（高级） ====================
    # SEMANTIC_EVADE_*：
    # - 含义：在某个相似度区间内判定“语义规避”（既不低也不高）
    # - 一般不建议改；除非你要调整“语义规避”判定敏感度
    SEMANTIC_EVADE_LOWER_THRESHOLD = 0.85
    SEMANTIC_EVADE_UPPER_THRESHOLD = 0.92
    SYNONYM_TOKEN_COUNT_DIFF_THRESHOLD = 0.2
    COMMON_TERM_COUNT_THRESHOLD = 2
    
    # ==================== 性能（高级） ====================
    # ENABLE_GPU：
    # - True：若机器有可用 CUDA，会使用 GPU 加速 embedding 与 FAISS（更快）
    # - False：强制 CPU（更稳更省显存，但更慢）
    ENABLE_GPU = True
    # BATCH_SIZE：
    # - 含义：向量化/检索时的批处理大小（影响速度与内存峰值）
    # - 调大：更快但更吃内存/显存，可能 OOM
    # - 调小：更稳但更慢
    # - 建议：通常不需要改；CPU/小内存机器可用 128~512
    BATCH_SIZE = 1000
    # MEMORY_CLEANUP_INTERVAL：
    # - 含义：清理内存/显存的节奏（越小越频繁，越稳但可能变慢）
    MEMORY_CLEANUP_INTERVAL = 500
    # SIMILARITY_TOP_K：
    # - 含义：每个片段检索前 K 个相似候选（K 越大越容易发现更多候选，但耗时更高）
    SIMILARITY_TOP_K = 3
    
    # 跨页段落合并配置
    # AUTO_MERGE_CROSS_PAGE_PARAGRAPH：
    # - True：会尝试把“跨页被截断的段落”拼起来（对 PDF 有时有帮助，但也可能误拼）
    # - 建议：默认 False；当你发现大量“上一页句子没结束，下一页接着来”的情况再打开
    AUTO_MERGE_CROSS_PAGE_PARAGRAPH = False

    # ==================== 字段分层：用于接口暴露/校验 ====================
    # 用户常用、建议暴露给 /api/config/get|update 的字段
    USER_TUNABLE_FIELDS = [
        "ENABLE_OCR",
        "DETECTION_MODE",
        "MIN_TEXT_LENGTH",
        "MIN_SEGMENT_LENGTH",
        "MAX_SEGMENT_LENGTH",
        "TENDER_SIMILARITY_THRESHOLD",
        "BID_SIMILARITY_THRESHOLD",
        "NEAR_IDENTICAL_THRESHOLD",
    ]

    # 高级字段：默认不建议用户调整（可在代码/运维层面改）
    ADVANCED_FIELDS = [
        "MAX_CONCURRENT_TASKS",
        "MAX_TASK_TIMEOUT",
        "BATCH_SIZE",
        "ENABLE_TABLE_DETECTION",
        "TABLE_MIN_ROWS",
        "TABLE_VALUE_TOLERANCE",
        "TABLE_TEXT_WEIGHT",
        "TABLE_VALUE_WEIGHT",
        "SEMANTIC_EVADE_LOWER_THRESHOLD",
        "SEMANTIC_EVADE_UPPER_THRESHOLD",
        "SYNONYM_TOKEN_COUNT_DIFF_THRESHOLD",
        "COMMON_TERM_COUNT_THRESHOLD",
        "ENABLE_GPU",
        "MEMORY_CLEANUP_INTERVAL",
        "SIMILARITY_TOP_K",
        "AUTO_MERGE_CROSS_PAGE_PARAGRAPH",
        "STORAGE_DIR",
    ]
    
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