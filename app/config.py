import os
from pathlib import Path

class Config:
    # 项目根目录
    ROOT_DIR = Path(__file__).parent.parent.resolve()
    
    # 文档存储配置
    UPLOAD_DIR = str(ROOT_DIR / 'uploads')
    TENDER_DIR = str(ROOT_DIR / 'tender_docs')
    BIDDING_DIR = str(ROOT_DIR / 'bidding_docs')
    
    # 文本处理配置
    MIN_CHUNK_LENGTH = 500  # 最小文本块长度(字符)
    SECTION_MERGE_THRESHOLD = 0.3  # 章节合并相似度阈值
    
    # 上下文检索配置
    CONTEXT_WINDOW_SIZE = 2  # 前后各检索2个文本块
    CONTINUOUS_MATCH_THRESHOLD = 3  # 连续匹配阈值
    CONTEXT_SIMILARITY_THRESHOLD = 0.6  # 上下文相似度阈值
    
    # 模型配置
    TEXT_VECTOR_MODEL_PATH = str(ROOT_DIR / 'local_text2vec_model')
    
    # 创建必要的目录
    @staticmethod
    def init_directories():
        for dir_path in [Config.UPLOAD_DIR, Config.TENDER_DIR, Config.BIDDING_DIR]:
            os.makedirs(dir_path, exist_ok=True)

# 初始化配置
Config.init_directories()