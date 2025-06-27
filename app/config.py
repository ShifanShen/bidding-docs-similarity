import os
import tempfile
from pathlib import Path

class Config:
    # 模型配置
    LOCAL_MODEL_PATH = "./local_bert_model"
    # 相似度阈值
    SIMILARITY_THRESHOLD_HIGH = 0.85
    SIMILARITY_THRESHOLD_MEDIUM = 0.65
    # 文本处理配置
    STOPWORDS_PATH = Path(__file__).parent.parent / 'stopwords.txt'
    # 加载停用词
    with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        STOPWORDS = {line.strip() for line in f if line.strip()}
    # 临时文件目录
    TEMP_DIR = Path(tempfile.gettempdir()) / 'bidding_docs'
    # 投标关键词
    BIDDING_KEYWORDS = {'投标', '招标', '采购', '合同', '项目', '技术方案', '报价', '预算', '中标', '评标'}

    @classmethod
    def init_temp_dir(cls):
        cls.TEMP_DIR.mkdir(exist_ok=True)

# 初始化临时目录
Config.init_temp_dir()