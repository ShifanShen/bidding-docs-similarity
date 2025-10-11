"""
基础类和接口定义
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExtractionMethod(Enum):
    """文本提取方法枚举"""
    PDFPLUMBER = "pdfplumber"
    OCR = "ocr"
    HYBRID = "hybrid"


@dataclass
class TaskInfo:
    """任务信息数据类"""
    task_id: str
    status: TaskStatus
    progress: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class DocumentSegment:
    """文档片段数据类"""
    page: int
    text: str
    grammar_errors: List[str]
    is_table_cell: bool = False
    row: Optional[int] = None
    col: Optional[int] = None
    table_idx: Optional[int] = None
    extraction_method: Optional[ExtractionMethod] = None
    is_page_level: bool = False
    is_merged: bool = False
    merged_pages: Optional[List[int]] = None
    segment_index: Optional[int] = None


@dataclass
class SimilarityResult:
    """相似度检测结果数据类"""
    bid_file: str
    similar_with: str
    page: int
    similar_page: int
    text: str
    similar_text: str
    similarity: float
    rank: int
    order_changed: bool = False
    stopword_evade: bool = False
    synonym_evade: bool = False
    semantic_evade: bool = False
    is_merged: bool = False
    merged_count: int = 1


class BaseService(ABC):
    """服务基类"""
    
    def __init__(self, config: Any = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化服务"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理资源"""
        pass


class BaseTextProcessor(ABC):
    """文本处理器基类"""
    
    def __init__(self, config: Any = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, text: str) -> str:
        """处理文本"""
        pass


class BaseExtractor(ABC):
    """文本提取器基类"""
    
    def __init__(self, config: Any = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def extract(self, file_path: str) -> List[DocumentSegment]:
        """提取文档片段"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查提取器是否可用"""
        pass


class BaseSimilarityCalculator(ABC):
    """相似度计算器基类"""
    
    def __init__(self, config: Any = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def calculate(self, text1: str, text2: str) -> float:
        """计算相似度"""
        pass
    
    @abstractmethod
    def calculate_batch(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """批量计算相似度"""
        pass
