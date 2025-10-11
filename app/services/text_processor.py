"""
文本处理服务
"""
import re
import logging
from typing import List, Dict, Any, Set
from app.core.base import BaseTextProcessor, DocumentSegment
from app.core.config_manager import config_manager
from app.service.text_utils import (
    remove_stopwords,
    detect_grammar_errors,
    split_text_to_segments,
    is_order_changed,
    is_stopword_evade,
    is_synonym_evade
)
from app.config.synonyms_config import SYNONYMS


class TextProcessorService(BaseTextProcessor):
    """文本处理服务"""
    
    def __init__(self):
        super().__init__(config_manager.get_similarity_config())
        self.logger = logging.getLogger(__name__)
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> Set[str]:
        """加载停用词"""
        try:
            stopwords_file = "stopwords.txt"
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                stopwords = {line.strip() for line in f if line.strip()}
            self.logger.info(f"加载了 {len(stopwords)} 个停用词")
            return stopwords
        except Exception as e:
            self.logger.warning(f"停用词加载失败: {str(e)}")
            return set()
    
    def process(self, text: str) -> str:
        """处理文本"""
        # 基础文本清理
        cleaned_text = self._clean_text(text)
        
        # 去除停用词
        if self.stopwords:
            cleaned_text = remove_stopwords(cleaned_text, list(self.stopwords))
        
        return cleaned_text
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留中文、英文、数字、基本标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,;:!?()（）【】《》""''、。，；：！？]', '', text)
        
        return text.strip()
    
    def segment_text(self, text: str) -> List[str]:
        """分段文本"""
        return split_text_to_segments(text)
    
    def detect_grammar_errors(self, text: str) -> List[str]:
        """检测语法错误"""
        return detect_grammar_errors(text)
    
    def process_segments(self, segments: List[DocumentSegment]) -> List[DocumentSegment]:
        """处理文档片段"""
        processed_segments = []
        
        for segment in segments:
            # 处理文本
            processed_text = self.process(segment.text)
            
            if len(processed_text) >= self.config.MIN_TEXT_LENGTH:
                # 检测语法错误
                grammar_errors = self.detect_grammar_errors(processed_text)
                
                # 创建处理后的片段
                processed_segment = DocumentSegment(
                    page=segment.page,
                    text=processed_text,
                    grammar_errors=grammar_errors,
                    is_table_cell=segment.is_table_cell,
                    row=segment.row,
                    col=segment.col,
                    table_idx=segment.table_idx,
                    extraction_method=segment.extraction_method,
                    is_page_level=segment.is_page_level,
                    is_merged=segment.is_merged,
                    merged_pages=segment.merged_pages,
                    segment_index=segment.segment_index
                )
                processed_segments.append(processed_segment)
        
        return processed_segments
    
    def detect_evasion_behavior(self, text1: str, text2: str, similarity: float) -> Dict[str, bool]:
        """检测规避行为"""
        return {
            'order_changed': is_order_changed(text1, text2),
            'stopword_evade': is_stopword_evade(text1, text2, list(self.stopwords)),
            'synonym_evade': is_synonym_evade(text1, text2),
            'semantic_evade': self._detect_semantic_evasion(text1, text2, similarity)
        }
    
    def _detect_semantic_evasion(self, text1: str, text2: str, similarity: float) -> bool:
        """检测语义规避"""
        if (similarity > self.config.SEMANTIC_EVADE_LOWER_THRESHOLD and
            similarity < self.config.SEMANTIC_EVADE_UPPER_THRESHOLD):
            # 检查是否存在更高级的规避行为
            return True
        return False
    
    def is_template_text(self, text: str) -> bool:
        """判断是否为模板文本"""
        if not self.config.ENABLE_ENHANCED_TEMPLATE_FILTERING:
            return False
        
        # 检查是否包含模板关键词
        template_count = sum(1 for pattern in self.config.TEMPLATE_PATTERNS if pattern in text)
        
        # 如果包含多个模板关键词，认为是模板文本
        if template_count >= 2:
            return True
        
        # 检查文本长度和结构特征
        if len(text) < 200:  # 短文本
            # 检查是否以模板开头
            for pattern in self.config.TEMPLATE_PATTERNS:
                if text.startswith(pattern):
                    return True
        
        return False
    
    def contains_seal_info(self, text: str) -> bool:
        """检查文本是否包含公章信息"""
        return any(pattern in text for pattern in self.config.SEAL_PATTERNS)
    
    def remove_seal_text(self, text: str) -> str:
        """移除文本中的公章相关信息"""
        # 移除公章关键词
        for pattern in self.config.SEAL_PATTERNS:
            text = text.replace(pattern, '')
        
        # 移除可能的公章位置描述
        seal_positions = ['左下角', '右下角', '左上角', '右上角', '中间', '中央']
        for pos in seal_positions:
            text = text.replace(pos, '')
        
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
