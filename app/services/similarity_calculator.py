"""
相似度计算服务
"""
import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from app.core.base import BaseSimilarityCalculator
from app.core.exceptions import SimilarityCalculationError, ModelLoadingError
from app.core.config_manager import config_manager


class SimilarityCalculatorService(BaseSimilarityCalculator):
    """相似度计算服务"""
    
    def __init__(self):
        super().__init__(config_manager.get_similarity_config())
        self.logger = logging.getLogger(__name__)
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """初始化模型"""
        try:
            model_path = "local_text2vec_model"
            self.model = SentenceTransformer(model_path)
            self.logger.info("文本向量化模型加载成功")
        except Exception as e:
            raise ModelLoadingError(f"模型加载失败: {str(e)}")
    
    def calculate(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        try:
            if not self.model:
                raise SimilarityCalculationError("模型未初始化")
            
            if self.config.ENABLE_MULTI_DIMENSIONAL_SIMILARITY:
                return self._calculate_multi_dimensional_similarity(text1, text2)
            else:
                return self._calculate_semantic_similarity(text1, text2)
                
        except Exception as e:
            if isinstance(e, SimilarityCalculationError):
                raise
            raise SimilarityCalculationError(f"相似度计算失败: {str(e)}")
    
    def calculate_batch(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """批量计算相似度"""
        try:
            if not self.model:
                raise SimilarityCalculationError("模型未初始化")
            
            if len(texts1) != len(texts2):
                raise SimilarityCalculationError("文本列表长度不匹配")
            
            similarities = []
            for text1, text2 in zip(texts1, texts2):
                sim = self.calculate(text1, text2)
                similarities.append(sim)
            
            return similarities
            
        except Exception as e:
            if isinstance(e, SimilarityCalculationError):
                raise
            raise SimilarityCalculationError(f"批量相似度计算失败: {str(e)}")
    
    def _calculate_multi_dimensional_similarity(self, text1: str, text2: str) -> float:
        """计算多维度相似度"""
        weights = self.config.SIMILARITY_WEIGHTS
        
        # 1. 语义相似度
        semantic_sim = self._calculate_semantic_similarity(text1, text2)
        
        # 2. 结构相似度
        structural_sim = self._calculate_structural_similarity(text1, text2)
        
        # 3. 词汇相似度
        lexical_sim = self._calculate_lexical_similarity(text1, text2)
        
        # 加权平均
        final_similarity = (
            weights['semantic'] * semantic_sim +
            weights['structural'] * structural_sim +
            weights['lexical'] * lexical_sim
        )
        
        return final_similarity
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        try:
            embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
            embeddings = embeddings.cpu().numpy()
            return float(np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            ))
        except Exception as e:
            self.logger.error(f"语义相似度计算失败: {str(e)}")
            return 0.0
    
    def _calculate_structural_similarity(self, text1: str, text2: str) -> float:
        """计算结构相似度"""
        import re
        
        # 提取结构特征
        def extract_structure_features(text):
            features = {
                'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
                'sentence_count': len(re.findall(r'[。！？]', text)),
                'avg_sentence_length': len(text) / max(len(re.findall(r'[。！？]', text)), 1),
                'has_numbers': bool(re.search(r'\d+', text)),
                'has_special_chars': bool(re.search(r'[（）【】《》]', text)),
                'line_count': len([l for l in text.split('\n') if l.strip()])
            }
            return features
        
        features1 = extract_structure_features(text1)
        features2 = extract_structure_features(text2)
        
        # 计算结构相似度
        similarities = []
        for key in features1:
            if key in features2:
                val1, val2 = features1[key], features2[key]
                if isinstance(val1, bool) and isinstance(val2, bool):
                    sim = 1.0 if val1 == val2 else 0.0
                else:
                    # 数值特征相似度
                    max_val = max(val1, val2)
                    min_val = min(val1, val2)
                    sim = min_val / max_val if max_val > 0 else 1.0
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_lexical_similarity(self, text1: str, text2: str) -> float:
        """计算词汇相似度"""
        import re
        
        # 简单分词
        def tokenize(text):
            return re.findall(r'[\u4e00-\u9fa50-9a-zA-Z]+', text)
        
        tokens1 = set(tokenize(text1))
        tokens2 = set(tokenize(text2))
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard相似度
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """将文本编码为向量"""
        try:
            if not self.model:
                raise SimilarityCalculationError("模型未初始化")
            
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings.cpu().numpy()
            
        except Exception as e:
            raise SimilarityCalculationError(f"文本编码失败: {str(e)}")
    
    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except Exception as e:
            raise SimilarityCalculationError(f"余弦相似度计算失败: {str(e)}")
