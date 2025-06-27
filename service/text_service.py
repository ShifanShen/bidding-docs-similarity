# 文本处理和相似度计算服务
import re
import os
import jieba
import jieba.posseg as pseg
import jieba.analyse  # Add this line to import the analyse module
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from app.config import Config

class TextProcessingService:
    @staticmethod
    def preprocess_text(text: str) -> str:
        # 实现原main.py中的文本预处理逻辑
        text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
        words = pseg.cut(text)
        filtered_words = [word for word, flag in words if word.strip() and word not in Config.STOPWORDS]
        return ' '.join(filtered_words)

    @staticmethod
    async def extract_text_from_file(file_path: str) -> str:
        """从不同类型文件中提取文本内容"""
        if file_path.endswith('.docx'):
            from docx import Document
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        elif file_path.endswith('.pdf'):
            import fitz
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"不支持的文件类型: {file_path.split('.')[-1]}")

class SimilarityService:
    @staticmethod
    def get_embeddings(text: str):
        # 从app模块导入已初始化的模型和分词器
        from app import tokenizer, model
        max_length = tokenizer.model_max_length
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> dict:
        # 实现原main.py中的相似度计算逻辑
        embeddings1 = SimilarityService.get_embeddings(text1)
        embeddings2 = SimilarityService.get_embeddings(text2)
        base_similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
        
        # 关键词提取和重叠计算
        keywords1 = jieba.analyse.extract_tags(text1, topK=20)
        keywords2 = jieba.analyse.extract_tags(text2, topK=20)
        keyword_overlap = len(set(keywords1) & set(keywords2)) / max(len(set(keywords1) | set(keywords2)), 1)
        
        # 综合相似度计算
        similarity = base_similarity * 0.7 + keyword_overlap * 0.3
        
        # 应用惩罚机制
        if len(text1) > 1000 or len(text2) > 1000:
            similarity *= 0.9  # 长文档惩罚
        
        threshold_level = "high" if similarity > 0.85 else "medium" if similarity > 0.65 else "low"
        
        return {
            "similarity_score": float(similarity),
            "threshold_level": threshold_level,
            "keyword_overlap": float(keyword_overlap),
            "analysis": f"文档相似度: {similarity:.2f}, 关键词重叠度: {keyword_overlap:.2f}"
        }

# 导出服务实例
text_service = TextProcessingService()
similarity_service = SimilarityService()