from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import tempfile
from docx import Document
import fitz  # PyMuPDF
import shutil
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

import re
import jieba
import jieba.posseg as pseg
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pathlib import Path


# 加载预训练的 BERT 模型和分词器
# 使用本地BERT模型
local_model_path = "./local_bert_model"
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
model = AutoModel.from_pretrained(local_model_path, local_files_only=True)

# 初始化 FastAPI 应用
app = FastAPI(
    title="文档相似度检测 API",
    description="提供文档上传和相似度检测功能",
    version="1.0.0",
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 创建临时目录
TEMP_DIR = Path(tempfile.gettempdir()) / "bidding_docs"
TEMP_DIR.mkdir(exist_ok=True)




# 加载中文停用词
stopwords_path = Path(__file__).parent / 'stopwords.txt'
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())

# 配置jieba
jieba.initialize()
def preprocess_text(text: str) -> str:
    """预处理文本，增强语义相关性"""
    original_text = text  # 保存原始文本用于保底
    # 1. 移除特殊字符和标点符号
    text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
    # 2. 使用jieba进行词性标注并过滤非名词性词语
    words = pseg.cut(text)
    # 3. 过滤停用词、单字和非名词性词语
    # 放宽过滤条件以保留更多有意义词汇
 # 移除长度过滤以保留单字词
    # 进一步放宽过滤条件，保留更多潜在有意义的词
    # 移除词性过滤以保留更多词汇
    filtered_words = [word for word, flag in words if word.strip() and word not in stopwords]
    # 处理空文本情况
    if not filtered_words:
        # 使用原始文本哈希作为保底内容，确保唯一性
        return f'default_content_{hash(text) % 1000000}'
    # 重新组合为字符串
    text = ' '.join(filtered_words)
    # 6. 移除多余空白
    text = ' '.join(text.split())
    # 确保最终文本不为空
    if not text.strip():
        return f'default_content_{hash(original_text) % 1000000}'
    return text

# 文档解析函数
async def extract_text_from_file(file_path: str) -> str:
    """从不同类型的文件中提取文本内容并进行预处理"""
    if file_path.endswith('.docx'):
        doc = Document(file_path)
        raw_text = '\n'.join([para.text for para in doc.paragraphs])
    elif file_path.endswith('.pdf'):
        # 使用 PyMuPDF 提取 PDF 文本
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        raw_text = text
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    else:
        raise ValueError(f"不支持的文件类型: {file_path.split('.')[-1]}")
    
    # 应用文本预处理
    return preprocess_text(raw_text)


# 定义函数来获取文本的嵌入向量
def get_embeddings(text):
    # 对于长文本，使用滑动窗口获取多个嵌入并平均
    max_length = tokenizer.model_max_length
    stride = 256  # 滑动窗口步长
    embeddings = []
    
    # 如果文本过短，直接处理
    if len(text) <= max_length:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()
    
    # 长文本滑动窗口处理
    for i in range(0, len(text), stride):
        end = i + max_length
        chunk = text[i:end]
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
    # 使用平均池化
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask, 1)
    sum_mask = input_mask.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_embedding = sum_embeddings / sum_mask
    chunk_embedding = mean_embedding.numpy()
    embeddings.append(chunk_embedding)
    
    # 平均所有窗口的嵌入向量
    embeddings = np.vstack(embeddings)
    return np.mean(embeddings, axis=0, keepdims=True)

# 定义文件上传和相似度检测的 API 端点
@app.post("/upload-and-compare")
async def upload_and_compare(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # 保存上传的文件到临时目录
    file1_path = TEMP_DIR / file1.filename
    file2_path = TEMP_DIR / file2.filename

    with open(file1_path, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
    with open(file2_path, "wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)

    # 提取文本内容
    text1 = await extract_text_from_file(str(file1_path))
    text2 = await extract_text_from_file(str(file2_path))
    print(text1)
    print(text2)
    # 获取文本嵌入
    embeddings1 = get_embeddings(text1)
    embeddings2 = get_embeddings(text2)
    
    # 处理可能的空文本情况
    if not text1.strip() or not text2.strip():
        # 添加基础内容防止空词汇表错误
        text1 = text1 or 'default_content'
        text2 = text2 or 'default_content'
    # 确保文本不为空以避免空词汇表错误
    text1 = text1.strip() or 'default_content_1'
    text2 = text2.strip() or 'default_content_2'
    # 计算TF-IDF加权嵌入
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    tfidf_weights1 = tfidf_matrix[0].toarray().flatten()
    tfidf_weights2 = tfidf_matrix[1].toarray().flatten()
    
    # 应用TF-IDF权重到BERT嵌入
    weighted_embeddings1 = embeddings1 * tfidf_weights1[:, np.newaxis]
    weighted_embeddings2 = embeddings2 * tfidf_weights2[:, np.newaxis]
    
    # 基础余弦相似度
    base_similarity = cosine_similarity(weighted_embeddings1, weighted_embeddings2)[0][0]
    
    # TF-IDF关键词提取
    keywords1 = jieba.analyse.extract_tags(text1, topK=20, withWeight=False)
    keywords2 = jieba.analyse.extract_tags(text2, topK=20, withWeight=False)
    keyword_overlap = len(set(keywords1) & set(keywords2)) / max(len(set(keywords1) | set(keywords2)), 1)

    # 2. 文本内容重叠度检查
    # 提取关键词
    keywords1 = set(jieba.cut_for_search(text1))
    keywords2 = set(jieba.cut_for_search(text2))
    common_keywords_ratio = len(keywords1 & keywords2) / max(len(keywords1 | keywords2), 1)

    # 结合关键词重叠度调整相似度
    similarity = base_similarity * (0.4 + 0.6 * keyword_overlap)
    
    # 添加语义角色过滤
    if common_keywords_ratio < 0.05 and similarity < 0.3:
        similarity *= 0.5  # 对语义相关性低的文档应用额外惩罚
    
    # 添加相似度阈值校准
    # 根据投标文档特点调整阈值，降低非相关文档的相似度分数
    # 1. 文本长度比率过滤 - 过滤长度差异过大的文档
    length_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2)) if max(len(text1), len(text2)) > 0 else 0
    if length_ratio < 0.3:
        similarity *= 0.3  # 对长度差异大的文档应用强惩罚
    
    # 增强关键词重叠惩罚机制
    if common_keywords_ratio < 0.15:
        similarity *= 0.3  # 更强的关键词重叠惩罚
    elif common_keywords_ratio < 0.3:
        similarity *= 0.6  # 中等关键词重叠惩罚
    
    # 3. 增强版动态阈值调整
    text1_length = len(text1)
    text2_length = len(text2)
    avg_length = (text1_length + text2_length) / 2
    # 增强长文档惩罚机制
    if avg_length > 1000:
        # 对长文档应用基础惩罚
        similarity *= 0.8
        # 根据相似度区间应用额外惩罚
        if similarity < 0.45:
            similarity *= 0.5
        elif 0.45 <= similarity < 0.75:
            similarity = similarity * 0.6 + 0.05  # 提高区分度
    else:
        if similarity < 0.35:
            similarity *= 0.4
        elif 0.35 <= similarity < 0.65:
            similarity = similarity * 0.6 + 0.15
    
    # 计算文本内容重叠度
    text1_words = set(text1.split())
    text2_words = set(text2.split())
    if len(text1_words) > 0 and len(text2_words) > 0:
        content_overlap = len(text1_words & text2_words) / len(text1_words | text2_words)
        # 根据内容重叠度调整相似度
        similarity = similarity * (0.3 + 0.7 * content_overlap)
    
    # 多维度相似度调整
    # 1. 文档类型过滤 - 投标文档特定过滤
    bidding_keywords = {'投标', '招标', '采购', '合同', '项目', '技术方案', '报价', '预算', '中标', '评标'}
    doc1_bidding_score = sum(1 for kw in bidding_keywords if kw in text1)
    doc2_bidding_score = sum(1 for kw in bidding_keywords if kw in text2)
    if (doc1_bidding_score + doc2_bidding_score) < 2:
        similarity *= 0.3  # 投标关键词不足时强惩罚
    
    # 2. 最终阈值限制
    similarity = max(0.0, min(1.0, similarity))
    
    # 清理临时文件
    os.remove(file1_path)
    os.remove(file2_path)
    
    return {"similarity": similarity, "confidence": "high" if similarity > 0.88 else "medium" if similarity > 0.65 else "low", "content_overlap": round(common_keywords_ratio, 2), "keyword_overlap": round(keyword_overlap, 2), "bidding_score": doc1_bidding_score + doc2_bidding_score}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)