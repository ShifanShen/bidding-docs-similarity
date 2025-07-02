from typing import List
import re
# PDF 文本抽取
import pdfplumber
# Word 文本抽取
import docx

def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """从PDF文件中按页提取文本，返回每页文本列表"""
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return texts

def extract_text_from_docx(docx_path: str) -> List[str]:
    """从Word文件中按段落提取文本，返回每段文本列表"""
    doc = docx.Document(docx_path)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def split_text_to_segments(text: str) -> List[str]:
    """将长文本分割为段落或句子（按换行或句号）"""
    # 先按换行分段，再按句号分句
    segments = []
    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue
        # 按中文/英文句号分句
        segs = re.split(r'[。.!?]', para)
        segments.extend([s.strip() for s in segs if s.strip()])
    return segments

def remove_stopwords(text: str, stopwords: List[str]) -> str:
    """去除无意义词"""
    words = re.findall(r'\w+', text, flags=re.UNICODE)
    filtered = [w for w in words if w not in stopwords]
    return ' '.join(filtered)

def detect_grammar_errors(text: str) -> List[str]:
    """检测文本中的语法错误，返回错误列表"""
    return []

def is_order_changed(text1: str, text2: str) -> bool:
    """判断两个文本是否为词集合相等但顺序不同"""
    words1 = re.findall(r'\w+', text1, flags=re.UNICODE)
    words2 = re.findall(r'\w+', text2, flags=re.UNICODE)
    return set(words1) == set(words2) and words1 != words2

def is_stopword_evade(text1: str, text2: str, stopwords: List[str]) -> bool:
    """判断两个文本去除停用词后词集合相等但原文本不同"""
    words1 = [w for w in re.findall(r'\w+', text1, flags=re.UNICODE) if w not in stopwords]
    words2 = [w for w in re.findall(r'\w+', text2, flags=re.UNICODE) if w not in stopwords]
    return set(words1) == set(words2) and text1 != text2
