from typing import List, Dict, Any
import re
import docx
import pdfplumber
from typing import List, Dict, Any
from app.config.similarity_config import default_config
from app.config.synonyms_config import SYNONYMS
from app.service.paddle_ocr_service import ocr_service as paddle_ocr_service

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """从PDF文件中按页提取文本和表格，返回每页内容列表"""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_data = {
                'page_num': page.page_number,
                'text': page.extract_text() or "",
                'tables': []
            }
            
            # 提取表格
            if hasattr(page, 'extract_tables'):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    table_data = {
                        'table_idx': table_idx,
                        'cells': []
                    }
                    for row_idx, row in enumerate(table):
                        for col_idx, cell in enumerate(row):
                            if cell:
                                table_data['cells'].append({
                                    'text': cell.strip(),
                                    'row': row_idx,
                                    'col': col_idx
                                })
                    if table_data['cells']:
                        page_data['tables'].append(table_data)
            
            # 当pdfplumber提取的文本长度低于阈值时，触发OCR识别
            if paddle_ocr_service.is_available():
                try:
                    page_data = paddle_ocr_service.process_page_with_ocr_fallback(pdf_path, page.page_number, page_data)
                except Exception as e:
                    # 记录OCR错误但继续使用pdfplumber的结果
                    import logging
                    logging.error(f"处理页面 {page.page_number} 的OCR识别失败: {str(e)}")
            
            pages.append(page_data)
    return pages

def extract_text_from_docx(docx_path: str) -> List[str]:
    """从Word文件中按段落提取文本，返回每段文本列表"""
    doc = docx.Document(docx_path)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def split_text_to_segments(text: str) -> List[str]:
    """将长文本分割为段落，保持上下文完整性"""
    # 先按换行分段
    segments = []
    current_segment = ""
    min_segment_length = default_config.MIN_SEGMENT_LENGTH  # 最小段落长度阈值
    max_segment_length = default_config.MAX_SEGMENT_LENGTH  # 最大段落长度阈值

    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue

        # 如果当前段落已经很长，直接添加
        if len(para) > max_segment_length:
            if current_segment:
                segments.append(current_segment)
                current_segment = ""
            segments.append(para)
            continue

        # 合并到当前段落
        if current_segment:
            current_segment += " " + para
        else:
            current_segment = para

        # 如果当前段落达到最小长度，添加到结果
        if len(current_segment) >= min_segment_length:
            segments.append(current_segment)
            current_segment = ""

    # 添加最后一个段落（如果有）
    if current_segment:
        segments.append(current_segment)

    return segments

def remove_stopwords(text: str, stopwords: List[str]) -> str:
    """去除无意义词，保留中文文本完整性"""
    # 对于中文文本，直接遍历每个字符去除停用词
    result = []
    for char in text:
        if char not in stopwords:
            result.append(char)
    return ''.join(result)

def detect_grammar_errors(text: str) -> List[str]:
    """检测文本中的语法错误，返回错误列表"""
    return []

def is_order_changed(text1: str, text2: str) -> bool:
    """判断两个文本是否为字符集合相等但顺序不同"""
    # 对于中文文本，直接比较字符集合
    return set(text1) == set(text2) and text1 != text2

def is_stopword_evade(text1: str, text2: str, stopwords: List[str]) -> bool:
    """判断两个文本去除停用词后字符集合相等但原文本不同"""
    # 对于中文文本，直接遍历每个字符去除停用词
    filtered1 = [char for char in text1 if char not in stopwords]
    filtered2 = [char for char in text2 if char not in stopwords]
    return set(filtered1) == set(filtered2) and text1 != text2

def remove_numbers(text: str) -> str:
    """移除文本中的所有数字，保留其他字符"""
    # 使用正则表达式替换所有数字为''
    return re.sub(r'[0-9]+', '', text)

def is_synonym_evade(text1: str, text2: str) -> bool:
    """判断两个文本是否通过替换同义词来规避相似度检测"""
    # 从配置文件导入同义词表
    
    # 构建反向同义词映射
    reverse_synonyms = {}
    for key, syn_list in SYNONYMS.items():
        for syn in syn_list:
            if syn not in reverse_synonyms:
                reverse_synonyms[syn] = []
            reverse_synonyms[syn].append(key)
        
    # 简单分词（按空格和标点符号）
    def simple_tokenize(text):
        # 保留中文、数字、字母，其他视为分隔符
        tokens = re.findall(r'[\u4e00-\u9fa50-9a-zA-Z]+', text)
        return tokens
    
    tokens1 = simple_tokenize(text1)
    tokens2 = simple_tokenize(text2)
    
    # 如果词数差异超过阈值，不认为是同义词替换
    if abs(len(tokens1) - len(tokens2)) / max(len(tokens1), len(tokens2)) > default_config.SYNONYM_TOKEN_COUNT_DIFF_THRESHOLD:
        return False
    
    # 检查同义词替换
    synonym_count = 0
    for i in range(min(len(tokens1), len(tokens2))):
        token1 = tokens1[i]
        token2 = tokens2[i]
        
        if token1 == token2:
            continue
        
        # 检查是否是同义词
        is_syn = False
        if token1 in SYNONYMS and token2 in SYNONYMS[token1]:
            is_syn = True
        elif token2 in SYNONYMS and token1 in SYNONYMS[token2]:
            is_syn = True
        elif token1 in reverse_synonyms and token2 in reverse_synonyms[token1]:
            is_syn = True
        elif token2 in reverse_synonyms and token1 in reverse_synonyms[token2]:
            is_syn = True
        
        if is_syn:
            synonym_count += 1
    
    # 如果同义词替换数量超过总词数的10%，则认为是规避行为
    total_tokens = max(len(tokens1), len(tokens2))
    if total_tokens == 0:
        return False
    
    return synonym_count / total_tokens > 0.1
