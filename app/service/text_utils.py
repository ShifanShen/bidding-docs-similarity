from typing import List, Dict, Any
import re
import docx
import pdfplumber
from app.config.similarity_config import default_config
from app.config.synonyms_config import SYNONYMS
from app.service.paddle_ocr_service import ocr_service as paddle_ocr_service

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """从PDF文件中按页提取文本和表格，返回每页内容列表（智能混合策略）"""
    pages = []
    
    # 根据配置选择提取策略
    if default_config.EXTRACTION_MODE == "ocr":
        return extract_text_with_ocr_only(pdf_path)
    elif default_config.EXTRACTION_MODE == "pdfplumber":
        return extract_text_with_pdfplumber_only(pdf_path)
    else:  # hybrid mode
        return extract_text_with_smart_hybrid(pdf_path)

def extract_text_with_ocr_only(pdf_path: str) -> List[Dict[str, Any]]:
    """纯OCR提取方式"""
    pages = []
    
    if not paddle_ocr_service.is_available():
        import logging
        logging.error("OCR服务不可用，无法进行纯OCR提取")
        return pages
    
    try:
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        
        for page_num in range(1, len(doc) + 1):
            try:
                ocr_result = paddle_ocr_service.recognize_text(pdf_path, page_num)
                page_data = {
                    'page_num': page_num,
                    'text': ocr_result.get('text', ''),
                    'tables': [],  # OCR不处理表格
                    'extraction_method': 'ocr'
                }
                pages.append(page_data)
            except Exception as e:
                import logging
                logging.error(f"OCR提取页面 {page_num} 失败: {str(e)}")
                pages.append({
                    'page_num': page_num,
                    'text': '',
                    'tables': [],
                    'extraction_method': 'ocr_failed'
                })
        
        doc.close()
    except Exception as e:
        import logging
        logging.error(f"纯OCR提取失败: {str(e)}")
    
    return pages

def extract_text_with_pdfplumber_only(pdf_path: str) -> List[Dict[str, Any]]:
    """纯pdfplumber提取方式（原有逻辑）"""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_data = {
                'page_num': page.page_number,
                'text': page.extract_text() or "",
                'tables': [],
                'extraction_method': 'pdfplumber'
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
            
            pages.append(page_data)
    
    return pages

def extract_text_with_smart_hybrid(pdf_path: str) -> List[Dict[str, Any]]:
    """智能混合提取策略"""
    pages = []
    
    # 第一步：使用pdfplumber提取
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_data = {
                'page_num': page.page_number,
                'text': page.extract_text() or "",
                'tables': [],
                'extraction_method': 'pdfplumber'
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
            
            # 第二步：智能判断是否需要OCR
            should_use_ocr = should_use_ocr_for_page(page_data, pdf_path)
            
            if should_use_ocr and paddle_ocr_service.is_available():
                try:
                    ocr_result = paddle_ocr_service.recognize_text(pdf_path, page.page_number)
                    if ocr_result.get('text'):
                        # 比较两种方法的结果，选择更好的
                        pdfplumber_text = page_data['text']
                        ocr_text = ocr_result['text']
                        
                        if is_ocr_result_better(pdfplumber_text, ocr_text):
                            page_data['text'] = ocr_text
                            page_data['extraction_method'] = 'ocr'
                            import logging
                            logging.info(f"页面 {page.page_number} 使用OCR提取，文本长度: {len(ocr_text)}")
                        else:
                            import logging
                            logging.info(f"页面 {page.page_number} 使用pdfplumber提取，文本长度: {len(pdfplumber_text)}")
                except Exception as e:
                    import logging
                    logging.error(f"页面 {page.page_number} OCR处理失败: {str(e)}")
            
            pages.append(page_data)
    
    return pages

def should_use_ocr_for_page(page_data: Dict[str, Any], pdf_path: str) -> bool:
    """判断页面是否需要使用OCR"""
    text_length = len(page_data.get('text', '').strip())
    
    # 1. 文本长度检查
    if text_length < default_config.OCR_FALLBACK_THRESHOLD:
        return True
    
    # 2. 扫描PDF检查
    if default_config.ENABLE_OCR_FOR_SCANNED_PDF and is_likely_scanned_pdf(page_data):
        return True
    
    # 3. 复杂布局检查
    if default_config.ENABLE_OCR_FOR_COMPLEX_LAYOUT and has_complex_layout(page_data):
        return True
    
    # 4. 表格密集页面检查
    if len(page_data.get('tables', [])) > 3:
        return True
    
    return False

def is_likely_scanned_pdf(page_data: Dict[str, Any]) -> bool:
    """判断是否可能是扫描PDF"""
    text = page_data.get('text', '')
    
    # 扫描PDF的特征：
    # 1. 文本很少但页面存在
    # 2. 包含OCR常见的错误模式
    # 3. 格式不规整
    
    if len(text) < 100:  # 文本很少
        return True
    
    # 检查是否包含OCR常见错误
    ocr_error_patterns = ['口', '□', '■', '●', '○']  # OCR常见错误字符
    if any(pattern in text for pattern in ocr_error_patterns):
        return True
    
    return False

def has_complex_layout(page_data: Dict[str, Any]) -> bool:
    """判断是否有复杂布局"""
    text = page_data.get('text', '')
    
    # 复杂布局的特征：
    # 1. 多列布局
    # 2. 大量特殊字符
    # 3. 不规整的换行
    
    # 检查多列布局（通过空格数量判断）
    lines = text.split('\n')
    if len(lines) > 10:
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        if avg_line_length < 20:  # 平均行长度很短，可能是多列
            return True
    
    # 检查特殊字符密度
    special_chars = len([c for c in text if c in '（）【】《》①②③④⑤⑥⑦⑧⑨⑩'])
    if special_chars > len(text) * 0.1:  # 特殊字符超过10%
        return True
    
    return False

def is_ocr_result_better(pdfplumber_text: str, ocr_text: str) -> bool:
    """判断OCR结果是否比pdfplumber结果更好"""
    pdfplumber_len = len(pdfplumber_text.strip())
    ocr_len = len(ocr_text.strip())
    
    # 1. 长度比较：OCR结果明显更长
    if ocr_len > pdfplumber_len * 1.5:
        return True
    
    # 2. 质量比较：OCR结果包含更多有效内容
    if pdfplumber_len < 50 and ocr_len > 100:
        return True
    
    # 3. 错误率比较：pdfplumber结果包含太多错误字符
    pdfplumber_errors = sum(1 for c in pdfplumber_text if c in '口□■●○')
    ocr_errors = sum(1 for c in ocr_text if c in '口□■●○')
    
    if pdfplumber_errors > ocr_errors * 2:
        return True
    
    return False

def extract_text_from_docx(docx_path: str) -> List[str]:
    """从Word文件中按段落提取文本，返回每段文本列表"""
    doc = docx.Document(docx_path)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def split_text_to_segments(text: str) -> List[str]:
    """将长文本分割为段落，保持上下文完整性"""
    if default_config.ENABLE_SMART_SEGMENTATION:
        return smart_segment_text(text)
    else:
        return basic_segment_text(text)

def basic_segment_text(text: str) -> List[str]:
    """基础分段逻辑（原有逻辑）"""
    segments = []
    current_segment = ""
    min_segment_length = default_config.MIN_SEGMENT_LENGTH
    max_segment_length = default_config.MAX_SEGMENT_LENGTH

    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue

        if len(para) > max_segment_length:
            if current_segment:
                segments.append(current_segment)
                current_segment = ""
            segments.append(para)
            continue

        if current_segment:
            current_segment += " " + para
        else:
            current_segment = para

        if len(current_segment) >= min_segment_length:
            segments.append(current_segment)
            current_segment = ""

    if current_segment:
        segments.append(current_segment)

    return segments

def smart_segment_text(text: str) -> List[str]:
    """智能分段逻辑：基于语义边界和文档结构"""
    segments = []
    
    # 第一步：识别文档结构
    structured_text = identify_document_structure(text)
    
    # 第二步：按结构分段
    for section in structured_text:
        if section['type'] == 'paragraph':
            # 段落：按句子边界智能分割
            para_segments = segment_paragraph_by_sentences(section['content'])
            segments.extend(para_segments)
        elif section['type'] == 'list':
            # 列表：保持列表项完整性
            segments.append(section['content'])
        elif section['type'] == 'table':
            # 表格：按行处理
            segments.append(section['content'])
        elif section['type'] == 'title':
            # 标题：与下一段合并
            if segments and len(segments[-1]) < default_config.MIN_SEGMENT_LENGTH:
                segments[-1] += " " + section['content']
            else:
                segments.append(section['content'])
    
    return segments

def identify_document_structure(text: str) -> List[Dict[str, Any]]:
    """识别文档结构：标题、段落、列表、表格等"""
    import re
    
    sections = []
    lines = text.split('\n')
    current_paragraph = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_paragraph:
                sections.append({
                    'type': 'paragraph',
                    'content': '\n'.join(current_paragraph)
                })
                current_paragraph = []
            continue
        
        # 识别标题（数字开头、短行、包含"章"、"节"等）
        if is_title(line):
            if current_paragraph:
                sections.append({
                    'type': 'paragraph',
                    'content': '\n'.join(current_paragraph)
                })
                current_paragraph = []
            sections.append({
                'type': 'title',
                'content': line
            })
        # 识别列表项
        elif is_list_item(line):
            if current_paragraph:
                sections.append({
                    'type': 'paragraph',
                    'content': '\n'.join(current_paragraph)
                })
                current_paragraph = []
            sections.append({
                'type': 'list',
                'content': line
            })
        # 识别表格行
        elif is_table_row(line):
            if current_paragraph:
                sections.append({
                    'type': 'paragraph',
                    'content': '\n'.join(current_paragraph)
                })
                current_paragraph = []
            sections.append({
                'type': 'table',
                'content': line
            })
        else:
            current_paragraph.append(line)
    
    # 处理最后一个段落
    if current_paragraph:
        sections.append({
            'type': 'paragraph',
            'content': '\n'.join(current_paragraph)
        })
    
    return sections

def is_title(line: str) -> bool:
    """判断是否为标题"""
    # 短行且包含章节标识
    if len(line) < 50 and any(keyword in line for keyword in ['第', '章', '节', '条', '款', '项']):
        return True
    # 数字开头的短行
    if len(line) < 30 and re.match(r'^\d+[\.、]', line):
        return True
    return False

def is_list_item(line: str) -> bool:
    """判断是否为列表项"""
    # 以数字、字母、符号开头的短行
    if re.match(r'^[\d\w\-\*•]\s*[\.、]', line) and len(line) < 100:
        return True
    return False

def is_table_row(line: str) -> bool:
    """判断是否为表格行"""
    # 包含多个制表符或空格分隔的列
    if '\t' in line or line.count('  ') >= 3:
        return True
    return False

def segment_paragraph_by_sentences(paragraph: str) -> List[str]:
    """按句子边界分割段落"""
    segments = []
    current_segment = ""
    
    # 按句子边界分割
    sentences = split_by_sentence_boundaries(paragraph)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # 如果当前段落加上新句子超过最大长度，保存当前段落
        if len(current_segment) + len(sentence) > default_config.MAX_SEGMENT_LENGTH and current_segment:
            segments.append(current_segment.strip())
            current_segment = sentence
        else:
            if current_segment:
                current_segment += sentence
            else:
                current_segment = sentence
        
        # 如果当前段落达到最小长度，添加到结果
        if len(current_segment) >= default_config.MIN_SEGMENT_LENGTH:
            segments.append(current_segment.strip())
            current_segment = ""
    
    # 添加最后一个段落
    if current_segment:
        segments.append(current_segment.strip())
    
    return segments

def split_by_sentence_boundaries(text: str) -> List[str]:
    """按句子边界分割文本"""
    import re
    
    # 使用正则表达式按句子边界分割
    pattern = r'([。！？；：])'
    parts = re.split(pattern, text)
    
    sentences = []
    for i in range(0, len(parts), 2):
        if i < len(parts):
            sentence = parts[i]
            if i + 1 < len(parts):
                sentence += parts[i + 1]  # 添加标点符号
            if sentence.strip():
                sentences.append(sentence.strip())
    
    return sentences

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
