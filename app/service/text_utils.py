"""
文本处理工具模块
"""
import os
import re
import tempfile

import docx
import fitz
import pdfplumber
import logging
from typing import List, Dict, Any, Optional
from app.config.similarity_config import default_config
from app.config.synonyms_config import SYNONYMS
from app.service.service_manager import get_oss_service
from app.service.paddle_ocr_service import ocr_service as paddle_ocr_service

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str, page_range: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """从PDF文件中按页提取文本和表格
    
    根据配置决定使用OCR还是pdfplumber：
    - 如果 ENABLE_OCR=True 且OCR服务可用，使用PaddleOCR提取
    - 否则使用pdfplumber提取
    """
    # 检查是否启用OCR
    enable_ocr = getattr(default_config, 'ENABLE_OCR', False)
    
    # 如果未启用OCR，直接使用pdfplumber
    if not enable_ocr:
        logger.info("OCR未启用，使用pdfplumber提取文本")
        return _extract_with_pdfplumber(pdf_path, page_range)
    
    # 如果启用OCR，检查OCR服务是否可用
    if not paddle_ocr_service.is_available():
        logger.warning("OCR已启用但服务不可用，回退到pdfplumber提取")
        return _extract_with_pdfplumber(pdf_path, page_range)
    
    pages = []
    
    try:
        # 首先获取PDF总页数
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        logger.info(f"开始使用OCR提取PDF文本，总页数: {total_pages}")
        
        # 确定要处理的页面范围
        if page_range:
            pages_to_process = [p for p in page_range if 1 <= p <= total_pages]
            logger.info(f"指定页面范围: {page_range}, 有效页面: {pages_to_process}")
        else:
            pages_to_process = list(range(1, total_pages + 1))
            logger.info(f"处理所有页面: 1-{total_pages}")
        
        # 逐页使用OCR提取
        for page_num in pages_to_process:
            try:
                logger.info(f"开始OCR提取第{page_num}页，总页数: {total_pages}")
                ocr_result = paddle_ocr_service.recognize_text(pdf_path, page_num)
                
                text_content = ocr_result.get('text', '')
                segments = split_text_to_segments(text_content) if text_content else []
                logger.info(f"OCR提取第{page_num}页完成，文本长度: {len(text_content)}")
                
                # 按页面格式提取，每页作为一个独立的文本段落
                if text_content.strip():  # 只添加非空页面
                    pages.append({
                        'page_num': page_num,
                        'text': text_content,
                        'segments': segments,
                        'tables': ocr_result.get('tables', [])
                    })
                else:
                    logger.debug(f"第{page_num}页文本为空，跳过")
                
            except IndexError as e:
                logger.error(f"页面索引错误: {str(e)}")
                # 页面索引错误，跳过该页面
                continue
            except Exception as e:
                logger.error(f"OCR提取第{page_num}页失败: {str(e)}")
                # 如果单页OCR失败，不添加空页面，直接跳过
                continue
            
    except Exception as e:
        logger.error(f"OCR提取失败: {str(e)}，回退到pdfplumber")
        return _extract_with_pdfplumber(pdf_path, page_range)
    
    # 检查提取结果
    if not pages:
        logger.warning("OCR提取结果为空，尝试使用pdfplumber")
        return _extract_with_pdfplumber(pdf_path, page_range)
    
    # 统计提取结果
    total_text_length = sum(len(page.get('text', '')) for page in pages)
    non_empty_pages = len(pages)  # 现在pages中只包含非空页面
    
    logger.info(f"OCR提取完成: 有效页数={len(pages)}, 总文本长度={total_text_length}")
    
    return pages

def _extract_with_pdfplumber(pdf_path: str, page_range: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """使用pdfplumber提取PDF文本和表格，并按段落切分"""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        # 确定要处理的页面范围
        if page_range:
            pages_to_process = [p for p in page_range if 1 <= p <= total_pages]
            logger.info(f"pdfplumber提取指定页面范围: {page_range}, 有效页面: {pages_to_process}")
        else:
            pages_to_process = list(range(1, total_pages + 1))
            logger.info(f"pdfplumber提取所有页面: 1-{total_pages}")
        
        # 提取指定页面的文本和表格
        for page_num in pages_to_process:
            try:
                page = pdf.pages[page_num - 1]  # pdfplumber使用0-based索引
                text_content = page.extract_text() or ""
                # 按段落切分，供后续使用
                segments = split_text_to_segments(text_content) if text_content else []
                
                # 提取表格
                tables = []
                try:
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            if table:  # 过滤空表格
                                tables.append({
                                    'data': table,
                                    'rows': len(table),
                                    'cols': len(table[0]) if table else 0
                                })
                except Exception as table_error:
                    logger.debug(f"提取第{page_num}页表格失败: {str(table_error)}")
                
                page_data = {
                    'page_num': page_num,
                    'text': text_content,
                    'segments': segments,
                    'tables': tables
                }
                pages.append(page_data)
                
            except IndexError as e:
                logger.error(f"页面索引错误: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"pdfplumber提取第{page_num}页失败: {str(e)}")
                continue
    
    logger.info(f"pdfplumber提取完成: 有效页数={len(pages)}, 总文本长度={sum(len(p.get('text', '')) for p in pages)}")
    return pages

def extract_kv_tables_from_text(text: str) -> List[Dict[str, Any]]:
    """基于规则从页面文本中抽取近似表格的KV块（最小可用版本）。

    规则：
    - 按行遍历，识别两列样式：
      1) key: value 或 key：value
      2) key  [2个及以上空格]  value
    - 连续满足规则的行数 >= TABLE_MIN_ROWS 视为一个表格块。
    - 返回items（dict）与raw_rows（原始行）。同key保留首次出现的值。
    """
    min_rows = getattr(default_config, 'TABLE_MIN_ROWS', 3)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    blocks: List[Dict[str, Any]] = []

    def parse_kv(line: str) -> Optional[tuple]:
        # 模式1：冒号
        m = re.match(r"^(.{1,40})[：:]\s*(.+)$", line)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        # 模式2：多空格分列
        m2 = re.match(r"^(.{1,40}?)[\t ]{2,}(.+)$", line)
        if m2:
            return m2.group(1).strip(), m2.group(2).strip()
        return None

    current_rows = []
    current_items: Dict[str, str] = {}

    def flush_block():
        nonlocal current_rows, current_items
        if len(current_rows) >= min_rows and current_items:
            blocks.append({'items': dict(current_items), 'raw_rows': list(current_rows)})
        current_rows = []
        current_items = {}

    for line in lines:
        kv = parse_kv(line)
        if kv:
            k, v = kv
            if k not in current_items:
                current_items[k] = v
            current_rows.append(line)
        else:
            # 断开
            flush_block()
    flush_block()

    return blocks

def extract_text_from_docx(docx_path: str) -> List[str]:
    """从Word文件中按段落提取文本"""
    doc = docx.Document(docx_path)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def split_text_to_segments(text: str) -> List[str]:
    """将长文本分割为段落，支持多种检测模式"""
    # 检查检测模式
    detection_mode = getattr(default_config, 'DETECTION_MODE', 'paragraph')
    
    if detection_mode == 'page':
        # 整页模式：不分割，直接返回
        return [text.strip()] if text.strip() else []
    elif detection_mode == 'sentence':
        # 句子模式：按句子分割
        return _split_by_sentences(text)
    elif detection_mode in ('chapter_paragraph', 'chapter'):
        # 按“章/条款/编号段”切分：更适合招标/投标文件（PDF换行多但自然段不明显）
        return _split_by_chapter_paragraphs(text)
    else:
        # 段落模式（默认）：智能段落分割
        return _split_by_paragraphs(text)

def _split_by_sentences(text: str) -> List[str]:
    """按句子分割文本"""
    import re
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # 按句号、问号、感叹号等分割句子
        line_sentences = re.split(r'[。！？；]', line)
        for sent in line_sentences:
            sent = sent.strip()
            if sent and len(sent) >= 50:  # 过滤过短的句子
                sentences.append(sent)
    return sentences

def _split_by_paragraphs(text: str) -> List[str]:
    """按段落分割文本，保持语义完整性
    
    优化策略：
    1. 优先在自然段落边界（空行）处分割
    2. 保持句子完整性，不丢失标点符号
    3. 尽量合并短句形成完整段落
    4. 避免在句子中间分割
    """
    import re
    segments = []
    min_length = default_config.MIN_SEGMENT_LENGTH
    max_length = default_config.MAX_SEGMENT_LENGTH

    # 首先按空行分割成自然段落
    paragraphs = re.split(r'\n\s*\n', text)
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # 如果段落本身已经很长，需要进一步分割
        if len(para) > max_length:
            # 在长段落中，优先在句号处分割，保持句子完整性
            para_segments = _split_long_paragraph(para, min_length, max_length)
            segments.extend(para_segments)
        elif len(para) >= min_length:
            # 段落长度合适，直接添加
            segments.append(para)
        else:
            # 段落太短，尝试与下一个段落合并
            if segments and len(segments[-1]) < max_length * 0.8:
                # 合并到上一个段落
                segments[-1] = segments[-1] + "\n" + para
            else:
                # 无法合并，单独添加（即使小于最小长度）
                segments.append(para)
    
    # 后处理：合并过短的段落
    merged_segments = []
    for seg in segments:
        if len(seg) < min_length and merged_segments:
            # 尝试合并到上一个段落
            if len(merged_segments[-1]) + len(seg) <= max_length:
                merged_segments[-1] = merged_segments[-1] + "\n" + seg
            else:
                merged_segments.append(seg)
        else:
            merged_segments.append(seg)
    
    return [s.strip() for s in merged_segments if s.strip()]


def _split_by_chapter_paragraphs(text: str) -> List[str]:
    """按“章/条款/编号段”切分，尽量做到“按章一段一段切”

    目标场景：
    - PDF 抽取文本通常只有单换行（换行=排版换行），很少出现空行作为自然段边界
    - 招标文件通常有清晰的“章/节/条款编号/列表编号”，更适合用这些作为段落边界

    策略：
    - 遇到“第X章/第X节/第X部分/第X条”等标题行，开始新段
    - 遇到“1、/2、/（1）/1.1/2.3.4”等编号行，开始新段
    - 空行作为硬边界
    - 不做“短段落合并”，避免把多个条款揉成一段
    - 超长段落仍按句子边界进一步切分（不破坏标点）
    """
    import re

    min_length = getattr(default_config, "MIN_SEGMENT_LENGTH", 100)
    max_length = getattr(default_config, "MAX_SEGMENT_LENGTH", 1200)

    # 标题/条款行识别（尽量宽松，覆盖中文招标文件常见格式）
    chapter_re = re.compile(
        r'^\s*(第[一二三四五六七八九十百千0-9]+[章节部分条卷编篇]|[一二三四五六七八九十]+、)\s+'
    )
    # 编号段：1、 2.1 2.3.4 （1） (1) ① 等
    numbered_re = re.compile(
        r'^\s*((\d+([.．]\d+){0,4})|(\d+、)|（\d+）|\(\d+\)|[①②③④⑤⑥⑦⑧⑨⑩])\s*'
    )

    lines = text.splitlines()
    segments: List[str] = []
    cur: List[str] = []

    def flush():
        nonlocal cur
        if not cur:
            return
        seg = "\n".join([l.rstrip() for l in cur]).strip()
        cur = []
        if not seg:
            return
        if len(seg) > max_length:
            segments.extend(_split_long_paragraph(seg, min_length, max_length))
        else:
            segments.append(seg)

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue

        is_boundary = bool(chapter_re.match(line)) or bool(numbered_re.match(line))
        if is_boundary:
            flush()
            cur.append(line)
        else:
            cur.append(line)

    flush()
    return [s for s in segments if s.strip()]


def _split_long_paragraph(text: str, min_length: int, max_length: int) -> List[str]:
    """分割超长段落，保持句子完整性"""
    import re
    segments = []
    current_segment = ""
    
    # 使用正向预查，保留标点符号
    # 匹配句子：以句号、问号、感叹号、分号结尾，但保留这些标点
    sentence_pattern = r'[^。！？；\n]+[。！？；]'
    sentences = re.findall(sentence_pattern, text)
    
    # 处理剩余文本（可能没有标点结尾）
    remaining = re.sub(sentence_pattern, '', text).strip()
    if remaining:
        sentences.append(remaining)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # 超长句子直接添加（虽然不应该出现）
        if len(sentence) > max_length:
            if current_segment:
                segments.append(current_segment.strip())
                current_segment = ""
            segments.append(sentence)
            continue
        
        # 尝试合并到当前段落
        potential_segment = current_segment + sentence if current_segment else sentence
        
        # 如果合并后超过最大长度，先保存当前段落
        if len(potential_segment) > max_length:
            if current_segment:
                segments.append(current_segment.strip())
            current_segment = sentence
        else:
            # 可以合并
            current_segment = potential_segment
            
            # 如果达到最小长度且以句号结尾，可以考虑分割
            # 但优先保持段落完整性，只在接近最大长度时分割
            if len(current_segment) >= min_length:
                if len(current_segment) >= max_length * 0.9:
                    # 接近最大长度，分割
                    segments.append(current_segment.strip())
                    current_segment = ""
    
    # 添加最后一个段落
    if current_segment:
        segments.append(current_segment.strip())
    
    return segments

def remove_stopwords(text: str, stopwords: List[str]) -> str:
    """去除停用词，保留中文文本完整性"""
    return ''.join(char for char in text if char not in stopwords)

def detect_grammar_errors(text: str) -> List[str]:
    """检测文本中的语法错误"""
    return []

def is_order_changed(text1: str, text2: str) -> bool:
    """判断两个文本是否为字符集合相等但顺序不同"""
    return set(text1) == set(text2) and text1 != text2

def is_stopword_evade(text1: str, text2: str, stopwords: List[str]) -> bool:
    """判断两个文本去除停用词后字符集合相等但原文本不同"""
    filtered1 = [char for char in text1 if char not in stopwords]
    filtered2 = [char for char in text2 if char not in stopwords]
    return set(filtered1) == set(filtered2) and text1 != text2

def remove_numbers(text: str) -> str:
    """移除文本中的所有数字"""
    return re.sub(r'[0-9]+', '', text)

def is_synonym_evade(text1: str, text2: str) -> bool:
    """判断两个文本是否通过替换同义词来规避相似度检测"""
    # 构建反向同义词映射
    reverse_synonyms = {}
    for key, syn_list in SYNONYMS.items():
        for syn in syn_list:
            if syn not in reverse_synonyms:
                reverse_synonyms[syn] = []
            reverse_synonyms[syn].append(key)
    
    # 简单分词
    def simple_tokenize(text):
        return re.findall(r'[\u4e00-\u9fa50-9a-zA-Z]+', text)
    
    tokens1 = simple_tokenize(text1)
    tokens2 = simple_tokenize(text2)
    
    # 检查词数差异
    if abs(len(tokens1) - len(tokens2)) / max(len(tokens1), len(tokens2)) > default_config.SYNONYM_TOKEN_COUNT_DIFF_THRESHOLD:
        return False
    
    # 检查同义词替换
    synonym_count = 0
    for i in range(min(len(tokens1), len(tokens2))):
        token1, token2 = tokens1[i], tokens2[i]
        if token1 == token2:
            continue
        
        # 检查是否是同义词
        is_syn = (
            (token1 in SYNONYMS and token2 in SYNONYMS[token1]) or
            (token2 in SYNONYMS and token1 in SYNONYMS[token2]) or
            (token1 in reverse_synonyms and token2 in reverse_synonyms[token1]) or
            (token2 in reverse_synonyms and token1 in reverse_synonyms[token2])
        )
        
        if is_syn:
            synonym_count += 1
    
    # 判断是否为规避行为
    total_tokens = max(len(tokens1), len(tokens2))
    return total_tokens > 0 and synonym_count / total_tokens > 0.1

def text_result_highlight_local(pdf_path: str, result_json: Dict[str, Any], pdf_name: Optional[str] = None) -> str:
    """文本高亮函数（本地路径版本）

    返回：高亮后的 PDF 文件路径
    """
    if not pdf_name:
        pdf_name = os.path.basename(pdf_path)

    # 提取json文件中的details
    details = []
    all_details = result_json["result"]["result"].get("details", [])
    details = [d for d in all_details if d.get("bid_file") == pdf_name]
    if not details:
        logger.warning(f"'{pdf_name}'不存在相似文本")
        return pdf_path

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_output:
        output_path = tmp_output.name

    # 打开PDF
    doc = fitz.open(pdf_path)

    # 定义三种高亮颜色
    colors = [(1, 1, 0), (0, 1, 1),(1, 0, 1)]
    opacities = [0.3, 0.3, 0.3]

    for i, detail in enumerate(details):
        # 提取匹配信息
        page_num = detail.get('page', 1)
        search_text = detail.get('text', '').strip()
        similar_with = detail.get('similar_with', '').strip()
        similarity = detail.get('similarity', 0)
        similar_page = detail.get('similar_page', '')

        # 在指定页面搜索文本
        page_idx = int(page_num) - 1
        page = doc[page_idx]
        text_instances = page.search_for(search_text)

        # 为当前detail选择一种颜色
        color_index = i % len(colors)
        color = colors[color_index]
        opacity = opacities[color_index]

        # 准备注释信息
        similar_filename = similar_with.rsplit('.', 1)[0]
        if len(similar_filename) > 10: similar_filename = similar_filename[:8] + ".."
        note_text = f"{similar_filename}|{similarity:.1%}|页{similar_page}"
        text_length = len(note_text)
        added_annotation = False

        # 切分文本
        delimiters = r'[，。、；：？！【】（）《》" ",\.;:?!\[\]\(\)\s\t\n\r]'
        segments = re.split(delimiters, search_text)

        # 过滤空字符串和太短的片段
        segments = [seg.strip() for seg in segments if seg.strip() and len(seg.strip()) >= 2]

        # 高亮文本
        for segment in segments:
            # 搜索每个片段
            text_instances = page.search_for(segment.strip())
            if text_instances:
                for rect in text_instances:
                    shape = page.new_shape()
                    shape.draw_rect(rect)
                    shape.finish(fill=color, fill_opacity=opacity)
                    shape.commit()
                    if not added_annotation:
                        note_width = max(100, text_length * 7)
                        note_height = 10
                        # 计算注释位置
                        note_x0 = rect.x0 + 5
                        note_y0 = rect.y0 - note_height - 3
                        note_x1 = note_x0 + note_width
                        note_y1 = note_y0 + note_height
                        # 如果上方空间不够,则放在下方
                        page_rect = page.rect
                        if note_y0 < page_rect.y0:
                            note_y0 = rect.y0 + rect.height + 5
                            note_y1 = note_y0 + note_height
                        # 确保注释框在页面水平方向内
                        if note_x1 > page_rect.x1:
                            note_x0 = max(page_rect.x0, rect.x0 - note_width - 5)
                            note_x1 = note_x0 + note_width
                        note_rect = fitz.Rect(note_x0, note_y0, note_x1, note_y1)
                        # 添加注释
                        page.add_freetext_annot(
                            note_rect,
                            f"↓{note_text}",
                            fontsize=7,
                            text_color=(0,0,0)
                        )
                        added_annotation = True

    # 保存高亮后的PDF
    doc.save(output_path)
    doc.close()
    return output_path


def text_result_highlight(pdf_oss_url: str, result_json: Dict[str, Any]) -> str:
    """兼容旧逻辑：传入 MinIO URL，输出高亮后的 MinIO URL（若 MinIO 不可用则抛出明确错误）"""
    pdf_name = os.path.basename(pdf_oss_url)
    oss = get_oss_service()
    if not oss.is_available():
        raise ConnectionError(f"MinIO不可用，无法从URL高亮: endpoint={oss.config.endpoint}")

    # 下载 -> 本地高亮 -> 上传
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_input:
        input_path = tmp_input.name

    try:
        if not oss.file_exists(pdf_name):
            raise FileNotFoundError(f"MinIO中不存在'{pdf_name}'")
        oss.download_file(pdf_name, input_path)
        out_path = text_result_highlight_local(input_path, result_json, pdf_name=pdf_name)
        return oss.upload_file(out_path, pdf_name)
    finally:
        try:
            if os.path.exists(input_path):
                os.unlink(input_path)
        except Exception:
            pass