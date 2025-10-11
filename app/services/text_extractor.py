"""
文本提取服务
"""
import os
import logging
from typing import List, Dict, Any, Optional
from app.core.base import BaseExtractor, DocumentSegment, ExtractionMethod
from app.core.exceptions import TextExtractionError, OCRServiceError
from app.core.config_manager import config_manager
from app.service.text_utils import (
    extract_text_with_ocr_only,
    extract_text_with_pdfplumber_only,
    extract_text_with_smart_hybrid
)


class TextExtractorService(BaseExtractor):
    """文本提取服务"""
    
    def __init__(self):
        super().__init__(config_manager.get_similarity_config())
        self.logger = logging.getLogger(__name__)
    
    def extract(self, file_path: str) -> List[DocumentSegment]:
        """提取文档片段"""
        try:
            if not os.path.exists(file_path):
                raise TextExtractionError(f"文件不存在: {file_path}")
            
            # 根据文件类型选择提取方法
            ext = os.path.splitext(file_path)[-1].lower()
            if ext == '.pdf':
                return self._extract_from_pdf(file_path)
            elif ext == '.docx':
                return self._extract_from_docx(file_path)
            else:
                raise TextExtractionError(f"不支持的文件类型: {ext}")
                
        except Exception as e:
            if isinstance(e, TextExtractionError):
                raise
            raise TextExtractionError(f"文本提取失败: {str(e)}")
    
    def _extract_from_pdf(self, file_path: str) -> List[DocumentSegment]:
        """从PDF提取文本"""
        extraction_mode = self.config.EXTRACTION_MODE
        
        if extraction_mode == "ocr":
            pages = extract_text_with_ocr_only(file_path)
        elif extraction_mode == "pdfplumber":
            pages = extract_text_with_pdfplumber_only(file_path)
        else:  # hybrid
            pages = extract_text_with_smart_hybrid(file_path)
        
        return self._convert_pages_to_segments(pages)
    
    def _extract_from_docx(self, file_path: str) -> List[DocumentSegment]:
        """从Word文档提取文本"""
        try:
            import docx
            doc = docx.Document(file_path)
            segments = []
            
            for i, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    segment = DocumentSegment(
                        page=i + 1,
                        text=para.text.strip(),
                        grammar_errors=[],
                        extraction_method=ExtractionMethod.PDFPLUMBER
                    )
                    segments.append(segment)
            
            return segments
            
        except Exception as e:
            raise TextExtractionError(f"Word文档提取失败: {str(e)}")
    
    def _convert_pages_to_segments(self, pages: List[Dict[str, Any]]) -> List[DocumentSegment]:
        """将页面数据转换为文档片段"""
        segments = []
        
        for page_data in pages:
            page_num = page_data.get('page_num', 0)
            text = page_data.get('text', '')
            extraction_method_str = page_data.get('extraction_method', 'pdfplumber')
            
            # 转换提取方法
            try:
                extraction_method = ExtractionMethod(extraction_method_str)
            except ValueError:
                extraction_method = ExtractionMethod.PDFPLUMBER
            
            if text.strip():
                segment = DocumentSegment(
                    page=page_num,
                    text=text.strip(),
                    grammar_errors=[],
                    extraction_method=extraction_method
                )
                segments.append(segment)
            
            # 处理表格
            for table in page_data.get('tables', []):
                table_idx = table.get('table_idx', 0)
                for cell in table.get('cells', []):
                    if cell.get('text', '').strip():
                        segment = DocumentSegment(
                            page=page_num,
                            text=cell['text'].strip(),
                            grammar_errors=[],
                            is_table_cell=True,
                            row=cell.get('row'),
                            col=cell.get('col'),
                            table_idx=table_idx,
                            extraction_method=extraction_method
                        )
                        segments.append(segment)
        
        return segments
    
    def is_available(self) -> bool:
        """检查提取器是否可用"""
        try:
            # 检查必要的依赖
            import pdfplumber
            import docx
            return True
        except ImportError:
            return False
