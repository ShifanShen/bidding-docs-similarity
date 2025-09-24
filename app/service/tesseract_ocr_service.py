import io
import logging
import os
import fitz  # pymupdf
from PIL import Image
import numpy as np
import camelot
import pytesseract
from typing import List, Dict, Any
from app.config.tesseract_ocr_config import default_tesseract_config

logger = logging.getLogger(__name__)

class TesseractOCRService:
    """Tesseract OCR文本识别服务"""
    
    def __init__(self):
        self._is_initialized = False
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """初始化Tesseract OCR引擎"""
        try:
            logger.info("开始初始化Tesseract OCR引擎...")
            
            # 检查Tesseract是否可用
            try:
                # 获取Tesseract版本信息以验证安装
                version = pytesseract.get_tesseract_version()
                logger.info(f"Tesseract OCR引擎初始化成功，版本: {version}")
                self._is_initialized = True
            except Exception as e:
                logger.error(f"无法获取Tesseract版本信息: {str(e)}")
                logger.warning("Tesseract可能未正确安装或不在系统PATH中")
                # 尝试继续，让后续错误更明确
                self._is_initialized = False
        except Exception as e:
            logger.error(f"初始化Tesseract OCR引擎失败: {str(e)}")
            self._is_initialized = False
    
    def is_available(self) -> bool:
        """检查OCR服务是否可用"""
        logger.debug("检查Tesseract OCR服务可用性")
        if not self._is_initialized:
            logger.debug("Tesseract OCR服务未初始化，尝试初始化...")
            self._initialize_ocr()
        status = "可用" if self._is_initialized else "不可用"
        logger.debug(f"Tesseract OCR服务状态: {status}")
        return self._is_initialized
    
    def _pdf_page_to_image(self, pdf_path: str, page_num: int) -> np.ndarray:
        """使用pymupdf将PDF页面转换为图像"""
        logger.info(f"开始将PDF页面转换为图像: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]  # pymupdf的页面索引从0开始
            
            # 设置DPI以获得更好的图像质量
            zoom = default_tesseract_config.DPI / 72  # 72是PDF的默认DPI
            mat = fitz.Matrix(zoom, zoom)
            
            logger.debug(f"PDF转图像配置: DPI={default_tesseract_config.DPI}, 缩放比例={zoom}")
            pix = page.get_pixmap(matrix=mat)
            
            # 转换为PIL图像
            img = Image.open(io.BytesIO(pix.tobytes()))
            # 转换为numpy数组
            img_array = np.array(img)
            
            doc.close()
            logger.info(f"PDF页面转换为图像成功: 文件={os.path.basename(pdf_path)}, 页码={page_num}, 图像尺寸={img_array.shape}")
            return img_array
        except Exception as e:
            logger.error(f"PDF页面转换为图像失败: 文件={os.path.basename(pdf_path)}, 页码={page_num}, 错误={str(e)}")
            raise
    
    def _extract_table_with_camelot(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """使用Camelot提取表格"""
        logger.info(f"开始使用Camelot提取表格: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
        tables = []
        
        try:
            # 使用Camelot读取PDF中的表格
            logger.debug(f"Camelot配置: 页码={page_num}, flavor=lattice, line_scale=40")
            extracted_tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_num),
                flavor='lattice',
                line_scale=40
            )
            
            logger.info(f"Camelot提取表格完成，共找到{len(extracted_tables)}个表格候选")
            
            # 处理提取的表格
            valid_tables_count = 0
            for table_idx, table in enumerate(extracted_tables):
                table_data = {
                    'table_idx': table_idx,
                    'cells': []
                }
                
                # 获取表格数据
                df = table.df
                
                # 遍历表格的每个单元格
                cells_count = 0
                for row_idx, row in df.iterrows():
                    for col_idx, cell in enumerate(row):
                        if cell and cell.strip():
                            table_data['cells'].append({
                                'text': cell.strip(),
                                'row': row_idx,
                                'col': col_idx
                            })
                            cells_count += 1
                
                if table_data['cells']:
                    tables.append(table_data)
                    valid_tables_count += 1
                    logger.info(f"成功提取页面 {page_num} 的表格 {table_idx}: 包含{cells_count}个有效单元格, 表格尺寸={df.shape}")
                else:
                    logger.debug(f"跳过页面 {page_num} 的表格 {table_idx}: 无有效单元格")
            
            logger.info(f"表格提取处理完成: 文件={os.path.basename(pdf_path)}, 页码={page_num}, 有效表格数量={valid_tables_count}")
            
        except Exception as e:
            logger.error(f"使用Camelot提取表格失败 (页面 {page_num}): {str(e)}")
        
        return tables
    
    def recognize_text(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """识别PDF页面中的文本，使用Tesseract OCR"""
        temp_image_path = None
        try:
            logger.info(f"开始OCR文本识别: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
            
            img_array = self._pdf_page_to_image(pdf_path, page_num)
            
            # 转换为PIL图像进行OCR处理
            img = Image.fromarray(img_array)
            
            # 调用Tesseract进行文本识别
            logger.info(f"调用Tesseract进行文本识别: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
            
            # 使用配置中的Tesseract参数
            text = pytesseract.image_to_string(img, config=default_tesseract_config.TESSERACT_CONFIG)
            logger.debug(f"使用Tesseract配置: {default_tesseract_config.TESSERACT_CONFIG}")
            
            # 清理文本中的多余空格，特别是中文字符之间的空格
            if text:
                # 使用正则表达式去除中文字符之间的空格
                # 中文字符范围：一-鿿
                import re
                
                # 中文文本清理和格式化
                
                # 1. 首先处理所有行，去除首尾空格
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                # 2. 将所有行合并成一个文本进行整体处理
                merged_text = ' '.join(lines)
                
                # 3. 去除中文字符之间的空格
                merged_text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', merged_text)
                
                # 4. 去除中文与数字/字母之间的空格
                merged_text = re.sub(r'([\u4e00-\u9fff])\s+([0-9a-zA-Z])', r'\1\2', merged_text)
                merged_text = re.sub(r'([0-9a-zA-Z])\s+([\u4e00-\u9fff])', r'\1\2', merged_text)
                
                # 5. 去除数字与标点符号之间的空格
                merged_text = re.sub(r'([0-9])\s+([，。！？；：])', r'\1\2', merged_text)
                merged_text = re.sub(r'([，。！？；：])\s+([0-9])', r'\1\2', merged_text)
                
                # 6. 修复冒号前的空格（包括英文和中文冒号）
                merged_text = re.sub(r'\s+(:|：)', r'\1', merged_text)
                
                # 7. 修复中文标点符号前的空格
                merged_text = re.sub(r'\s+([，。！？；：])', r'\1', merged_text)
                
                # 8. 修复中文标点符号后的多余空格
                merged_text = re.sub(r'([，。！？；：])\s+', r'\1', merged_text)
                
                # 9. 处理连续的多个空格，保留单个空格
                merged_text = re.sub(r'\s+', ' ', merged_text)
                
                # 10. 在中文句号、问号、感叹号后添加换行，以分割句子
                merged_text = re.sub(r'([。！？])', r'\1\n', merged_text)
                
                # 11. 处理"+"等特殊符号周围的空格
                merged_text = re.sub(r'\s*\+\s*', '+', merged_text)
                
                # 8. 再次去除行首尾空格并移除空行
                formatted_lines = [line.strip() for line in merged_text.split('\n') if line.strip()]
                text = '\n'.join(formatted_lines)
                logger.debug("已清理文本中的多余空格")
            
            logger.debug(f"Tesseract OCR识别完成，获取到文本")
            
            # 处理识别结果
            text_lines_count = len(text.strip().split('\n')) if text else 0
            
            # 使用Camelot提取表格
            tables = self._extract_table_with_camelot(pdf_path, page_num)
            
            recognized_text_length = len(text.strip())
            logger.info(f"OCR文本识别完成: 文件={os.path.basename(pdf_path)}, 页码={page_num}, 识别文本长度={recognized_text_length}, 识别文本行数={text_lines_count}, 提取表格数量={len(tables)}")
            
            return {
                'text': text.strip(),
                'tables': tables
            }
            
        except Exception as e:
            logger.error(f"OCR识别失败 (页面 {page_num}): {str(e)}")
            return {
                'text': "",
                'tables': []
            }
    
    def process_page_with_ocr_fallback(self, pdf_path: str, page_num: int, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """当pdfplumber提取的文本不足时，使用OCR作为后备方案"""
        logger.info(f"开始OCR后备处理: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
        
        # 检查当前页面文本长度是否低于阈值
        current_text_length = len(page_data.get('text', ''))
        tables_count = len(page_data.get('tables', []))
        
        logger.debug(f"当前页面状态: 文本长度={current_text_length}, 表格数量={tables_count}, OCR阈值={default_tesseract_config.OCR_THRESHOLD}")
        
        if current_text_length < default_tesseract_config.OCR_THRESHOLD:
            logger.info(f"页面 {page_num} 文本长度 {current_text_length} 低于阈值 {default_tesseract_config.OCR_THRESHOLD}，触发OCR识别")
            
            # 调用OCR识别
            ocr_result = self.recognize_text(pdf_path, page_num)
            
            # 更新文本和表格数据
            if ocr_result.get('text'):
                logger.info(f"OCR识别成功，更新页面文本: 原长度={current_text_length}, 新长度={len(ocr_result['text'])}")
                page_data['text'] = ocr_result['text']
            else:
                logger.warning(f"OCR未能识别到有效文本: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
            
            if ocr_result.get('tables'):
                logger.info(f"OCR识别成功，更新页面表格: 原数量={tables_count}, 新数量={len(ocr_result['tables'])}")
                page_data['tables'] = ocr_result['tables']
        else:
            logger.info(f"页面 {page_num} 文本长度 {current_text_length} 满足要求，无需OCR识别")
        
        logger.debug(f"OCR后备处理完成: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
        return page_data

# 创建全局OCR服务实例
ocr_service = TesseractOCRService()


