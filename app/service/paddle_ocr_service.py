import io
import logging
import os
import fitz  # pymupdf
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict, Any
from app.config.paddle_ocr_config import default_paddle_ocr_config

logger = logging.getLogger(__name__)

class PaddleOCRService:
    """PaddleOCR文本识别服务"""
    
    def __init__(self):
        self._is_initialized = False
        self._pipeline = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """初始化PaddleOCR引擎"""
        try:
            logger.info("开始初始化PaddleOCR引擎...")
            
            # 创建PaddleOCR实例
            self._pipeline = PaddleOCR(
                use_angle_cls=default_paddle_ocr_config.USE_ANGLE_CLS,
                lang=default_paddle_ocr_config.LANG,
                device=default_paddle_ocr_config.DEVICE
            )
            
            logger.info("PaddleOCR引擎初始化成功")
            self._is_initialized = True
            
        except Exception as e:
            logger.error(f"初始化PaddleOCR引擎失败: {str(e)}")
            self._is_initialized = False
    
    def is_available(self) -> bool:
        """检查OCR服务是否可用"""
        logger.debug("检查PaddleOCR服务可用性")
        if not self._is_initialized:
            logger.debug("PaddleOCR服务未初始化，尝试初始化...")
            self._initialize_ocr()
        status = "可用" if self._is_initialized else "不可用"
        logger.debug(f"PaddleOCR服务状态: {status}")
        return self._is_initialized
    
    def _pdf_page_to_image(self, pdf_path: str, page_num: int) -> np.ndarray:
        """使用pymupdf将PDF页面转换为图像"""
        logger.info(f"开始将PDF页面转换为图像: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]  # pymupdf的页面索引从0开始
            
            # 设置DPI以获得更好的图像质量
            zoom = default_paddle_ocr_config.DPI / 72  # 72是PDF的默认DPI
            mat = fitz.Matrix(zoom, zoom)
            
            logger.debug(f"PDF转图像配置: DPI={default_paddle_ocr_config.DPI}, 缩放比例={zoom}")
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
    
    
    def _process_paddle_ocr_result(self, ocr_result) -> str:
        """处理PaddleOCR识别结果，提取文本内容"""
        try:
            if not ocr_result or len(ocr_result) == 0:
                return ""
            
            # 从PaddleOCR结果中提取文本
            text_lines = []
            for line in ocr_result:
                if line and len(line) >= 2:
                    # PaddleOCR结果格式: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
                    text_content = line[1][0] if line[1] else ""
                    confidence = line[1][1] if line[1] and len(line[1]) > 1 else 0.0
                    
                    # 只保留置信度高于阈值的文本
                    if confidence >= default_paddle_ocr_config.MIN_CONFIDENCE and text_content.strip():
                        text_lines.append(text_content.strip())
            
            # 合并所有文本行
            full_text = '\n'.join(text_lines)
            
            # 清理文本中的多余空格，特别是中文字符之间的空格
            if full_text:
                import re
                
                # 1. 首先处理所有行，去除首尾空格
                lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                
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
                
                # 12. 再次去除行首尾空格并移除空行
                formatted_lines = [line.strip() for line in merged_text.split('\n') if line.strip()]
                full_text = '\n'.join(formatted_lines)
                logger.debug("已清理文本中的多余空格")
            
            return full_text
            
        except Exception as e:
            logger.error(f"处理PaddleOCR识别结果失败: {str(e)}")
            return ""
    
    def recognize_text(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """识别PDF页面中的文本，使用PaddleOCR"""
        temp_image_path = None
        try:
            logger.info(f"开始OCR文本识别: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
            
            img_array = self._pdf_page_to_image(pdf_path, page_num)
            
            # 保存临时图像文件供PaddleOCR处理
            temp_image_path = f"temp_ocr_{page_num}.png"
            temp_img = Image.fromarray(img_array)
            temp_img.save(temp_image_path)
            
            # 调用PaddleOCR进行文本识别
            logger.info(f"调用PaddleOCR进行文本识别: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
            
            ocr_result = self._pipeline.predict(temp_image_path)
            
            # 处理识别结果
            text = self._process_paddle_ocr_result(ocr_result)
            
            logger.debug(f"PaddleOCR识别完成，获取到文本")
            
            # 处理识别结果
            text_lines_count = len(text.strip().split('\n')) if text else 0
            
            recognized_text_length = len(text.strip())
            logger.info(f"OCR文本识别完成: 文件={os.path.basename(pdf_path)}, 页码={page_num}, 识别文本长度={recognized_text_length}, 识别文本行数={text_lines_count}")
            
            return {
                'text': text.strip(),
                'tables': []  # 不再识别表格，返回空列表
            }
            
        except Exception as e:
            logger.error(f"OCR识别失败 (页面 {page_num}): {str(e)}")
            return {
                'text': "",
                'tables': []
            }
        finally:
            # 清理临时文件
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {str(e)}")
    
    def process_page_with_ocr_fallback(self, pdf_path: str, page_num: int, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """当pdfplumber提取的文本不足时，使用OCR作为后备方案"""
        logger.info(f"开始OCR后备处理: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
        
        # 检查当前页面文本长度是否低于阈值
        current_text_length = len(page_data.get('text', ''))
        tables_count = len(page_data.get('tables', []))
        
        logger.debug(f"当前页面状态: 文本长度={current_text_length}, 表格数量={tables_count}, OCR阈值={default_paddle_ocr_config.OCR_THRESHOLD}")
        
        if current_text_length < default_paddle_ocr_config.OCR_THRESHOLD:
            logger.info(f"页面 {page_num} 文本长度 {current_text_length} 低于阈值 {default_paddle_ocr_config.OCR_THRESHOLD}，触发OCR识别")
            
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
ocr_service = PaddleOCRService()
