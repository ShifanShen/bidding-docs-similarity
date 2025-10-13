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
        self._ocr = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """初始化PaddleOCR引擎"""
        try:
            logger.info("开始初始化PaddleOCR引擎...")
            
            # 创建PaddleOCR实例，使用您配置的模型
            self._ocr = PaddleOCR(
                text_detection_model_name=default_paddle_ocr_config.TEXT_DETECTION_MODEL_NAME,
                text_recognition_model_name=default_paddle_ocr_config.TEXT_RECOGNITION_MODEL_NAME,
                use_doc_orientation_classify=default_paddle_ocr_config.USE_DOC_ORIENTATION_CLASSIFY,
                use_doc_unwarping=default_paddle_ocr_config.USE_DOC_UNWARPING,
                use_textline_orientation=default_paddle_ocr_config.USE_TEXTLINE_ORIENTATION
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
            total_pages = len(doc)
            
            # 检查页面索引是否有效
            if page_num < 1 or page_num > total_pages:
                raise IndexError(f"页面索引超出范围: {page_num}，总页数: {total_pages}")
            
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
            if not ocr_result:
                logger.debug("OCR结果为空")
                return ""
            
            logger.debug(f"OCR结果类型: {type(ocr_result)}")
            
            # 从PaddleOCR结果中提取文本
            text_lines = []
            
            # 新的PaddleOCR返回格式：OCRResult对象，包含json属性
            if ocr_result and len(ocr_result) > 0:
                ocr_result_obj = ocr_result[0]
                
                # 检查是否有json属性
                if hasattr(ocr_result_obj, 'json'):
                    json_data = ocr_result_obj.json
                    logger.debug(f"JSON数据键: {list(json_data.keys())}")
                    
                    # 检查是否有res字段（新的数据格式）
                    if 'res' in json_data:
                        res_data = json_data['res']
                        logger.debug(f"res数据类型: {type(res_data)}")
                        
                        # 从res数据中提取文本
                        if isinstance(res_data, dict) and 'rec_texts' in res_data and res_data['rec_texts']:
                            rec_texts = res_data['rec_texts']
                            rec_scores = res_data.get('rec_scores', [])
                            
                            logger.debug(f"res中rec_texts数量: {len(rec_texts)}")
                            logger.debug(f"res中rec_scores数量: {len(rec_scores)}")
                            
                            # 确保文本和分数数量一致
                            min_length = min(len(rec_texts), len(rec_scores)) if rec_scores else len(rec_texts)
                            
                            for i in range(min_length):
                                text_content = rec_texts[i] if i < len(rec_texts) else ""
                                confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                                
                                logger.debug(f"识别文本: '{text_content}', 置信度: {confidence:.3f}")
                                
                                # 只保留置信度高于阈值且非空的文本
                                if confidence >= default_paddle_ocr_config.MIN_CONFIDENCE and text_content.strip():
                                    text_lines.append(text_content.strip())
                                    logger.debug(f"✅ 保留文本: '{text_content}'")
                                else:
                                    logger.debug(f"❌ 丢弃文本: '{text_content}' (置信度: {confidence:.3f})")
                        else:
                            logger.debug("res数据中没有rec_texts或为空")
                    
                    # 兼容旧格式：直接从JSON数据中提取文本
                    elif 'rec_texts' in json_data and json_data['rec_texts']:
                        rec_texts = json_data['rec_texts']
                        rec_scores = json_data.get('rec_scores', [])
                        
                        logger.debug(f"JSON中rec_texts数量: {len(rec_texts)}")
                        logger.debug(f"JSON中rec_scores数量: {len(rec_scores)}")
                        
                        # 确保文本和分数数量一致
                        min_length = min(len(rec_texts), len(rec_scores)) if rec_scores else len(rec_texts)
                        
                        for i in range(min_length):
                            text_content = rec_texts[i] if i < len(rec_texts) else ""
                            confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                            
                            logger.debug(f"识别文本: '{text_content}', 置信度: {confidence:.3f}")
                            
                            # 只保留置信度高于阈值且非空的文本
                            if confidence >= default_paddle_ocr_config.MIN_CONFIDENCE and text_content.strip():
                                text_lines.append(text_content.strip())
                                logger.debug(f"✅ 保留文本: '{text_content}'")
                            else:
                                logger.debug(f"❌ 丢弃文本: '{text_content}' (置信度: {confidence:.3f})")
                    else:
                        logger.debug("JSON数据中没有rec_texts或为空")
            
            # 合并所有文本行
            full_text = '\n'.join(text_lines)
            logger.debug(f"提取到文本行数: {len(text_lines)}")
            logger.debug(f"合并文本长度: {len(full_text)}")
            
            # 简单清理文本
            if full_text:
                import re
                # 去除多余空格，保留基本格式
                full_text = re.sub(r'\s+', ' ', full_text).strip()
                logger.debug("已清理文本中的多余空格")
            
            return full_text
            
        except Exception as e:
            logger.error(f"处理PaddleOCR识别结果失败: {str(e)}")
            return ""
    
    def recognize_text(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """识别PDF页面中的文本，使用PaddleOCR"""
        try:
            logger.info(f"开始OCR文本识别: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
            
            img_array = self._pdf_page_to_image(pdf_path, page_num)
            
            # 调用PaddleOCR进行文本识别
            logger.info(f"调用PaddleOCR进行文本识别: 文件={os.path.basename(pdf_path)}, 页码={page_num}")
            
            ocr_result = self._ocr.ocr(img_array)
            
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
    
    def recognize_image(self, image_path: str) -> Dict[str, Any]:
        """直接识别图像文件中的文本"""
        try:
            logger.info(f"开始OCR图像识别: {os.path.basename(image_path)}")
            
            # 读取图像文件
            from PIL import Image
            import numpy as np
            
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # 调用PaddleOCR进行文本识别
            logger.info(f"调用PaddleOCR进行图像识别: {os.path.basename(image_path)}")
            
            ocr_result = self._ocr.ocr(img_array)
            
            # 处理识别结果
            text = self._process_paddle_ocr_result(ocr_result)
            
            logger.debug(f"PaddleOCR图像识别完成，获取到文本")
            
            # 处理识别结果
            text_lines_count = len(text.strip().split('\n')) if text else 0
            
            recognized_text_length = len(text.strip())
            logger.info(f"OCR图像识别完成: 文件={os.path.basename(image_path)}, 识别文本长度={recognized_text_length}, 识别文本行数={text_lines_count}")
            
            return {
                'text': text.strip(),
                'tables': []  # 不再识别表格，返回空列表
            }
            
        except Exception as e:
            logger.error(f"OCR图像识别失败: {str(e)}")
            return {
                'text': "",
                'tables': []
            }
    
    def process_page_with_ocr_fallback(self, pdf_path: str, page_num: int, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """当pdfplumber提取的文本不足时，使用OCR作为后备方案"""
        current_text_length = len(page_data.get('text', '').strip())
        
        if current_text_length < default_paddle_ocr_config.OCR_THRESHOLD:
            logger.info(f"页面 {page_num} 文本长度不足，触发OCR识别")
            ocr_result = self.recognize_text(pdf_path, page_num)
            
            if ocr_result.get('text'):
                page_data['text'] = ocr_result['text']
                logger.info(f"OCR识别成功，文本长度: {len(ocr_result['text'])}")
        
        return page_data

# 创建全局OCR服务实例
ocr_service = PaddleOCRService()
