"""
OCR文本提取API路由
提供PDF和图像文件的OCR文本识别功能
"""
import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from app.service.paddle_ocr_service import ocr_service
from app.service.text_utils import extract_text_from_pdf
from app.models.schemas import OCRExtractResponse, OCRStatusResponse, HealthCheckResponse
from app.models.errors import ErrorCode, ErrorMessage, get_error_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ocr", tags=["OCR文本提取"])

# 支持的文件类型
SUPPORTED_FILE_TYPES = {
    'application/pdf': '.pdf',
    'image/jpeg': '.jpg',
    'image/jpg': '.jpg', 
    'image/png': '.png',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff'
}


@router.post(
    "/upload-and-extract",
    response_model=OCRExtractResponse,
    summary="上传文件并提取文本",
    description="上传并提取文本（multipart/form-data: file，page_range 等可选）"
)
async def upload_and_extract_text(
    file: UploadFile = File(..., description="PDF或图像文件"),
    page_range: Optional[str] = Form(None, description="页面范围，如'1-3'或'1,3,5'，默认所有页面"),
    extract_tables: bool = Form(False, description="是否提取表格"),
    confidence_threshold: float = Form(0.5, description="OCR置信度阈值", ge=0.0, le=1.0)
):
    """上传文件并提取文本"""
    try:
        # 检查文件类型
        if file.content_type not in SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=ErrorCode.BAD_REQUEST,
                detail=get_error_response(
                    ErrorCode.BAD_REQUEST,
                    ErrorMessage.FILE_TYPE_NOT_SUPPORTED,
                    f"不支持的文件类型: {file.content_type}。支持的类型: {list(SUPPORTED_FILE_TYPES.keys())}"
                )
            )
        
        # 检查OCR服务可用性
        if not ocr_service.is_available():
            raise HTTPException(
                status_code=ErrorCode.SERVICE_UNAVAILABLE,
                detail=get_error_response(
                    ErrorCode.SERVICE_UNAVAILABLE,
                    ErrorMessage.OCR_SERVICE_UNAVAILABLE
                )
            )
        
        # 保存上传的文件
        file_extension = SUPPORTED_FILE_TYPES[file.content_type]
        temp_file_path = f"temp_upload_{file.filename}{file_extension}"
        
        try:
            # 保存文件内容
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info(f"文件上传成功: {file.filename}, 大小: {len(content)} 字节")
            
            # 根据文件类型处理
            if file.content_type == 'application/pdf':
                result = await process_pdf_file(temp_file_path, page_range, extract_tables, confidence_threshold)
            else:
                result = await process_image_file(temp_file_path, confidence_threshold)
            
            # 添加文件元数据
            result.update({
                "filename": file.filename,
                "file_size": len(content),
                "file_type": file.content_type,
                "processing_time": result.get("processing_time", 0)
            })
            
            return OCRExtractResponse(
                code=ErrorCode.SUCCESS,
                msg="提取成功",
                data=result
            )
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {str(e)}")
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件处理失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, ErrorMessage.OCR_PROCESSING_FAILED, str(e))
        )


async def process_pdf_file(file_path: str, page_range: Optional[str], extract_tables: bool, confidence_threshold: float) -> Dict[str, Any]:
    """处理PDF文件"""
    start_time = time.time()
    
    try:
        # 解析页面范围
        pages_to_process = parse_page_range(page_range)
        
        # 使用text_utils提取文本，直接传递页面范围
        pages_data = extract_text_from_pdf(file_path, pages_to_process)
        
        # 处理结果
        result = {
            "total_pages": len(pages_data),
            "pages": [],
            "full_text": "",
            "tables": [],
            "processing_time": 0
        }
        
        all_texts = []
        all_tables = []
        
        for page_data in pages_data:
            page_result = {
                "page_num": page_data['page_num'],
                "text": page_data.get('text', ''),
                "text_length": len(page_data.get('text', '')),
                "tables_count": len(page_data.get('tables', []))
            }
            
            if extract_tables:
                page_result["tables"] = page_data.get('tables', [])
                all_tables.extend(page_data.get('tables', []))
            
            result["pages"].append(page_result)
            all_texts.append(page_data.get('text', ''))
        
        # 合并所有文本
        result["full_text"] = '\n\n'.join(all_texts)
        result["full_text_length"] = len(result["full_text"])
        
        if extract_tables:
            result["tables"] = all_tables
            result["total_tables"] = len(all_tables)
        
        result["processing_time"] = time.time() - start_time
        return result
        
    except Exception as e:
        logger.error(f"PDF处理失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, "PDF处理失败", str(e))
        )


async def process_image_file(file_path: str, confidence_threshold: float) -> Dict[str, Any]:
    """处理图像文件"""
    start_time = time.time()
    
    try:
        # 使用PaddleOCR处理图像
        ocr_result = ocr_service.recognize_image(file_path)
        
        result = {
            "text": ocr_result.get('text', ''),
            "text_length": len(ocr_result.get('text', '')),
            "tables": ocr_result.get('tables', []),
            "tables_count": len(ocr_result.get('tables', [])),
            "processing_time": time.time() - start_time
        }
        
        return result
        
    except Exception as e:
        logger.error(f"图像处理失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, "图像处理失败", str(e))
        )


def parse_page_range(page_range: Optional[str]) -> Optional[List[int]]:
    """解析页面范围字符串"""
    if page_range is None:
        return None
    
    try:
        # 兼容空字符串或全空白
        if isinstance(page_range, str) and not page_range.strip():
            return None

        pages = []
        for part in page_range.split(','):
            part = part.strip()
            if not part:
                # 跳过空段（例如尾逗号 "1-3,"）
                continue
            if '-' in part:
                # 处理范围，如"1-3"
                start, end = map(int, part.split('-'))
                pages.extend(range(start, end + 1))
            else:
                # 处理单页
                pages.append(int(part))
        return sorted(set(pages))  # 去重并排序
    except ValueError:
        raise HTTPException(
            status_code=ErrorCode.BAD_REQUEST,
            detail=get_error_response(ErrorCode.BAD_REQUEST, ErrorMessage.INVALID_PAGE_RANGE)
        )


@router.get(
    "/status",
    response_model=OCRStatusResponse,
    summary="获取OCR服务状态",
    description="查询OCR状态"
)
async def get_ocr_status():
    """获取OCR服务状态"""
    return OCRStatusResponse(
        code=ErrorCode.SUCCESS,
        msg=ErrorMessage.SUCCESS,
        data={
            "ocr_available": ocr_service.is_available(),
            "supported_file_types": list(SUPPORTED_FILE_TYPES.keys()),
            "max_file_size": "50MB",
            "supported_languages": ["中文", "英文"]
        }
    )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="健康检查",
    description="健康检查"
)
async def health_check():
    """健康检查端点"""
    return HealthCheckResponse(
        code=ErrorCode.SUCCESS,
        msg="服务正常",
        data={
            "status": "healthy",
            "ocr_service": "available" if ocr_service.is_available() else "unavailable",
            "timestamp": datetime.now().isoformat()
        }
    )
