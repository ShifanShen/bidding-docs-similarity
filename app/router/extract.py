"""
文本提取API路由
提供PDF和图像文件的文本提取功能（OCR和pdfplumber）
"""
import logging
import time
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from app.service.paddle_ocr_service import ocr_service
from app.service.text_utils import extract_text_from_pdf
from app.models.schemas import OCRExtractResponse
from app.models.errors import ErrorCode, ErrorMessage, get_error_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/extract", tags=["文本提取"])

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
    "/text",
    response_model=OCRExtractResponse,
    summary="提取文本",
    description="提取文本（multipart/form-data：file，page_range，可选 extract_tables/use_ocr）"
)
async def extract_text(
    file: UploadFile = File(..., description="PDF或图像文件"),
    page_range: Optional[str] = Form(None, description="页面范围，如'1-3'或'1,3,5'，默认所有页面"),
    extract_tables: bool = Form(False, description="是否提取表格"),
    use_ocr: bool = Form(False, description="是否强制使用OCR")
):
    """提取文本"""
    import os
    from app.config.similarity_config import default_config
    
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
        
        # 确定是否使用OCR
        enable_ocr = use_ocr
        if not enable_ocr:
            enable_ocr = getattr(default_config, 'ENABLE_OCR', False)
        
        # 如果强制使用OCR，检查OCR服务可用性
        if use_ocr and not ocr_service.is_available():
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
            
            start_time = time.time()
            
            # 根据文件类型处理
            if file.content_type == 'application/pdf':
                # 解析页面范围
                pages_to_process = parse_page_range(page_range)
                
                # 临时修改配置以强制使用OCR（如果需要）
                original_enable_ocr = getattr(default_config, 'ENABLE_OCR', False)
                if use_ocr:
                    setattr(default_config, 'ENABLE_OCR', True)
                
                try:
                    # 使用text_utils提取文本
                    pages_data = extract_text_from_pdf(temp_file_path, pages_to_process)
                finally:
                    # 恢复原始配置
                    if use_ocr:
                        setattr(default_config, 'ENABLE_OCR', original_enable_ocr)
                
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
                
            else:
                # 处理图像文件
                if not ocr_service.is_available():
                    raise HTTPException(
                        status_code=ErrorCode.SERVICE_UNAVAILABLE,
                        detail=get_error_response(
                            ErrorCode.SERVICE_UNAVAILABLE,
                            ErrorMessage.OCR_SERVICE_UNAVAILABLE
                        )
                    )
                
                ocr_result = ocr_service.recognize_image(temp_file_path)
                
                result = {
                    "text": ocr_result.get('text', ''),
                    "text_length": len(ocr_result.get('text', '')),
                    "tables": ocr_result.get('tables', []),
                    "tables_count": len(ocr_result.get('tables', [])),
                    "processing_time": 0
                }
            
            result["processing_time"] = time.time() - start_time
            
            # 添加文件元数据
            result.update({
                "filename": file.filename,
                "file_size": len(content),
                "file_type": file.content_type,
                "extraction_method": "OCR" if enable_ocr else "pdfplumber"
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


def parse_page_range(page_range: Optional[str]) -> Optional[list]:
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

