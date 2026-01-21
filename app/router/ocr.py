"""
OCR服务状态API路由
提供OCR服务的状态查询功能（文本提取功能已迁移到 /api/extract）
"""
import logging
from datetime import datetime
from fastapi import APIRouter
from app.service.paddle_ocr_service import ocr_service
from app.models.schemas import OCRStatusResponse, HealthCheckResponse
from app.models.errors import ErrorCode, ErrorMessage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ocr", tags=["OCR服务状态"])

# 支持的文件类型（用于状态信息展示）
SUPPORTED_FILE_TYPES = {
    'application/pdf': '.pdf',
    'image/jpeg': '.jpg',
    'image/jpg': '.jpg', 
    'image/png': '.png',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff'
}


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
