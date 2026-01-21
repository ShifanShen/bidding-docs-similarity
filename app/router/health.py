"""
健康检查API路由
提供系统健康状态检查功能
"""
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/health", tags=["健康检查"])


@router.get("/", summary="系统健康检查", description="检查系统各服务的健康状态")
async def health_check():
    """系统健康检查"""
    try:
        health_status = {
            "status": "healthy",
            "services": {}
        }
        
        # 检查相似度分析服务
        try:
            from app.service.service_manager import get_similarity_service
            service = get_similarity_service()
            health_status["services"]["similarity"] = {
                "available": True,
                "model_loaded": service.model is not None
            }
        except Exception as e:
            health_status["services"]["similarity"] = {
                "available": False,
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # 检查实体识别服务
        try:
            from app.service.service_manager import get_entity_rec_service
            service = get_entity_rec_service()
            health_status["services"]["entity_recognition"] = {
                "available": True,
                "hanlp_available": service.is_hanlp_available()
            }
        except Exception as e:
            health_status["services"]["entity_recognition"] = {
                "available": False,
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # 检查正则实体识别服务
        try:
            from app.service.service_manager import get_entity_regex_service
            _ = get_entity_regex_service()
            health_status["services"]["entity_regex"] = {
                "available": True
            }
        except Exception as e:
            health_status["services"]["entity_regex"] = {
                "available": False,
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # 检查OCR服务
        try:
            from app.service.paddle_ocr_service import ocr_service
            health_status["services"]["ocr"] = {
                "available": ocr_service.is_available()
            }
        except Exception as e:
            health_status["services"]["ocr"] = {
                "available": False,
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )
