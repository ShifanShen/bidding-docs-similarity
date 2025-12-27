"""
配置管理API路由
提供系统配置的查询和更新功能
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from app.config.similarity_config import default_config, SimilarityConfig
from app.config.paddle_ocr_config import default_paddle_ocr_config, PaddleOCRConfig
from app.models.schemas import ConfigGetResponse, ConfigUpdateResponse, ConfigUpdateRequest
from app.models.errors import ErrorCode, ErrorMessage, get_error_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/config", tags=["配置管理"])


@router.get(
    "/get",
    response_model=ConfigGetResponse,
    summary="获取配置",
    description="查询配置（config_type: similarity/ocr/all）"
)
async def get_config(config_type: str = "all"):
    """获取配置"""
    try:
        config_type = config_type.lower()
        
        if config_type == "similarity":
            # 获取相似度配置
            config_dict = {
                "STORAGE_DIR": default_config.STORAGE_DIR,
                "MAX_CONCURRENT_TASKS": default_config.MAX_CONCURRENT_TASKS,
                "MAX_TASK_TIMEOUT": default_config.MAX_TASK_TIMEOUT,
                "MIN_TEXT_LENGTH": default_config.MIN_TEXT_LENGTH,
                "MIN_SEGMENT_LENGTH": default_config.MIN_SEGMENT_LENGTH,
                "MAX_SEGMENT_LENGTH": default_config.MAX_SEGMENT_LENGTH,
                "BATCH_SIZE": default_config.BATCH_SIZE,
                "ENABLE_OCR": default_config.ENABLE_OCR,
                "DETECTION_MODE": default_config.DETECTION_MODE,
                "PAGE_LEVEL_DETECTION": default_config.PAGE_LEVEL_DETECTION,
                "ENABLE_TABLE_DETECTION": default_config.ENABLE_TABLE_DETECTION,
                "TABLE_MIN_ROWS": default_config.TABLE_MIN_ROWS,
                "TENDER_SIMILARITY_THRESHOLD": default_config.TENDER_SIMILARITY_THRESHOLD,
                "BID_SIMILARITY_THRESHOLD": default_config.BID_SIMILARITY_THRESHOLD,
                "NEAR_IDENTICAL_THRESHOLD": default_config.NEAR_IDENTICAL_THRESHOLD,
                "HIGH_SIMILARITY_THRESHOLD": default_config.HIGH_SIMILARITY_THRESHOLD,
                "VERY_HIGH_SIMILARITY_THRESHOLD": default_config.VERY_HIGH_SIMILARITY_THRESHOLD,
                "SEMANTIC_EVADE_LOWER_THRESHOLD": default_config.SEMANTIC_EVADE_LOWER_THRESHOLD,
                "SEMANTIC_EVADE_UPPER_THRESHOLD": default_config.SEMANTIC_EVADE_UPPER_THRESHOLD,
                "SYNONYM_TOKEN_COUNT_DIFF_THRESHOLD": default_config.SYNONYM_TOKEN_COUNT_DIFF_THRESHOLD,
                "COMMON_TERM_COUNT_THRESHOLD": default_config.COMMON_TERM_COUNT_THRESHOLD,
                "ENABLE_GPU": default_config.ENABLE_GPU,
                "MEMORY_CLEANUP_INTERVAL": default_config.MEMORY_CLEANUP_INTERVAL,
                "SIMILARITY_TOP_K": default_config.SIMILARITY_TOP_K,
                "LOG_LEVEL": default_config.LOG_LEVEL,
            }
            
            return ConfigGetResponse(
                code=ErrorCode.SUCCESS,
                msg=ErrorMessage.SUCCESS,
                data={
                    "config_type": "similarity",
                    "config": config_dict
                }
            )
            
        elif config_type == "ocr":
            # 获取OCR配置
            config_dict = {
                "OCR_THRESHOLD": default_paddle_ocr_config.OCR_THRESHOLD,
                "DPI": default_paddle_ocr_config.DPI,
                "MAX_IMAGE_WIDTH": default_paddle_ocr_config.MAX_IMAGE_WIDTH,
                "MAX_IMAGE_HEIGHT": default_paddle_ocr_config.MAX_IMAGE_HEIGHT,
                "MAX_PIXELS": default_paddle_ocr_config.MAX_PIXELS,
                "TEXT_DETECTION_MODEL_NAME": default_paddle_ocr_config.TEXT_DETECTION_MODEL_NAME,
                "TEXT_RECOGNITION_MODEL_NAME": default_paddle_ocr_config.TEXT_RECOGNITION_MODEL_NAME,
                "USE_DOC_ORIENTATION_CLASSIFY": default_paddle_ocr_config.USE_DOC_ORIENTATION_CLASSIFY,
                "USE_DOC_UNWARPING": default_paddle_ocr_config.USE_DOC_UNWARPING,
                "USE_TEXTLINE_ORIENTATION": default_paddle_ocr_config.USE_TEXTLINE_ORIENTATION,
                "LANG": default_paddle_ocr_config.LANG,
                "DEVICE": default_paddle_ocr_config.DEVICE,
                "MIN_CONFIDENCE": default_paddle_ocr_config.MIN_CONFIDENCE,
                "TEXT_REC_SCORE_THRESH": default_paddle_ocr_config.TEXT_REC_SCORE_THRESH,
            }
            
            return ConfigGetResponse(
                code=ErrorCode.SUCCESS,
                msg=ErrorMessage.SUCCESS,
                data={
                    "config_type": "ocr",
                    "config": config_dict
                }
            )
            
        elif config_type == "all":
            # 获取所有配置
            similarity_config = {
                "STORAGE_DIR": default_config.STORAGE_DIR,
                "MAX_CONCURRENT_TASKS": default_config.MAX_CONCURRENT_TASKS,
                "MAX_TASK_TIMEOUT": default_config.MAX_TASK_TIMEOUT,
                "MIN_TEXT_LENGTH": default_config.MIN_TEXT_LENGTH,
                "MIN_SEGMENT_LENGTH": default_config.MIN_SEGMENT_LENGTH,
                "MAX_SEGMENT_LENGTH": default_config.MAX_SEGMENT_LENGTH,
                "BATCH_SIZE": default_config.BATCH_SIZE,
                "ENABLE_OCR": default_config.ENABLE_OCR,
                "DETECTION_MODE": default_config.DETECTION_MODE,
                "PAGE_LEVEL_DETECTION": default_config.PAGE_LEVEL_DETECTION,
                "ENABLE_TABLE_DETECTION": default_config.ENABLE_TABLE_DETECTION,
                "TABLE_MIN_ROWS": default_config.TABLE_MIN_ROWS,
                "TENDER_SIMILARITY_THRESHOLD": default_config.TENDER_SIMILARITY_THRESHOLD,
                "BID_SIMILARITY_THRESHOLD": default_config.BID_SIMILARITY_THRESHOLD,
                "NEAR_IDENTICAL_THRESHOLD": default_config.NEAR_IDENTICAL_THRESHOLD,
                "HIGH_SIMILARITY_THRESHOLD": default_config.HIGH_SIMILARITY_THRESHOLD,
                "VERY_HIGH_SIMILARITY_THRESHOLD": default_config.VERY_HIGH_SIMILARITY_THRESHOLD,
                "SEMANTIC_EVADE_LOWER_THRESHOLD": default_config.SEMANTIC_EVADE_LOWER_THRESHOLD,
                "SEMANTIC_EVADE_UPPER_THRESHOLD": default_config.SEMANTIC_EVADE_UPPER_THRESHOLD,
                "SYNONYM_TOKEN_COUNT_DIFF_THRESHOLD": default_config.SYNONYM_TOKEN_COUNT_DIFF_THRESHOLD,
                "COMMON_TERM_COUNT_THRESHOLD": default_config.COMMON_TERM_COUNT_THRESHOLD,
                "ENABLE_GPU": default_config.ENABLE_GPU,
                "MEMORY_CLEANUP_INTERVAL": default_config.MEMORY_CLEANUP_INTERVAL,
                "SIMILARITY_TOP_K": default_config.SIMILARITY_TOP_K,
                "LOG_LEVEL": default_config.LOG_LEVEL,
            }
            
            ocr_config = {
                "OCR_THRESHOLD": default_paddle_ocr_config.OCR_THRESHOLD,
                "DPI": default_paddle_ocr_config.DPI,
                "MAX_IMAGE_WIDTH": default_paddle_ocr_config.MAX_IMAGE_WIDTH,
                "MAX_IMAGE_HEIGHT": default_paddle_ocr_config.MAX_IMAGE_HEIGHT,
                "MAX_PIXELS": default_paddle_ocr_config.MAX_PIXELS,
                "TEXT_DETECTION_MODEL_NAME": default_paddle_ocr_config.TEXT_DETECTION_MODEL_NAME,
                "TEXT_RECOGNITION_MODEL_NAME": default_paddle_ocr_config.TEXT_RECOGNITION_MODEL_NAME,
                "USE_DOC_ORIENTATION_CLASSIFY": default_paddle_ocr_config.USE_DOC_ORIENTATION_CLASSIFY,
                "USE_DOC_UNWARPING": default_paddle_ocr_config.USE_DOC_UNWARPING,
                "USE_TEXTLINE_ORIENTATION": default_paddle_ocr_config.USE_TEXTLINE_ORIENTATION,
                "LANG": default_paddle_ocr_config.LANG,
                "DEVICE": default_paddle_ocr_config.DEVICE,
                "MIN_CONFIDENCE": default_paddle_ocr_config.MIN_CONFIDENCE,
                "TEXT_REC_SCORE_THRESH": default_paddle_ocr_config.TEXT_REC_SCORE_THRESH,
            }
            
            return ConfigGetResponse(
                code=ErrorCode.SUCCESS,
                msg=ErrorMessage.SUCCESS,
                data={
                    "config_type": "all",
                    "config": {
                        "similarity": similarity_config,
                        "ocr": ocr_config
                    }
                }
            )
        else:
            raise HTTPException(
                status_code=ErrorCode.BAD_REQUEST,
                detail=get_error_response(
                    ErrorCode.BAD_REQUEST,
                    "无效的配置类型",
                    f"配置类型必须是 'similarity'、'ocr' 或 'all'，当前值: {config_type}"
                )
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取配置失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, "获取配置失败", str(e))
        )


@router.post(
    "/update",
    response_model=ConfigUpdateResponse,
    summary="更新配置",
    description="更新配置（JSON：config_type, config）"
)
async def update_config(request: ConfigUpdateRequest):
    """更新配置"""
    try:
        config_type = request.config_type.lower()
        updated_fields = []
        
        if config_type == "similarity":
            # 更新相似度配置
            valid_fields = [
                "MAX_CONCURRENT_TASKS", "MAX_TASK_TIMEOUT", "MIN_TEXT_LENGTH",
                "MIN_SEGMENT_LENGTH", "MAX_SEGMENT_LENGTH", "BATCH_SIZE",
                "ENABLE_OCR", "DETECTION_MODE", "PAGE_LEVEL_DETECTION",
                "ENABLE_TABLE_DETECTION", "TABLE_MIN_ROWS",
                "TENDER_SIMILARITY_THRESHOLD", "BID_SIMILARITY_THRESHOLD",
                "NEAR_IDENTICAL_THRESHOLD", "HIGH_SIMILARITY_THRESHOLD",
                "VERY_HIGH_SIMILARITY_THRESHOLD", "SEMANTIC_EVADE_LOWER_THRESHOLD",
                "SEMANTIC_EVADE_UPPER_THRESHOLD", "SYNONYM_TOKEN_COUNT_DIFF_THRESHOLD",
                "COMMON_TERM_COUNT_THRESHOLD", "ENABLE_GPU",
                "MEMORY_CLEANUP_INTERVAL", "SIMILARITY_TOP_K", "LOG_LEVEL"
            ]
            
            for key, value in request.config.items():
                if key not in valid_fields:
                    raise HTTPException(
                        status_code=ErrorCode.BAD_REQUEST,
                        detail=get_error_response(
                            ErrorCode.BAD_REQUEST,
                            f"无效的配置项: {key}",
                            f"相似度配置的有效字段: {', '.join(valid_fields)}"
                        )
                    )
                
                if hasattr(default_config, key):
                    setattr(default_config, key, value)
                    updated_fields.append(key)
                    logger.info(f"更新相似度配置: {key} = {value}")
            
        elif config_type == "ocr":
            # 更新OCR配置
            valid_fields = [
                "OCR_THRESHOLD", "DPI", "MAX_IMAGE_WIDTH", "MAX_IMAGE_HEIGHT",
                "MAX_PIXELS", "TEXT_DETECTION_MODEL_NAME", "TEXT_RECOGNITION_MODEL_NAME",
                "USE_DOC_ORIENTATION_CLASSIFY", "USE_DOC_UNWARPING",
                "USE_TEXTLINE_ORIENTATION", "LANG", "DEVICE",
                "MIN_CONFIDENCE", "TEXT_REC_SCORE_THRESH"
            ]
            
            for key, value in request.config.items():
                if key not in valid_fields:
                    raise HTTPException(
                        status_code=ErrorCode.BAD_REQUEST,
                        detail=get_error_response(
                            ErrorCode.BAD_REQUEST,
                            f"无效的配置项: {key}",
                            f"OCR配置的有效字段: {', '.join(valid_fields)}"
                        )
                    )
                
                if hasattr(default_paddle_ocr_config, key):
                    setattr(default_paddle_ocr_config, key, value)
                    updated_fields.append(key)
                    logger.info(f"更新OCR配置: {key} = {value}")
            
        else:
            raise HTTPException(
                status_code=ErrorCode.BAD_REQUEST,
                detail=get_error_response(
                    ErrorCode.BAD_REQUEST,
                    "无效的配置类型",
                    f"配置类型必须是 'similarity' 或 'ocr'，当前值: {config_type}"
                )
            )
        
        if not updated_fields:
            raise HTTPException(
                status_code=ErrorCode.BAD_REQUEST,
                detail=get_error_response(ErrorCode.BAD_REQUEST, "没有有效的配置项需要更新")
            )
        
        return ConfigUpdateResponse(
            code=ErrorCode.SUCCESS,
            msg="配置更新成功",
            data={
                "config_type": config_type,
                "updated_fields": updated_fields
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新配置失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, "更新配置失败", str(e))
        )

