"""
实体识别API路由
提供文本实体识别功能（支持HanLP和正则两种方式）
"""
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Body, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.service.service_manager import get_entity_rec_service, get_entity_regex_service
from app.models.errors import ErrorCode, ErrorMessage, get_error_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/entity", tags=["实体识别"])


# ==================== 请求/响应模型 ====================

class EntityRecognitionRequest(BaseModel):
    """实体识别请求"""
    text: str = Field(..., description="要识别的文本内容")
    entity_keys: Optional[List[str]] = Field(None, description="要识别的实体类型列表，如['公司', '人名', '联系方式', '身份证']")
    keep_offsets: bool = Field(False, description="是否返回实体在文本中的位置信息")
    use_hanlp: bool = Field(True, description="是否使用HanLP（True使用HanLP+正则，False仅使用正则）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "联系人：张三，电话：13800138000，公司：北京科技有限公司",
                "entity_keys": ["公司", "人名", "联系方式"],
                "keep_offsets": False,
                "use_hanlp": True
            }
        }


class EntityItem(BaseModel):
    """实体项"""
    entity: str = Field(..., description="实体类型，如'公司'、'人名'等")
    text_content: str = Field(..., description="实体文本内容")
    start: Optional[int] = Field(None, description="实体在文本中的起始位置（如果keep_offsets=True）")
    end: Optional[int] = Field(None, description="实体在文本中的结束位置（如果keep_offsets=True）")


class EntityRecognitionResponse(BaseModel):
    """实体识别响应"""
    code: int = Field(200, description="响应状态码，200表示成功")
    msg: str = Field(..., description="响应消息")
    data: Dict[str, Any] = Field(..., description="响应数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "识别成功",
                "data": {
                    "entities": [
                        {"entity": "人名", "text_content": "张三"},
                        {"entity": "联系方式", "text_content": "13800138000"},
                        {"entity": "公司", "text_content": "北京科技有限公司"}
                    ],
                    "method": "hanlp+regex",
                    "total_count": 3
                }
            }
        }


class EntityRecognitionFromJSONRequest(BaseModel):
    """从JSON数据中识别实体请求"""
    json_data: Dict[str, Any] = Field(..., description="包含text字段的JSON数据")
    entity_keys: Optional[List[str]] = Field(None, description="要识别的实体类型列表")
    keep_offsets: bool = Field(False, description="是否返回实体在文本中的位置信息")
    use_hanlp: bool = Field(True, description="是否使用HanLP")


# ==================== API端点 ====================

@router.post(
    "/recognize",
    response_model=EntityRecognitionResponse,
    summary="识别文本中的实体",
    description="从文本中识别实体（公司、人名、联系方式、身份证等），支持HanLP和正则两种方式"
)
async def recognize_entities(request: EntityRecognitionRequest = Body(...)):
    """识别文本中的实体"""
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(
                status_code=ErrorCode.BAD_REQUEST,
                detail=get_error_response(
                    ErrorCode.BAD_REQUEST,
                    "文本内容不能为空"
                )
            )
        
        # 选择使用的服务
        if request.use_hanlp:
            # 使用HanLP + 正则
            try:
                # 检查HanLP是否可用
                entity_service = get_entity_rec_service()
                if entity_service.is_hanlp_available():
                    entities = entity_service.extract_entities_from_text(
                        raw_text=text,
                        entity_keys=request.entity_keys,
                        keep_offsets=request.keep_offsets
                    )
                    method = "hanlp+regex"
                else:
                    # HanLP不可用，使用正则
                    logger.warning("HanLP不可用，使用纯正则")
                    entities = default_regex_service.extract_entities(text)
                    method = "regex_fallback"
            except Exception as e:
                logger.warning(f"HanLP识别失败，回退到纯正则: {str(e)}")
                # 回退到纯正则
                entities = get_entity_regex_service().extract_entities(text)
                method = "regex_fallback"
        else:
            # 仅使用正则
            entities = get_entity_regex_service().extract_entities(text)
            method = "regex_only"
        
        # 格式化响应数据
        formatted_entities = []
        for entity in entities:
            item = {
                "entity": entity.get("entity", ""),
                "text_content": entity.get("text_content", "")
            }
            if request.keep_offsets:
                item["start"] = entity.get("start_pos") or entity.get("start", -1)
                item["end"] = entity.get("end_pos") or entity.get("end", -1)
            formatted_entities.append(item)
        
        return EntityRecognitionResponse(
            code=ErrorCode.SUCCESS,
            msg="识别成功",
            data={
                "entities": formatted_entities,
                "method": method,
                "total_count": len(formatted_entities)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"实体识别失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, "实体识别失败", str(e))
        )


@router.post(
    "/recognize-from-json",
    response_model=EntityRecognitionResponse,
    summary="从JSON数据中识别实体",
    description="从包含text字段的JSON数据中识别实体"
)
async def recognize_entities_from_json(request: EntityRecognitionFromJSONRequest = Body(...)):
    """从JSON数据中识别实体"""
    try:
        json_data = request.json_data
        
        # 选择使用的服务
        if request.use_hanlp:
            try:
                # 检查HanLP是否可用
                entity_service = get_entity_rec_service()
                if entity_service.is_hanlp_available():
                    # 使用HanLP服务
                    enriched_data = entity_service.enrich_extracted_data(
                        payload=json_data,
                        entity_keys=request.entity_keys,
                        keep_offsets=request.keep_offsets
                    )
                    method = "hanlp+regex"
                else:
                    # HanLP不可用，使用正则
                    logger.warning("HanLP不可用，使用纯正则")
                    enriched_data = _enrich_with_regex(json_data, request.entity_keys, request.keep_offsets)
                    method = "regex_fallback"
            except Exception as e:
                logger.warning(f"HanLP识别失败，回退到纯正则: {str(e)}")
                # 回退到纯正则
                enriched_data = _enrich_with_regex(json_data, request.entity_keys, request.keep_offsets)
                method = "regex_fallback"
        else:
            # 仅使用正则
            enriched_data = _enrich_with_regex(json_data, request.entity_keys, request.keep_offsets)
            method = "regex_only"
        
        return EntityRecognitionResponse(
            code=ErrorCode.SUCCESS,
            msg="识别成功",
            data={
                "enriched_data": enriched_data,
                "method": method
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"从JSON识别实体失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, "从JSON识别实体失败", str(e))
        )


@router.get(
    "/health",
    summary="检查实体识别服务健康状态",
    description="检查HanLP和正则服务的可用性"
)
async def check_entity_service_health():
    """检查实体识别服务健康状态"""
    try:
        health_status = {
            "hanlp": {
                "available": False,
                "error": None
            },
            "regex": {
                "available": True,
                "error": None
            }
        }
        
        # 检查HanLP服务
        try:
            # 检查HanLP是否可用（不强制加载）
            entity_service = get_entity_rec_service()
            health_status["hanlp"]["available"] = entity_service.is_hanlp_available()
            if not health_status["hanlp"]["available"]:
                health_status["hanlp"]["error"] = "HanLP模型未加载或加载失败"
        except Exception as e:
            health_status["hanlp"]["error"] = str(e)
            logger.warning(f"HanLP服务检查失败: {str(e)}")
        
        # 检查正则服务
        try:
            test_text = "测试文本"
            _ = get_entity_regex_service().extract_entities(test_text)
            health_status["regex"]["available"] = True
        except Exception as e:
            health_status["regex"]["available"] = False
            health_status["regex"]["error"] = str(e)
            logger.error(f"正则服务不可用: {str(e)}")
        
        # 判断整体状态
        all_available = health_status["hanlp"]["available"] and health_status["regex"]["available"]
        any_available = health_status["hanlp"]["available"] or health_status["regex"]["available"]
        
        if all_available:
            status_msg = "所有服务正常"
            status_code = ErrorCode.SUCCESS
        elif any_available:
            status_msg = "部分服务可用（建议检查HanLP配置）"
            status_code = ErrorCode.SUCCESS
        else:
            status_msg = "服务不可用"
            status_code = ErrorCode.SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=200,
            content={
                "code": status_code,
                "msg": status_msg,
                "data": health_status
            }
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, "健康检查失败", str(e))
        )


# ==================== 辅助函数 ====================

def _enrich_with_regex(json_data: Dict[str, Any], entity_keys: Optional[List[str]], keep_offsets: bool) -> Dict[str, Any]:
    """使用正则服务丰富JSON数据"""
    import copy
    
    enriched = copy.deepcopy(json_data)
    regex_service = get_entity_regex_service()
    
    def extract_from_text(text: str) -> List[Dict[str, Any]]:
        """从文本中提取实体"""
        entities = regex_service.extract_entities(text)
        result = []
        for entity in entities:
            item = {
                "entity": entity.get("entity", ""),
                "text_content": entity.get("text_content", "")
            }
            if keep_offsets:
                item["start"] = entity.get("start_pos", -1)
                item["end"] = entity.get("end_pos", -1)
            result.append(item)
        return result
    
    # 递归处理JSON中的text字段
    if isinstance(enriched, dict):
        if "text" in enriched and isinstance(enriched["text"], str):
            enriched["entities"] = extract_from_text(enriched["text"])
        for value in enriched.values():
            if isinstance(value, (dict, list)):
                _enrich_with_regex(value, entity_keys, keep_offsets)
    elif isinstance(enriched, list):
        for item in enriched:
            if isinstance(item, (dict, list)):
                _enrich_with_regex(item, entity_keys, keep_offsets)
    
    return enriched
