# app/router/entity.py
from typing import Any, List, Optional
import json
import logging

from fastapi import APIRouter, Body, HTTPException, Query, UploadFile, File

# 导入两个实体识别服务
from app.service.entity_rec_service import default_entity_rec_service as hanlp_service
from app.service.entity_regex_rec_service import default_entity_rec_service as regex_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/entity", tags=["实体识别"])

# 你可以按部署环境调整上传限制（例如 20MB）
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@router.post("/enrich")
def enrich(
        payload: Any = Body(...),
        keep_offsets: bool = Query(False, description="是否保留 start/end（调试用）"),
        override_existing: bool = Query(True, description="已有 entities 是否覆盖重算"),
        entity_keys: Optional[List[str]] = Query(None, description="指定识别的实体类型，可重复传参"),
        service_type: str = Query("hanlp", description="实体识别服务类型: hanlp 或 regex")
):
    """
    输入：extracted_data JSON（或 JSON 字符串）
    输出：同结构 JSON，在每个 text 节点补 entities: [...]
    """
    try:
        if service_type == "regex":
            # 使用正则实体识别服务
            # 检查regex服务是否有enrich_extracted_data方法
            if hasattr(regex_service, "enrich_extracted_data"):
                return regex_service.enrich_extracted_data(
                    payload=payload,
                    entity_keys=entity_keys,
                    keep_offsets=keep_offsets,
                    override_existing=override_existing,
                    copy_input=True,
                )
            else:
                # 如果regex服务没有enrich_extracted_data方法，需要自己实现
                if isinstance(payload, str):
                    payload_obj = json.loads(payload)
                else:
                    payload_obj = payload

                if not isinstance(payload_obj, (dict, list)):
                    raise ValueError("payload 必须是 JSON 对象或 JSON 数组")

                # 处理单个文档
                if isinstance(payload_obj, dict):
                    if "text" in payload_obj:
                        entities = regex_service.extract_entities(payload_obj["text"])
                        payload_obj["entities"] = entities
                        return payload_obj
                    elif "tender_texts" in payload_obj and "bid_files" in payload_obj:
                        # 处理相似度分析格式
                        # 处理招标文件文本
                        if "tender_texts" in payload_obj:
                            for text_item in payload_obj["tender_texts"]:
                                if "text" in text_item:
                                    entities = regex_service.extract_entities(text_item["text"])
                                    text_item["entities"] = entities

                        # 处理投标文件文本
                        if "bid_files" in payload_obj:
                            for bid_file in payload_obj["bid_files"]:
                                if "texts" in bid_file:
                                    for text_item in bid_file["texts"]:
                                        if "text" in text_item:
                                            entities = regex_service.extract_entities(text_item["text"])
                                            text_item["entities"] = entities

                        return payload_obj
                    else:
                        raise ValueError("payload 格式不支持")
                elif isinstance(payload_obj, list):
                    # 处理文档列表
                    result = []
                    for doc in payload_obj:
                        if isinstance(doc, dict) and "text" in doc:
                            entities = regex_service.extract_entities(doc["text"])
                            doc["entities"] = entities
                            result.append(doc)
                    return result
                else:
                    raise ValueError("payload 格式不支持")
        else:
            # 默认使用HanLP实体识别服务
            return hanlp_service.enrich_extracted_data(
                payload=payload,
                entity_keys=entity_keys,
                keep_offsets=keep_offsets,
                override_existing=override_existing,
                copy_input=True,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("实体增强失败: %s", e)
        raise HTTPException(status_code=500, detail="实体增强失败")


@router.post("/enrich-file")
async def enrich_file(
        file: UploadFile = File(..., description="上传 JSON 文件（extracted_data）"),
        keep_offsets: bool = Query(False, description="是否保留 start/end（调试用）"),
        override_existing: bool = Query(True, description="已有 entities 是否覆盖重算"),
        entity_keys: Optional[List[str]] = Query(None, description="指定识别的实体类型，可重复传参"),
        service_type: str = Query("hanlp", description="实体识别服务类型: hanlp 或 regex")
):
    """
    输入：multipart/form-data 上传 JSON 文件
    输出：同结构 JSON，在每个 text 节点补 entities: [...]
    """
    try:
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"文件过大，限制 {MAX_UPLOAD_BYTES} bytes")

        # utf-8-sig 兼容 BOM
        text = content.decode("utf-8-sig")
        payload = json.loads(text)

        if service_type == "regex":
            # 使用正则实体识别服务
            # 检查regex服务是否有enrich_extracted_data方法
            if hasattr(regex_service, "enrich_extracted_data"):
                return regex_service.enrich_extracted_data(
                    payload=payload,
                    entity_keys=entity_keys,
                    keep_offsets=keep_offsets,
                    override_existing=override_existing,
                    copy_input=True,
                )
            else:
                # 如果regex服务没有enrich_extracted_data方法，需要自己实现
                if not isinstance(payload, (dict, list)):
                    raise ValueError("payload 必须是 JSON 对象或 JSON 数组")

                # 处理单个文档
                if isinstance(payload, dict):
                    if "text" in payload:
                        entities = regex_service.extract_entities(payload["text"])
                        payload["entities"] = entities
                        return payload
                    elif "tender_texts" in payload and "bid_files" in payload:
                        # 处理相似度分析格式
                        # 处理招标文件文本
                        if "tender_texts" in payload:
                            for text_item in payload["tender_texts"]:
                                if "text" in text_item:
                                    entities = regex_service.extract_entities(text_item["text"])
                                    text_item["entities"] = entities

                        # 处理投标文件文本
                        if "bid_files" in payload:
                            for bid_file in payload["bid_files"]:
                                if "texts" in bid_file:
                                    for text_item in bid_file["texts"]:
                                        if "text" in text_item:
                                            entities = regex_service.extract_entities(text_item["text"])
                                            text_item["entities"] = entities

                        return payload
                    else:
                        raise ValueError("payload 格式不支持")
                elif isinstance(payload, list):
                    # 处理文档列表
                    result = []
                    for doc in payload:
                        if isinstance(doc, dict) and "text" in doc:
                            entities = regex_service.extract_entities(doc["text"])
                            doc["entities"] = entities
                            result.append(doc)
                    return result
                else:
                    raise ValueError("payload 格式不支持")
        else:
            # 默认使用HanLP实体识别服务
            return hanlp_service.enrich_extracted_data(
                payload=payload,
                entity_keys=entity_keys,
                keep_offsets=keep_offsets,
                override_existing=override_existing,
                copy_input=True,
            )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="上传文件不是有效 JSON")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("实体增强失败: %s", e)
        raise HTTPException(status_code=500, detail="实体增强失败")