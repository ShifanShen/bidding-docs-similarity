# app/router/entity.py
from typing import Any, List, Optional
import json
import logging

from fastapi import APIRouter, Body, HTTPException, Query, UploadFile, File

from app.service.entity_rec_service import default_entity_rec_service

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
):
    """
    输入：extracted_data JSON（或 JSON 字符串）
    输出：同结构 JSON，在每个 text 节点补 entities: [...]
    """
    try:
        return default_entity_rec_service.enrich_extracted_data(
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

        return default_entity_rec_service.enrich_extracted_data(
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
