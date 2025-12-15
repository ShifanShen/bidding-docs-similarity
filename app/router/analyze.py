"""
相似度分析API路由
提供文档相似度检测和规避行为分析功能
"""
import io
import json
import logging
import os
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Query, Form
from fastapi.responses import StreamingResponse, JSONResponse
import openpyxl
from app.service.similarity_service import SimilarityService
from app.models.schemas import (
    AnalyzeRequest, AnalyzeResponse, ResultResponse,
    TaskListResponse, CancelTaskRequest, CancelTaskResponse, CleanupTasksRequest,
    CleanupTasksResponse, AnalyzeExtractedRequest, FilterResultRequest
)
from app.models.errors import ErrorCode, ErrorMessage, get_error_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analyze", tags=["相似度分析"])
similarity_service = SimilarityService()


@router.post(
    "/start",
    response_model=AnalyzeResponse,
    summary="启动相似度分析",
    description="启动分析（JSON：tender_file_path, bid_file_paths）"
)
async def start_analysis(request: AnalyzeRequest):
    """启动相似度分析任务"""
    try:
        task_id = similarity_service.start_analysis(
            request.tender_file_path,
            request.bid_file_paths
        )
        return AnalyzeResponse(
            code=ErrorCode.SUCCESS,
            msg="分析任务已启动",
            data={"task_id": task_id}
        )
    except ValueError as e:
        logger.error(f"分析任务启动失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.BAD_REQUEST,
            detail=get_error_response(ErrorCode.BAD_REQUEST, ErrorMessage.TASK_START_FAILED, str(e))
        )
    except Exception as e:
        logger.error(f"分析任务启动失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, ErrorMessage.TASK_START_FAILED, str(e))
        )


@router.get(
    "/result",
    response_model=ResultResponse,
    summary="查询分析结果",
    description="按 task_id 查询结果"
)
async def get_result(
    task_id: str = Query(..., description="任务ID", example="abc123-def456-ghi789")
):
    """查询分析结果"""
    try:
        result = similarity_service.get_result(task_id)
        if result is None:
            raise HTTPException(
                status_code=ErrorCode.NOT_FOUND,
                detail=get_error_response(ErrorCode.NOT_FOUND, ErrorMessage.TASK_NOT_FOUND)
            )
        return ResultResponse(
            code=ErrorCode.SUCCESS,
            msg=ErrorMessage.SUCCESS,
            data=result
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, "查询失败", str(e))
        )


@router.get(
    "/export",
    summary="导出Excel结果",
    description="按 task_id 导出Excel"
)
def export_excel(
    task_id: str = Query(..., description="任务ID", example="abc123-def456-ghi789")
):
    """导出分析结果为Excel文件"""
    try:
        result = similarity_service.get_result(task_id)
        if not result or result.get('status') != 'done' or not result.get('result'):
            raise HTTPException(
                status_code=ErrorCode.NOT_FOUND,
                detail=get_error_response(ErrorCode.NOT_FOUND, ErrorMessage.RESULT_NOT_FOUND)
            )
        
        data = result['result']
        wb = openpyxl.Workbook()
        ws = wb.active
        assert ws is not None
        ws.title = "雷同片段"
        
        # 添加表头
        ws.append(["投标文件", "页码", "雷同文件", "雷同页码", "相似度", "雷同文本", "语法错误", "规避行为", "几乎相同页面"])

        # 添加相似片段数据
        for d in data.get('details', []):
            evade = []
            if d.get('order_changed'): evade.append('语序规避')
            if d.get('stopword_evade'): evade.append('无意义词插入规避')
            if d.get('synonym_evade'): evade.append('同义词替换规避')
            if d.get('semantic_evade'): evade.append('语义规避')
            
            is_near_identical = "是" if d.get('is_near_identical', False) else "否"
            
            ws.append([
                d.get('bid_file', ''), d.get('page', ''), d.get('similar_with', ''), d.get('similar_page', ''),
                d.get('similarity', ''), d.get('text', ''), '\n'.join(d.get('grammar_errors', [])), ','.join(evade), is_near_identical
            ])
        
        # 添加语法错误工作表
        ws2 = wb.create_sheet("相同语法错误")
        ws2.append(["语法错误", "片段内容", "出现位置"])
        for g in data.get('grammar_errors', []):
            locs = '\n'.join([f"{l['bid_file']} 第{l['page']}页" for l in g.get('locations', [])])
            ws2.append([g.get('error', ''), g.get('text', ''), locs])
        
        # 生成Excel文件
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': f'attachment; filename="similarity_result_{task_id}.xlsx"'}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Excel导出失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, "Excel导出失败", str(e))
        )


@router.get(
    "/tasks",
    response_model=TaskListResponse,
    summary="获取任务列表",
    description="查询任务列表"
)
async def get_all_tasks():
    """获取所有任务列表"""
    try:
        tasks = similarity_service.get_all_tasks()
        return TaskListResponse(
            code=ErrorCode.SUCCESS,
            msg=ErrorMessage.SUCCESS,
            data={"tasks": tasks}
        )
    except Exception as e:
        logger.error(f"查询失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, "查询失败", str(e))
        )


@router.post(
    "/cancel",
    response_model=CancelTaskResponse,
    summary="取消任务",
    description="取消任务（JSON：task_id）"
)
async def cancel_task(request: CancelTaskRequest):
    """取消指定任务"""
    try:
        success = similarity_service.cancel_task(request.task_id)
        if success:
            return CancelTaskResponse(
                code=ErrorCode.SUCCESS,
                msg="任务已取消"
            )
        else:
            raise HTTPException(
                status_code=ErrorCode.BAD_REQUEST,
                detail=get_error_response(ErrorCode.BAD_REQUEST, ErrorMessage.TASK_CANNOT_CANCEL)
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, "取消失败", str(e))
        )


@router.post(
    "/cleanup",
    response_model=CleanupTasksResponse,
    summary="清理过期任务",
    description="清理任务（JSON：max_age_hours）"
)
async def cleanup_tasks(request: CleanupTasksRequest):
    """清理过期任务"""
    try:
        similarity_service.cleanup_old_tasks(request.max_age_hours)
        return CleanupTasksResponse(
            code=ErrorCode.SUCCESS,
            msg=f"已清理{request.max_age_hours}小时前的任务"
        )
    except Exception as e:
        logger.error(f"清理失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, "清理失败", str(e))
        )


@router.post(
    "/from-extracted",
    response_model=AnalyzeResponse,
    summary="基于已提取文本分析",
    description="使用已提取文本启动分析"
)
async def analyze_extracted_texts(request: AnalyzeExtractedRequest):
    """基于已提取的文本数据进行相似度分析"""
    try:
        # 转换为字典格式
        extracted_data = {
            "tender_texts": [item.dict() for item in request.tender_texts],
            "bid_files": [{"file_name": bf.file_name, "texts": [t.dict() for t in bf.texts]} for bf in request.bid_files]
        }
        
        # 启动分析任务
        task_id = similarity_service.start_analysis_from_extracted_texts(extracted_data)
        
        return AnalyzeResponse(
            code=ErrorCode.SUCCESS,
            msg="基于提取文本的分析任务已启动",
            data={"task_id": task_id}
        )
    except ValueError as e:
        logger.error(f"数据格式错误: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.BAD_REQUEST,
            detail=get_error_response(ErrorCode.BAD_REQUEST, ErrorMessage.INVALID_DATA_FORMAT, str(e))
        )
    except Exception as e:
        logger.error(f"分析任务启动失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, ErrorMessage.TASK_START_FAILED, str(e))
        )


@router.post(
    "/filter",
    summary="筛选分析结果",
    description="筛选分析结果（表单：result_file, min_similarity）"
)
async def filter_result(
    result_file: UploadFile = File(..., description="分析结果JSON文件"),
    min_similarity: float = Form(0.0, description="最小相似度阈值", ge=0.0, le=1.0)
):
    """上传分析结果JSON并按相似度阈值筛选"""
    try:
        raw = await result_file.read()
        data = json.loads(raw.decode("utf-8"))

        # 兼容多种JSON格式
        result_obj = data
        if isinstance(data, dict):
            if "result" in data and isinstance(data.get("result"), dict):
                result_obj = data["result"]
            elif "status" in data and "result" in data.get("result", {}):
                result_obj = data.get("result", {}).get("result", data)

        filtered = dict(result_obj)
        details = list(result_obj.get("details", []))
        
        logger.info(f"接收到的JSON: details数量={len(details)}, 原始summary={result_obj.get('summary', 'N/A')}")

        new_details = [
            d for d in details
            if float(d.get('similarity', 0.0)) >= float(min_similarity)
        ]

        filtered["details"] = new_details
        
        # 重新计算统计信息
        total_similarity_count = len(new_details)
        grammar_errors_count = len(filtered.get("grammar_errors", []))
        new_summary = f'分析完成，发现{total_similarity_count}处高相似度片段，{grammar_errors_count}组相同语法错误。'
        filtered["summary"] = new_summary
        
        # 重新计算平均和最大相似度
        if new_details:
            import numpy as np
            similarities = [float(d.get('similarity', 0.0)) for d in new_details]
            filtered["avg_similarity_score"] = float(f'{np.mean(similarities):.4f}')
            filtered["max_similarity_score"] = float(f'{max(similarities):.4f}')
        else:
            filtered["avg_similarity_score"] = 0.0
            filtered["max_similarity_score"] = 0.0
        
        filtered["total_similarity_count"] = total_similarity_count
        
        logger.info(f"筛选完成: 原始{len(details)}条 -> 筛选后{total_similarity_count}条, 阈值={min_similarity}")

        return JSONResponse(content={
            "code": ErrorCode.SUCCESS,
            "msg": ErrorMessage.SUCCESS,
            "data": filtered
        })
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.BAD_REQUEST,
            detail=get_error_response(ErrorCode.BAD_REQUEST, "JSON格式错误", str(e))
        )
    except Exception as e:
        logger.error(f"筛选失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, "筛选失败", str(e))
        )

