"""
相似度分析路由（重构版）
"""
import os
import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from app.core.exceptions import BiddingDocsException
from app.core.config_manager import config_manager
from app.services.similarity_service_v2 import SimilarityAnalysisService

router = APIRouter(prefix="/api/similarity/v2", tags=["similarity-v2"])
logger = logging.getLogger(__name__)

# 全局服务实例
similarity_service = SimilarityAnalysisService()


@router.on_event("startup")
async def startup_event():
    """启动事件"""
    if not similarity_service.initialize():
        logger.error("相似度分析服务初始化失败")
        raise HTTPException(status_code=500, detail="服务初始化失败")


@router.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    similarity_service.cleanup()


@router.post("/analyze")
async def analyze_similarity(
    background_tasks: BackgroundTasks,
    tender_file: UploadFile = File(..., description="招标文件"),
    bid_files: List[UploadFile] = File(..., description="投标文件列表")
):
    """
    分析投标文件相似度
    """
    try:
        # 验证文件
        if not tender_file.filename:
            raise HTTPException(status_code=400, detail="招标文件不能为空")
        
        if not bid_files:
            raise HTTPException(status_code=400, detail="投标文件不能为空")
        
        if len(bid_files) < 2:
            raise HTTPException(status_code=400, detail="至少需要2个投标文件")
        
        # 保存文件
        config = config_manager.config
        tender_file_path = await _save_uploaded_file(tender_file, config.storage_dir)
        bid_file_paths = []
        
        for bid_file in bid_files:
            if bid_file.filename:
                file_path = await _save_uploaded_file(bid_file, config.storage_dir)
                bid_file_paths.append(file_path)
        
        # 启动分析任务
        task_id = similarity_service.analyze_similarity(tender_file_path, bid_file_paths)
        
        return {
            "task_id": task_id,
            "message": "分析任务已启动",
            "tender_file": tender_file.filename,
            "bid_files": [f.filename for f in bid_files if f.filename]
        }
        
    except BiddingDocsException as e:
        logger.error(f"分析任务创建失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"未知错误: {str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """
    获取任务状态
    """
    try:
        task_info = similarity_service.get_task_status(task_id)
        if not task_info:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return {
            "task_id": task_id,
            "status": task_info.status.value,
            "progress": task_info.progress,
            "result": task_info.result,
            "error": task_info.error,
            "created_at": task_info.created_at,
            "started_at": task_info.started_at,
            "completed_at": task_info.completed_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.get("/tasks")
async def get_all_tasks():
    """
    获取所有任务列表
    """
    try:
        tasks = similarity_service.get_all_tasks()
        
        task_list = []
        for task_id, task_info in tasks.items():
            task_list.append({
                "task_id": task_id,
                "status": task_info.status.value,
                "progress": task_info.progress,
                "created_at": task_info.created_at,
                "started_at": task_info.started_at,
                "completed_at": task_info.completed_at
            })
        
        return {"tasks": task_list}
        
    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.post("/task/{task_id}/cancel")
async def cancel_task(task_id: str):
    """
    取消任务
    """
    try:
        success = similarity_service.cancel_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="任务不存在或无法取消")
        
        return {"message": "任务已取消"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.post("/tasks/cleanup")
async def cleanup_tasks(max_age_hours: int = 24):
    """
    清理过期任务
    """
    try:
        cleaned_count = similarity_service.cleanup_tasks(max_age_hours)
        return {"message": f"已清理 {cleaned_count} 个过期任务"}
        
    except Exception as e:
        logger.error(f"清理任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.get("/result/{task_id}")
async def get_analysis_result(task_id: str):
    """
    获取分析结果
    """
    try:
        task_info = similarity_service.get_task_status(task_id)
        if not task_info:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if task_info.status != task_info.status.COMPLETED:
            raise HTTPException(status_code=400, detail="任务未完成")
        
        if not task_info.result:
            raise HTTPException(status_code=404, detail="结果不存在")
        
        return task_info.result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取分析结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.get("/export/{task_id}")
async def export_result(task_id: str, format: str = "json"):
    """
    导出分析结果
    """
    try:
        task_info = similarity_service.get_task_status(task_id)
        if not task_info:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if task_info.status != task_info.status.COMPLETED:
            raise HTTPException(status_code=400, detail="任务未完成")
        
        config = config_manager.config
        result_file = os.path.join(config.storage_dir, f"similarity_result_{task_id}.json")
        
        if not os.path.exists(result_file):
            raise HTTPException(status_code=404, detail="结果文件不存在")
        
        if format.lower() == "json":
            return FileResponse(
                result_file,
                media_type="application/json",
                filename=f"similarity_result_{task_id}.json"
            )
        else:
            raise HTTPException(status_code=400, detail="不支持的导出格式")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


async def _save_uploaded_file(file: UploadFile, storage_dir: str) -> str:
    """保存上传的文件"""
    try:
        # 确保存储目录存在
        os.makedirs(storage_dir, exist_ok=True)
        
        # 生成文件路径
        file_path = os.path.join(storage_dir, file.filename)
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"文件已保存: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"文件保存失败: {str(e)}")
        raise BiddingDocsException(f"文件保存失败: {str(e)}")
