from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional
import os
from app.service.similarity_service import SimilarityService
import io
import openpyxl  # type: ignore

router = APIRouter(prefix="/api/similarity", tags=["similarity"])

# 初始化服务实例
similarity_service = SimilarityService()

# 上传文件接口（支持多文件）
@router.post("/upload")
async def upload_files(
    tender_file: UploadFile = File(..., description="招标文件"),
    bid_files: List[UploadFile] = File(..., description="投标文件（可多份）")
):
    try:
        tender_path = similarity_service.save_file(await tender_file.read(), tender_file.filename or "tender_file")
        bid_paths = []
        for f in bid_files:
            path = similarity_service.save_file(await f.read(), f.filename or "bid_file")
            bid_paths.append(path)
        return {
            "msg": "文件上传成功",
            "tender_file_path": tender_path,
            "bid_file_paths": bid_paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

# 发起分析接口
@router.post("/analyze")
async def analyze_files(
    background_tasks: BackgroundTasks,
    tender_file_path: str = Form(...),
    bid_file_paths: List[str] = Form(...)
):
    try:
        task_id = similarity_service.start_analysis(tender_file_path, bid_file_paths)
        return {"msg": "分析任务已启动", "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析任务启动失败: {str(e)}")

# 查询分析结果接口
@router.get("/result")
async def get_result(task_id: str):
    try:
        result = similarity_service.get_result(task_id)
        return {"msg": "查询成功", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

# 导出Excel接口
@router.get("/export_excel")
def export_excel(task_id: str):
    result = similarity_service.get_result(task_id)
    if not result or result.get('status') != 'done' or not result.get('result'):
        raise HTTPException(status_code=404, detail="分析结果未找到或未完成")
    data = result['result']
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "雷同片段"
    ws.append(["投标文件", "页码", "雷同文件", "雷同页码", "相似度", "雷同文本", "语法错误", "规避行为"])
    for d in data.get('details', []):
        evade = []
        if d.get('order_changed'): evade.append('语序规避')
        if d.get('stopword_evade'): evade.append('无意义词插入规避')
        ws.append([
            d.get('bid_file', ''), d.get('page', ''), d.get('similar_with', ''), d.get('similar_page', ''),
            d.get('similarity', ''), d.get('text', ''), '\n'.join(d.get('grammar_errors', [])), ','.join(evade)
        ])
    ws2 = wb.create_sheet("相同语法错误")
    ws2.append(["语法错误", "片段内容", "出现位置"])
    for g in data.get('grammar_errors', []):
        locs = '\n'.join([f"{l['bid_file']} 第{l['page']}页" for l in g.get('locations', [])])
        ws2.append([g.get('error', ''), g.get('text', ''), locs])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return StreamingResponse(buf, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers={
        'Content-Disposition': f'attachment; filename="similarity_result_{task_id}.xlsx"'
    })

# 任务管理接口
@router.get("/tasks")
async def get_all_tasks():
    """获取所有任务列表"""
    try:
        tasks = similarity_service.get_all_tasks()
        return {"msg": "查询成功", "tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.post("/cancel_task")
async def cancel_task(task_id: str = Form(...)):
    """取消指定任务"""
    try:
        success = similarity_service.cancel_task(task_id)
        if success:
            return {"msg": "任务已取消"}
        else:
            raise HTTPException(status_code=400, detail="任务无法取消或不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取消失败: {str(e)}")

@router.post("/cleanup_tasks")
async def cleanup_tasks(max_age_hours: int = Form(24)):
    """清理过期任务"""
    try:
        similarity_service.cleanup_old_tasks(max_age_hours)
        return {"msg": f"已清理{max_age_hours}小时前的任务"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")
