from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import json
import tempfile
import os
from app.service.text_utils import text_result_highlight_local
from app.service.service_manager import get_oss_service

router = APIRouter(prefix="/api/highlight", tags=["高亮标记"])

@router.post("/result-files")
async def highlight_from_files(
    pdf_file: UploadFile = File(..., description="PDF文件"),
    json_file: UploadFile = File(..., description="JSON分析结果文件")
):
    """
    上传PDF和JSON文件，返回高亮后的PDF（优先本地处理；MinIO可用则同时上传）
    """
    try:
        # 验证文件类型
        if not pdf_file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="PDF文件格式错误")
        
        # 读取并解析JSON
        json_content = await json_file.read()
        try:
            json_data = json.loads(json_content.decode('utf-8'))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JSON文件格式错误: {str(e)}")
        
        # 处理PDF文件
        content = await pdf_file.read()
        pdf_filename = pdf_file.filename
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(content)
        
        try:
            # 计算details数量
            all_details = json_data["result"]["result"].get("details", [])
            pdf_details = [d for d in all_details if d.get("bid_file") == pdf_filename]
            highlight_count = len(pdf_details)
            
            # 本地高亮（不依赖MinIO）
            try:
                out_path = text_result_highlight_local(tmp_path, json_data, pdf_name=pdf_filename)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"高亮处理失败: {str(e)}")

            # 若 MinIO 可用，额外上传一份（不影响下载返回）
            original_url = None
            highlighted_url = None
            try:
                oss = get_oss_service()
                if oss.is_available():
                    original_url = oss.upload_file(tmp_path, pdf_filename)
                    highlighted_url = oss.upload_file(out_path, pdf_filename)
            except Exception:
                # 不阻塞主流程
                pass

            # 返回高亮PDF文件流
            f = open(out_path, "rb")
            headers = {
                "Content-Disposition": f"attachment; filename*=UTF-8''{pdf_filename}",
            }
            if original_url:
                headers["X-Original-MinIO-URL"] = original_url
            if highlighted_url:
                headers["X-Highlighted-MinIO-URL"] = highlighted_url
            headers["X-Highlight-Count"] = str(highlight_count)

            return StreamingResponse(
                f,
                media_type="application/pdf",
                headers=headers,
            )
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")