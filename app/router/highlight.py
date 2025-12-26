# app/router/highlight.py
from fastapi import APIRouter, HTTPException, UploadFile, File
import json
import tempfile
import os
from app.service.text_utils import text_result_highlight
from app.service.oss_service import oss_service

router = APIRouter(prefix="/api/highlight", tags=["高亮标记"])

@router.post("/result-files")
async def highlight_from_files(
    pdf_file: UploadFile = File(..., description="PDF文件"),
    json_file: UploadFile = File(..., description="JSON分析结果文件")
):
    """
    上传PDF和JSON文件，返回高亮后的PDF
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
            # 上传PDF到MinIO
            minio_url = oss_service.upload_file(tmp_path, pdf_filename)
            if not minio_url:
                raise HTTPException(status_code=500, detail="PDF上传到MinIO失败")
            
            # 计算details数量
            all_details = json_data["result"]["result"].get("details", [])
            pdf_details = [d for d in all_details if d.get("bid_file") == pdf_filename]
            highlight_count = len(pdf_details)
            
            # 调用高亮函数
            try:
                highlighted_url = text_result_highlight(minio_url, json_data)
                # 如果有匹配项
                if highlight_count > 0:
                    return {
                        "success": True,
                        "message": "PDF高亮处理成功",
                        "data": {
                            "original_pdf": pdf_filename,
                            "original_pdf_url": minio_url,
                            "highlighted_pdf": pdf_filename,
                            "highlighted_pdf_url": highlighted_url,
                            "highlight_count": highlight_count
                        }
                    }
                else:
                    # 没有匹配项
                    return {
                        "success": True,
                        "message": f"PDF '{pdf_filename}' 中没有找到匹配的相似文本",
                        "data": {
                            "original_pdf": pdf_filename,
                            "original_pdf_url": minio_url,
                            "highlighted_pdf": pdf_filename,
                            "highlighted_pdf_url": minio_url,
                            "highlight_count": 0
                        }
                    }
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"高亮处理失败: {str(e)}")
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")