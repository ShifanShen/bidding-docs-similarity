"""
文件上传API路由
提供统一的文件上传功能
"""
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.service.similarity_service import SimilarityService
from app.models.schemas import FileUploadResponse
from app.models.errors import ErrorCode, ErrorMessage, get_error_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/upload", tags=["文件上传"])
similarity_service = SimilarityService()


@router.post(
    "/files",
    response_model=FileUploadResponse,
    summary="上传文件",
    description="上传文件（multipart/form-data，字段：files，多文件）"
)
async def upload_files(
    files: List[UploadFile] = File(..., description="文件列表（可多份）")
):
    """上传文件"""
    try:
        if not files:
            raise HTTPException(
                status_code=ErrorCode.BAD_REQUEST,
                detail=get_error_response(ErrorCode.BAD_REQUEST, "文件列表不能为空")
            )
        
        file_paths = []
        for file in files:
            try:
                file_path = similarity_service.save_file(
                    await file.read(),
                    file.filename or "uploaded_file"
                )
                file_paths.append(file_path)
            except Exception as e:
                logger.error(f"文件 {file.filename} 上传失败: {str(e)}")
                raise HTTPException(
                    status_code=ErrorCode.INTERNAL_ERROR,
                    detail=get_error_response(ErrorCode.INTERNAL_ERROR, f"文件 {file.filename} 上传失败", str(e))
                )
        
        return FileUploadResponse(
            code=ErrorCode.SUCCESS,
            msg="文件上传成功",
            data={
                "file_paths": file_paths,
                "file_count": len(file_paths)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(
            status_code=ErrorCode.INTERNAL_ERROR,
            detail=get_error_response(ErrorCode.INTERNAL_ERROR, ErrorMessage.FILE_UPLOAD_FAILED, str(e))
        )

