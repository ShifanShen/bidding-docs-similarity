# 文档上传和相似度检测路由
from fastapi import APIRouter, File, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app import app, Config
import os
import shutil
from service.text_service import TextProcessingService
from service.text_service import SimilarityService
router = APIRouter()

@router.post("/upload-and-compare")
async def upload_and_compare(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # 实现原main.py中的/upload-and-compare路由逻辑
    # 临时文件保存
    file1_path = os.path.join(Config.TEMP_DIR, file1.filename)
    file2_path = os.path.join(Config.TEMP_DIR, file2.filename)
    
    with open(file1_path, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
    with open(file2_path, "wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)
    
    # 提取文本内容
    text1 = await TextProcessingService.extract_text_from_file(file1_path)
    text2 = await TextProcessingService.extract_text_from_file(file2_path)
    # 预处理文本
    processed_text1 = TextProcessingService.preprocess_text(text1)
    processed_text2 = TextProcessingService.preprocess_text(text2)
    
    # 计算相似度
    # 移除await关键字，因为calculate_similarity是同步方法
    similarity_result = SimilarityService.calculate_similarity(processed_text1, processed_text2)
    
    return {
        "similarity_score": similarity_result["similarity_score"],
        "threshold_level": similarity_result["threshold_level"],
        "analysis": similarity_result["analysis"]
    }
      