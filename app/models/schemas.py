"""
API请求和响应数据模型
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ==================== 相似度分析相关模型 ====================

class FileUploadResponse(BaseModel):
    """文件上传响应"""
    code: int = Field(200, description="响应状态码，200表示成功")
    msg: str = Field(..., description="响应消息")
    data: Dict[str, Any] = Field(..., description="响应数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "文件上传成功",
                "data": {
                    "tender_file_path": "upload_files/tender_xxx.pdf",
                    "bid_file_paths": ["upload_files/bid_xxx.pdf"]
                }
            }
        }


class AnalyzeRequest(BaseModel):
    """启动相似度分析请求"""
    tender_file_path: Optional[str] = Field(
        None,
        description="招标文件路径（可为空；为空时仅做投标文件之间互相比对）",
        example="upload_files/tender_xxx.pdf",
    )
    bid_file_paths: List[str] = Field(..., description="投标文件路径列表", example=["upload_files/bid1.pdf", "upload_files/bid2.pdf"])
    
    class Config:
        json_schema_extra = {
            "example": {
                "tender_file_path": "upload_files/tender_xxx.pdf",
                "bid_file_paths": ["upload_files/bid1.pdf", "upload_files/bid2.pdf"]
            }
        }


class AnalyzeResponse(BaseModel):
    """启动分析任务响应"""
    code: int = Field(200, description="响应状态码，200表示成功")
    msg: str = Field(..., description="响应消息")
    data: Dict[str, str] = Field(..., description="响应数据，包含task_id")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "分析任务已启动",
                "data": {
                    "task_id": "abc123-def456-ghi789"
                }
            }
        }


class SimilarityDetail(BaseModel):
    """相似度详情"""
    bid_file: str = Field(..., description="投标文件名")
    page: int = Field(..., description="页码")
    text: str = Field(..., description="雷同文本片段")
    similar_with: str = Field(..., description="相似的文件名")
    similar_page: int = Field(..., description="相似的页码")
    similarity: float = Field(..., description="相似度分数，范围0-1")
    order_changed: bool = Field(False, description="是否语序规避")
    stopword_evade: bool = Field(False, description="是否无意义词插入规避")
    synonym_evade: bool = Field(False, description="是否同义词替换规避")
    semantic_evade: bool = Field(False, description="是否语义规避")
    is_near_identical: bool = Field(False, description="是否几乎相同页面")
    grammar_errors: List[str] = Field(default_factory=list, description="语法错误列表")


class GrammarError(BaseModel):
    """语法错误信息"""
    error: str = Field(..., description="错误描述")
    text: str = Field(..., description="错误文本片段")
    locations: List[Dict[str, Any]] = Field(..., description="错误位置列表")


class SimilarityResult(BaseModel):
    """相似度分析结果"""
    summary: str = Field(..., description="分析结果摘要")
    total_similarity_count: int = Field(..., description="相似片段总数")
    avg_similarity_score: float = Field(..., description="平均相似度分数")
    max_similarity_score: float = Field(..., description="最大相似度分数")
    details: List[SimilarityDetail] = Field(..., description="相似度详情列表")
    grammar_errors: List[GrammarError] = Field(default_factory=list, description="语法错误列表")


class ResultResponse(BaseModel):
    """查询结果响应"""
    code: int = Field(200, description="响应状态码，200表示成功")
    msg: str = Field(..., description="响应消息")
    data: Dict[str, Any] = Field(..., description="响应数据，包含任务状态和结果")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "查询成功",
                "data": {
                    "status": "done",
                    "task_id": "abc123-def456-ghi789",
                    "result": {
                        "summary": "分析完成，发现10处高相似度片段，2组相同语法错误。",
                        "total_similarity_count": 10,
                        "avg_similarity_score": 0.85,
                        "max_similarity_score": 0.95,
                        "details": [],
                        "grammar_errors": []
                    }
                }
            }
        }


class TaskInfo(BaseModel):
    """任务信息"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态：pending/running/done/failed/cancelled")
    tender_file: str = Field(..., description="招标文件路径")
    bid_files: List[str] = Field(..., description="投标文件路径列表")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    progress: Optional[float] = Field(None, description="进度百分比，0-100")


class TaskListResponse(BaseModel):
    """任务列表响应"""
    code: int = Field(200, description="响应状态码，200表示成功")
    msg: str = Field(..., description="响应消息")
    data: Dict[str, List[TaskInfo]] = Field(..., description="任务列表数据")


class CancelTaskRequest(BaseModel):
    """取消任务请求"""
    task_id: str = Field(..., description="任务ID", example="abc123-def456-ghi789")


class CancelTaskResponse(BaseModel):
    """取消任务响应"""
    code: int = Field(200, description="响应状态码，200表示成功")
    msg: str = Field(..., description="响应消息")


class CleanupTasksRequest(BaseModel):
    """清理任务请求"""
    max_age_hours: int = Field(24, description="最大保留时间（小时）", ge=1, le=720)


class CleanupTasksResponse(BaseModel):
    """清理任务响应"""
    code: int = Field(200, description="响应状态码，200表示成功")
    msg: str = Field(..., description="响应消息")


class ExtractedTextItem(BaseModel):
    """提取的文本项"""
    page: int = Field(..., description="页码")
    text: str = Field(..., description="文本内容")
    is_table_cell: bool = Field(False, description="是否为表格单元格")


class BidFileTexts(BaseModel):
    """投标文件文本数据"""
    file_name: str = Field(..., description="文件名")
    texts: List[ExtractedTextItem] = Field(..., description="文本列表")


class AnalyzeExtractedRequest(BaseModel):
    """基于已提取文本的分析请求"""
    tender_texts: Optional[List[ExtractedTextItem]] = Field(
        None,
        description="招标文件文本列表（可为空；为空时仅做投标文件之间互相比对）",
    )
    bid_files: List[BidFileTexts] = Field(..., description="投标文件文本列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tender_texts": [
                    {
                        "page": 1,
                        "text": "招标文件文本内容",
                        "is_table_cell": False
                    }
                ],
                "bid_files": [
                    {
                        "file_name": "投标文件1.pdf",
                        "texts": [
                            {
                                "page": 1,
                                "text": "投标文件文本内容",
                                "is_table_cell": False
                            }
                        ]
                    }
                ]
            }
        }


class FilterResultRequest(BaseModel):
    """筛选结果请求"""
    min_similarity: float = Field(0.0, description="最小相似度阈值", ge=0.0, le=1.0)


# ==================== OCR相关模型 ====================

class OCRUploadRequest(BaseModel):
    """OCR上传请求（用于文档说明，实际使用multipart/form-data）"""
    page_range: Optional[str] = Field(None, description="页面范围，如'1-3'或'1,3,5'，默认所有页面", example="1-10")
    extract_tables: bool = Field(False, description="是否提取表格")
    confidence_threshold: float = Field(0.5, description="OCR置信度阈值", ge=0.0, le=1.0)


class OCRPageResult(BaseModel):
    """OCR页面结果"""
    page_num: int = Field(..., description="页码")
    text: str = Field(..., description="提取的文本")
    text_length: int = Field(..., description="文本长度")
    tables_count: int = Field(0, description="表格数量")
    tables: Optional[List[Dict[str, Any]]] = Field(None, description="表格数据（如果extract_tables=True）")


class OCRExtractResponse(BaseModel):
    """OCR提取响应"""
    code: int = Field(200, description="响应状态码，200表示成功")
    msg: str = Field(..., description="响应消息")
    data: Dict[str, Any] = Field(..., description="提取结果数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "提取成功",
                "data": {
                    "filename": "document.pdf",
                    "file_size": 1024000,
                    "file_type": "application/pdf",
                    "total_pages": 10,
                    "pages": [],
                    "full_text": "提取的完整文本...",
                    "full_text_length": 5000,
                    "processing_time": 2.5
                }
            }
        }


class OCRStatusResponse(BaseModel):
    """OCR状态响应"""
    code: int = Field(200, description="响应状态码，200表示成功")
    msg: str = Field(..., description="响应消息")
    data: Dict[str, Any] = Field(..., description="状态数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "查询成功",
                "data": {
                    "ocr_available": True,
                    "supported_file_types": ["application/pdf", "image/jpeg", "image/png"],
                    "max_file_size": "50MB",
                    "supported_languages": ["中文", "英文"]
                }
            }
        }


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    code: int = Field(200, description="响应状态码，200表示成功")
    msg: str = Field(..., description="响应消息")
    data: Dict[str, Any] = Field(..., description="健康状态数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "服务正常",
                "data": {
                    "status": "healthy",
                    "ocr_service": "available",
                    "timestamp": "2025-01-01T12:00:00"
                }
            }
        }


# ==================== 配置管理相关模型 ====================

class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    config_type: str = Field(..., description="配置类型：similarity或ocr", example="similarity")
    config: Dict[str, Any] = Field(..., description="配置项字典")
    
    class Config:
        json_schema_extra = {
            "example": {
                "config_type": "similarity",
                "config": {
                    "ENABLE_OCR": True,
                    "BID_SIMILARITY_THRESHOLD": 0.85
                }
            }
        }


class ConfigGetResponse(BaseModel):
    """获取配置响应"""
    code: int = Field(200, description="响应状态码，200表示成功")
    msg: str = Field(..., description="响应消息")
    data: Dict[str, Any] = Field(..., description="配置数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "查询成功",
                "data": {
                    "config_type": "similarity",
                    "config": {
                        "ENABLE_OCR": False,
                        "BID_SIMILARITY_THRESHOLD": 0.9
                    }
                }
            }
        }


class ConfigUpdateResponse(BaseModel):
    """配置更新响应"""
    code: int = Field(200, description="响应状态码，200表示成功")
    msg: str = Field(..., description="响应消息")
    data: Dict[str, Any] = Field(..., description="更新后的配置数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "配置更新成功",
                "data": {
                    "config_type": "similarity",
                    "updated_fields": ["ENABLE_OCR", "BID_SIMILARITY_THRESHOLD"]
                }
            }
        }

