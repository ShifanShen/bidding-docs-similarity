"""
API错误码定义
"""
from enum import IntEnum


class ErrorCode(IntEnum):
    """错误码枚举"""
    # 成功
    SUCCESS = 200
    
    # 客户端错误 4xx
    BAD_REQUEST = 400          # 请求参数错误
    UNAUTHORIZED = 401          # 未授权
    FORBIDDEN = 403             # 禁止访问
    NOT_FOUND = 404             # 资源不存在
    METHOD_NOT_ALLOWED = 405    # 方法不允许
    CONFLICT = 409              # 资源冲突
    
    # 服务器错误 5xx
    INTERNAL_ERROR = 500        # 服务器内部错误
    SERVICE_UNAVAILABLE = 503   # 服务不可用
    GATEWAY_TIMEOUT = 504       # 网关超时


class ErrorMessage:
    """错误消息常量"""
    # 通用错误
    SUCCESS = "操作成功"
    BAD_REQUEST = "请求参数错误"
    INTERNAL_ERROR = "服务器内部错误"
    SERVICE_UNAVAILABLE = "服务暂时不可用"
    
    # 文件相关错误
    FILE_UPLOAD_FAILED = "文件上传失败"
    FILE_NOT_FOUND = "文件不存在"
    FILE_TYPE_NOT_SUPPORTED = "不支持的文件类型"
    FILE_TOO_LARGE = "文件过大"
    
    # 任务相关错误
    TASK_NOT_FOUND = "任务不存在"
    TASK_ALREADY_RUNNING = "任务正在运行中"
    TASK_CANNOT_CANCEL = "任务无法取消"
    TASK_START_FAILED = "任务启动失败"
    
    # OCR相关错误
    OCR_SERVICE_UNAVAILABLE = "OCR服务不可用，请检查PaddleOCR配置"
    OCR_PROCESSING_FAILED = "OCR处理失败"
    INVALID_PAGE_RANGE = "无效的页面范围格式"
    
    # 分析相关错误
    ANALYSIS_FAILED = "分析失败"
    RESULT_NOT_FOUND = "分析结果未找到或未完成"
    INVALID_DATA_FORMAT = "数据格式错误"
    DATA_EMPTY = "请求数据不能为空"


def get_error_response(code: int, message: str, detail: str = None) -> dict:
    """生成标准错误响应"""
    return {
        "code": code,
        "msg": message,
        "detail": detail or message,
        "data": None
    }

