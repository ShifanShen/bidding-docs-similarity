"""
自定义异常类
"""


class BiddingDocsException(Exception):
    """基础异常类"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class FileProcessingError(BiddingDocsException):
    """文件处理错误"""
    pass


class TextExtractionError(FileProcessingError):
    """文本提取错误"""
    pass


class OCRServiceError(TextExtractionError):
    """OCR服务错误"""
    pass


class ModelLoadingError(BiddingDocsException):
    """模型加载错误"""
    pass


class SimilarityCalculationError(BiddingDocsException):
    """相似度计算错误"""
    pass


class TaskManagementError(BiddingDocsException):
    """任务管理错误"""
    pass


class ConfigurationError(BiddingDocsException):
    """配置错误"""
    pass


class ValidationError(BiddingDocsException):
    """验证错误"""
    pass


class ResourceError(BiddingDocsException):
    """资源错误"""
    pass
