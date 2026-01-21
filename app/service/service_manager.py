"""
服务管理器
统一管理所有服务的单例实例，避免重复创建和初始化问题
"""
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# 线程锁，确保单例创建的线程安全
_service_lock = threading.Lock()

# 全局服务实例
_similarity_service = None
_entity_rec_service = None
_entity_regex_service = None
_oss_service = None


def get_similarity_service():
    """获取相似度分析服务单例"""
    global _similarity_service
    
    if _similarity_service is None:
        with _service_lock:
            if _similarity_service is None:
                try:
                    from app.service.similarity_service import SimilarityService
                    logger.info("初始化相似度分析服务...")
                    _similarity_service = SimilarityService()
                    logger.info("相似度分析服务初始化成功")
                except Exception as e:
                    logger.error(f"相似度分析服务初始化失败: {str(e)}", exc_info=True)
                    # 即使初始化失败，也创建一个占位对象，避免后续调用崩溃
                    raise
    
    return _similarity_service


def get_entity_rec_service():
    """获取实体识别服务单例（HanLP+正则，线程安全）"""
    global _entity_rec_service
    
    if _entity_rec_service is None:
        with _service_lock:
            # 双重检查锁定模式
            if _entity_rec_service is None:
                try:
                    from app.service.entity_rec_service import EntityRecService
                    logger.info("初始化实体识别服务（HanLP+正则）...")
                    _entity_rec_service = EntityRecService()
                    logger.info("实体识别服务初始化成功（HanLP将在首次使用时加载）")
                except Exception as e:
                    logger.error(f"实体识别服务初始化失败: {str(e)}", exc_info=True)
                    raise RuntimeError(f"实体识别服务初始化失败: {str(e)}") from e
    
    return _entity_rec_service


def get_entity_regex_service():
    """获取正则实体识别服务单例（线程安全）"""
    global _entity_regex_service
    
    if _entity_regex_service is None:
        with _service_lock:
            # 双重检查锁定模式
            if _entity_regex_service is None:
                try:
                    from app.service.entity_regex_rec_service import EntityRecognitionService
                    logger.info("初始化正则实体识别服务...")
                    _entity_regex_service = EntityRecognitionService()
                    logger.info("正则实体识别服务初始化成功")
                except Exception as e:
                    logger.error(f"正则实体识别服务初始化失败: {str(e)}", exc_info=True)
                    raise RuntimeError(f"正则实体识别服务初始化失败: {str(e)}") from e
    
    return _entity_regex_service


def get_oss_service():
    """获取MinIO OSS服务单例（线程安全，延迟连接）"""
    global _oss_service

    if _oss_service is None:
        with _service_lock:
            if _oss_service is None:
                try:
                    from app.service.oss_service import OSSService
                    logger.info("初始化OSS服务（MinIO）...")
                    _oss_service = OSSService()
                    logger.info("OSS服务初始化成功（MinIO将在首次调用时探测可用性）")
                except Exception as e:
                    logger.error(f"OSS服务初始化失败: {str(e)}", exc_info=True)
                    raise RuntimeError(f"OSS服务初始化失败: {str(e)}") from e

    return _oss_service


def reset_services():
    """重置所有服务实例（用于测试或重新初始化）"""
    global _similarity_service, _entity_rec_service, _entity_regex_service, _oss_service
    
    with _service_lock:
        _similarity_service = None
        _entity_rec_service = None
        _entity_regex_service = None
        _oss_service = None
        logger.info("所有服务实例已重置")
