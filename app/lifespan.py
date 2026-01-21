"""
应用生命周期管理
在应用启动和关闭时执行初始化/清理操作
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    在启动时初始化服务，在关闭时清理资源
    """
    # 启动时初始化
    logger.info("=" * 60)
    logger.info("应用启动中...")
    
    try:
        # 启动时只做轻量级检查，不加载大型模型
        # 所有服务都采用延迟加载策略，在首次API调用时初始化
        logger.info("服务采用延迟加载策略，将在首次使用时初始化")
        logger.info("=" * 60)
        logger.info("应用启动完成（服务延迟加载）")
        
    except Exception as e:
        logger.error(f"应用启动过程中发生错误: {str(e)}", exc_info=True)
        # 不阻止应用启动，但记录错误
    
    yield
    
    # 关闭时清理
    logger.info("应用关闭中，清理资源...")
    try:
        # 这里可以添加清理逻辑，如关闭数据库连接等
        pass
    except Exception as e:
        logger.error(f"清理资源时发生错误: {str(e)}")
    
    logger.info("应用已关闭")
