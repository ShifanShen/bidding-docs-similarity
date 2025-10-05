import os
import logging
from logging.handlers import TimedRotatingFileHandler
import datetime

class LogConfig:
    # 日志目录
    LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs'))
    
    # 日志文件格式
    LOG_FILE_FORMAT = '%Y-%m-%d.log'
    
    # 日志级别
    LOG_LEVEL = logging.INFO
    
    # 日志格式字符串
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def setup_logging(cls):
        """
        设置全局日志配置，包括控制台和文件输出
        """
        # 确保日志目录存在
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        
        # 获取root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(cls.LOG_LEVEL)
        
        # 清除现有的handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(cls.LOG_LEVEL)
        
        # 创建TimedRotatingFileHandler，每天一个文件
        log_file_path = os.path.join(cls.LOG_DIR, datetime.datetime.now().strftime(cls.LOG_FILE_FORMAT))
        file_handler = TimedRotatingFileHandler(
            log_file_path,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.setLevel(cls.LOG_LEVEL)
        
        # 设置日志格式
        formatter = logging.Formatter(cls.LOG_FORMAT)
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 添加handlers到logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # 设置常见第三方库的日志级别，减少不必要的日志输出
        for lib_name in ['uvicorn', 'fastapi', 'paddle', 'PIL']:
            logging.getLogger(lib_name).setLevel(logging.WARNING)

# 创建默认日志配置实例
log_config = LogConfig()