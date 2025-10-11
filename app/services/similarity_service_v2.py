"""
重构后的相似度分析服务
"""
import os
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from app.core.base import BaseService, TaskInfo, TaskStatus, SimilarityResult
from app.core.exceptions import BiddingDocsException, TaskManagementError
from app.core.config_manager import config_manager
from app.core.task_manager import TaskManager
from app.services.text_extractor import TextExtractorService
from app.services.similarity_calculator import SimilarityCalculatorService
from app.services.text_processor import TextProcessorService


class SimilarityAnalysisService(BaseService):
    """相似度分析服务（重构版）"""
    
    def __init__(self):
        super().__init__(config_manager.get_similarity_config())
        self.logger = logging.getLogger(__name__)
        
        # 初始化子服务
        self.text_extractor = TextExtractorService()
        self.similarity_calculator = SimilarityCalculatorService()
        self.text_processor = TextProcessorService()
        self.task_manager = TaskManager(
            max_concurrent_tasks=self.config.MAX_CONCURRENT_TASKS,
            max_task_timeout=self.config.MAX_TASK_TIMEOUT
        )
        
        # 确保存储目录存在
        os.makedirs(self.config.STORAGE_DIR, exist_ok=True)
    
    def initialize(self) -> bool:
        """初始化服务"""
        try:
            # 检查子服务是否可用
            if not self.text_extractor.is_available():
                self.logger.error("文本提取服务不可用")
                return False
            
            self.logger.info("相似度分析服务初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"服务初始化失败: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            # 清理任务管理器
            self.task_manager.cleanup_tasks()
            self.logger.info("资源清理完成")
        except Exception as e:
            self.logger.error(f"资源清理失败: {str(e)}")
    
    def analyze_similarity(self, tender_file_path: str, bid_file_paths: List[str]) -> str:
        """分析相似度（异步）"""
        try:
            # 验证文件
            self._validate_files(tender_file_path, bid_file_paths)
            
            # 创建任务
            task_data = {
                'tender_file': tender_file_path,
                'bid_files': bid_file_paths,
                'created_at': time.time()
            }
            task_id = self.task_manager.create_task(task_data)
            
            # 启动任务
            self.task_manager.start_task(task_id, self._analyze_task, task_id, tender_file_path, bid_file_paths)
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"分析任务创建失败: {str(e)}")
            raise BiddingDocsException(f"分析任务创建失败: {str(e)}")
    
    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务状态"""
        return self.task_manager.get_task(task_id)
    
    def get_all_tasks(self) -> Dict[str, TaskInfo]:
        """获取所有任务"""
        return self.task_manager.get_all_tasks()
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        return self.task_manager.cancel_task(task_id)
    
    def cleanup_tasks(self, max_age_hours: int = 24) -> int:
        """清理过期任务"""
        return self.task_manager.cleanup_tasks(max_age_hours)
    
    def _validate_files(self, tender_file_path: str, bid_file_paths: List[str]) -> None:
        """验证文件"""
        if not os.path.exists(tender_file_path):
            raise BiddingDocsException(f"招标文件不存在: {tender_file_path}")
        
        for bid_file in bid_file_paths:
            if not os.path.exists(bid_file):
                raise BiddingDocsException(f"投标文件不存在: {bid_file}")
    
    def _analyze_task(self, task_id: str, tender_file_path: str, bid_file_paths: List[str]) -> Dict[str, Any]:
        """执行分析任务"""
        try:
            self.logger.info(f"开始分析任务: {task_id}")
            
            # 更新任务进度
            self.task_manager.update_task_progress(task_id, {"current": 0, "total": 100, "stage": "开始分析"})
            
            # 1. 提取招标文件文本
            self.task_manager.update_task_progress(task_id, {"current": 10, "total": 100, "stage": "提取招标文件"})
            tender_segments = self.text_extractor.extract(tender_file_path)
            tender_segments = self.text_processor.process_segments(tender_segments)
            
            # 2. 提取投标文件文本
            self.task_manager.update_task_progress(task_id, {"current": 20, "total": 100, "stage": "提取投标文件"})
            bid_segments_list = []
            for bid_file in bid_file_paths:
                segments = self.text_extractor.extract(bid_file)
                segments = self.text_processor.process_segments(segments)
                bid_segments_list.append(segments)
            
            # 3. 向量化
            self.task_manager.update_task_progress(task_id, {"current": 30, "total": 100, "stage": "文本向量化"})
            tender_vectors = self._vectorize_segments(tender_segments)
            bid_vectors_list = [self._vectorize_segments(segments) for segments in bid_segments_list]
            
            # 4. 过滤与招标文件相似的内容
            self.task_manager.update_task_progress(task_id, {"current": 40, "total": 100, "stage": "过滤招标文件内容"})
            filtered_segments, filtered_vectors = self._filter_tender_similar_content(
                bid_segments_list, bid_vectors_list, tender_vectors
            )
            
            # 5. 投标文件间相似度分析
            self.task_manager.update_task_progress(task_id, {"current": 50, "total": 100, "stage": "相似度分析"})
            similarity_results = self._analyze_bid_similarity(
                filtered_segments, filtered_vectors, bid_file_paths
            )
            
            # 6. 生成结果
            self.task_manager.update_task_progress(task_id, {"current": 90, "total": 100, "stage": "生成结果"})
            result = self._generate_analysis_result(
                similarity_results, tender_file_path, bid_file_paths
            )
            
            # 7. 保存结果
            self.task_manager.update_task_progress(task_id, {"current": 95, "total": 100, "stage": "保存结果"})
            self._save_result(task_id, result)
            
            self.task_manager.update_task_progress(task_id, {"current": 100, "total": 100, "stage": "完成"})
            
            self.logger.info(f"分析任务完成: {task_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"分析任务失败: {task_id}, 错误: {str(e)}")
            raise
    
    def _vectorize_segments(self, segments: List) -> List:
        """向量化文档片段"""
        texts = [segment.text for segment in segments]
        if not texts:
            return []
        
        vectors = self.similarity_calculator.encode_texts(texts)
        return vectors.tolist()
    
    def _filter_tender_similar_content(self, bid_segments_list: List, bid_vectors_list: List, tender_vectors: List) -> Tuple[List, List]:
        """过滤与招标文件相似的内容"""
        # 这里实现过滤逻辑
        # 简化版本，实际应该实现完整的过滤逻辑
        return bid_segments_list, bid_vectors_list
    
    def _analyze_bid_similarity(self, segments_list: List, vectors_list: List, file_paths: List[str]) -> List[SimilarityResult]:
        """分析投标文件间相似度"""
        # 这里实现相似度分析逻辑
        # 简化版本，实际应该实现完整的分析逻辑
        return []
    
    def _generate_analysis_result(self, similarity_results: List[SimilarityResult], tender_file: str, bid_files: List[str]) -> Dict[str, Any]:
        """生成分析结果"""
        return {
            'task_id': '',
            'tender_file': tender_file,
            'bid_files': bid_files,
            'similarity_results': similarity_results,
            'summary': {
                'total_similarities': len(similarity_results),
                'analysis_time': time.time()
            }
        }
    
    def _save_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """保存结果"""
        result_file = os.path.join(self.config.STORAGE_DIR, f"similarity_result_{task_id}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存: {result_file}")
