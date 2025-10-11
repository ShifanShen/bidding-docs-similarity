"""
任务管理器
"""
import time
import uuid
import threading
from typing import Dict, List, Optional, Callable, Any
from threading import Lock, Thread
from app.core.base import TaskInfo, TaskStatus
from app.core.exceptions import TaskManagementError
import logging


class TaskManager:
    """任务管理器"""
    
    def __init__(self, max_concurrent_tasks: int = 3, max_task_timeout: int = 1200):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_task_timeout = max_task_timeout
        self.tasks: Dict[str, TaskInfo] = {}
        self.running_tasks = 0
        self.task_lock = Lock()
        self.task_queue: List[tuple] = []
        self.logger = logging.getLogger(__name__)
        
        # 启动任务处理线程
        self._start_task_processor()
    
    def create_task(self, task_data: Dict[str, Any]) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())
        
        with self.task_lock:
            task_info = TaskInfo(
                task_id=task_id,
                status=TaskStatus.PENDING,
                progress={"current": 0, "total": 0},
                created_at=time.time()
            )
            self.tasks[task_id] = task_info
        
        self.logger.info(f"创建任务: {task_id}")
        return task_id
    
    def start_task(self, task_id: str, task_func: Callable, *args, **kwargs) -> bool:
        """启动任务"""
        with self.task_lock:
            if task_id not in self.tasks:
                raise TaskManagementError(f"任务不存在: {task_id}")
            
            if self.running_tasks >= self.max_concurrent_tasks:
                # 添加到队列
                self.task_queue.append((task_id, task_func, args, kwargs))
                self.logger.info(f"任务 {task_id} 已加入队列")
                return False
            
            task_info = self.tasks[task_id]
            if task_info.status != TaskStatus.PENDING:
                raise TaskManagementError(f"任务状态错误: {task_info.status}")
            
            # 更新任务状态
            task_info.status = TaskStatus.RUNNING
            task_info.started_at = time.time()
            self.running_tasks += 1
        
        # 启动任务线程
        thread = Thread(target=self._execute_task, args=(task_id, task_func, args, kwargs))
        thread.daemon = True
        thread.start()
        
        self.logger.info(f"启动任务: {task_id}")
        return True
    
    def _execute_task(self, task_id: str, task_func: Callable, args: tuple, kwargs: dict) -> None:
        """执行任务"""
        try:
            # 执行任务函数
            result = task_func(*args, **kwargs)
            
            # 更新任务状态
            with self.task_lock:
                if task_id in self.tasks:
                    task_info = self.tasks[task_id]
                    task_info.status = TaskStatus.COMPLETED
                    task_info.result = result
                    task_info.completed_at = time.time()
                    task_info.progress = {"current": 100, "total": 100}
            
            self.logger.info(f"任务完成: {task_id}")
            
        except Exception as e:
            # 处理任务错误
            with self.task_lock:
                if task_id in self.tasks:
                    task_info = self.tasks[task_id]
                    task_info.status = TaskStatus.FAILED
                    task_info.error = str(e)
                    task_info.completed_at = time.time()
            
            self.logger.error(f"任务失败: {task_id}, 错误: {str(e)}")
        
        finally:
            # 减少运行任务数
            with self.task_lock:
                self.running_tasks -= 1
            
            # 处理队列中的下一个任务
            self._process_next_task()
    
    def _process_next_task(self) -> None:
        """处理队列中的下一个任务"""
        with self.task_lock:
            if self.task_queue and self.running_tasks < self.max_concurrent_tasks:
                task_id, task_func, args, kwargs = self.task_queue.pop(0)
                self.start_task(task_id, task_func, *args, **kwargs)
    
    def _start_task_processor(self) -> None:
        """启动任务处理线程"""
        def process_tasks():
            while True:
                try:
                    time.sleep(1)  # 每秒检查一次
                    self._cleanup_expired_tasks()
                except Exception as e:
                    self.logger.error(f"任务处理器错误: {str(e)}")
        
        thread = Thread(target=process_tasks, daemon=True)
        thread.start()
    
    def _cleanup_expired_tasks(self) -> None:
        """清理过期任务"""
        current_time = time.time()
        expired_tasks = []
        
        with self.task_lock:
            for task_id, task_info in self.tasks.items():
                if (task_info.status == TaskStatus.RUNNING and 
                    task_info.started_at and 
                    current_time - task_info.started_at > self.max_task_timeout):
                    expired_tasks.append(task_id)
        
        for task_id in expired_tasks:
            self.cancel_task(task_id)
            self.logger.warning(f"任务超时，已取消: {task_id}")
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        with self.task_lock:
            return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, TaskInfo]:
        """获取所有任务"""
        with self.task_lock:
            return self.tasks.copy()
    
    def update_task_progress(self, task_id: str, progress: Dict[str, Any]) -> None:
        """更新任务进度"""
        with self.task_lock:
            if task_id in self.tasks:
                self.tasks[task_id].progress.update(progress)
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self.task_lock:
            if task_id not in self.tasks:
                return False
            
            task_info = self.tasks[task_id]
            if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            task_info.status = TaskStatus.CANCELLED
            task_info.completed_at = time.time()
            
            # 从队列中移除
            self.task_queue = [(tid, func, args, kwargs) for tid, func, args, kwargs in self.task_queue if tid != task_id]
            
            self.logger.info(f"任务已取消: {task_id}")
            return True
    
    def cleanup_tasks(self, max_age_hours: int = 24) -> int:
        """清理过期任务"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        with self.task_lock:
            expired_tasks = []
            for task_id, task_info in self.tasks.items():
                if (task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    current_time - task_info.created_at > max_age_seconds):
                    expired_tasks.append(task_id)
            
            for task_id in expired_tasks:
                del self.tasks[task_id]
                cleaned_count += 1
        
        self.logger.info(f"清理了 {cleaned_count} 个过期任务")
        return cleaned_count
