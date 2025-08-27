import os
import threading
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from threading import Thread, Lock, Timer
import numpy as np
from sentence_transformers import SentenceTransformer
from app.service.text_utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    split_text_to_segments,
    remove_stopwords,
    detect_grammar_errors,
    is_order_changed,
    is_stopword_evade,
    remove_numbers,
    is_synonym_evade
)
import faiss
import gc
import torch
import logging
import psutil
from app.config.similarity_config import default_config
from app.config.terms_config import COMMON_TERMS

# 配置日志格式
logging.basicConfig(level=getattr(logging, default_config.LOG_LEVEL), 
                    format='%(asctime)s %(levelname)s %(message)s')

# =============================
# 相似度分析服务主类
# =============================
class SimilarityService:
    def __init__(self, config=None):
        """
        初始化服务，包括：
        - 创建存储目录
        - 加载本地文本向量模型
        - 加载停用词表
        - 初始化任务队列和并发控制
        
        参数:
        - config: 配置对象，如不提供则使用默认配置
        """
        # 使用配置对象或默认配置
        self.config = config or default_config
        self.storage_dir = self.config.STORAGE_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # 任务管理相关
        self.tasks: Dict[str, Dict[str, Any]] = {}  # 存储所有任务信息
        self.running_tasks = 0  # 当前运行的任务数
        self.task_lock = Lock()  # 任务同步锁
        self.task_queue: List[Tuple[str, str, List[str]]] = []  # 任务队列
        
        # 模型和资源初始化
        self.model = self._load_text2vec_model()  # 文本向量化模型
        self.stopwords = self._load_stopwords()  # 停用词集合
        self.stopwords_list = list(self.stopwords)  # 预转换为列表以提高性能
        
        # 预计算和缓存配置值，避免重复访问
        self.max_concurrent_tasks = self.config.MAX_CONCURRENT_TASKS
        self.table_processing_mode = self.config.TABLE_PROCESSING_MODE \
            if self.config.TABLE_PROCESSING_MODE in ["cell", "row"] else "row"
        self.min_text_length = self.config.MIN_TEXT_LENGTH
        self.tender_similarity_threshold = self.config.TENDER_SIMILARITY_THRESHOLD
        self.bid_similarity_threshold = self.config.BID_SIMILARITY_THRESHOLD
        self.batch_size = self.config.BATCH_SIZE
        self.max_task_timeout = self.config.MAX_TASK_TIMEOUT
        self.memory_cleanup_interval = self.config.MEMORY_CLEANUP_INTERVAL
        self.similarity_top_k = self.config.SIMILARITY_TOP_K
        
        # 预计算表格相似度阈值（提高区分度）
        self.table_row_threshold = self.bid_similarity_threshold + self.config.TABLE_ROW_THRESHOLD_OFFSET
        self.table_cell_threshold = self.bid_similarity_threshold + self.config.TABLE_CELL_THRESHOLD_OFFSET
        
        # 规避行为检测阈值
        self.high_similarity_threshold = self.config.HIGH_SIMILARITY_THRESHOLD
        self.very_high_similarity_threshold = self.config.VERY_HIGH_SIMILARITY_THRESHOLD
        self.common_term_count_threshold = self.config.COMMON_TERM_COUNT_THRESHOLD
        self.semantic_evade_lower_threshold = self.config.SEMANTIC_EVADE_LOWER_THRESHOLD
        self.semantic_evade_upper_threshold = self.config.SEMANTIC_EVADE_UPPER_THRESHOLD
        
        # 内存清理相关
        self.cleanup_counter = 0  # 初始化内存清理计数器
        self.cleanup_interval = self.memory_cleanup_interval  # 使用配置的内存清理间隔
        
        
    def _load_text2vec_model(self) -> SentenceTransformer:
        """
        加载本地 text2vec 语义向量模型，根据配置决定是否使用 GPU。
        """
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../local_text2vec_model'))
        model = SentenceTransformer(model_path)
        if self.config.ENABLE_GPU and torch.cuda.is_available():
            model = model.to('cuda')
        return model

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        对文本列表进行向量化，在向量化前先移除数字，避免无意义标号影响相似度计算。
        """
        # 移除文本中的数字后再进行向量化
        texts_without_numbers = [remove_numbers(text) for text in texts]
        return self.model.encode(texts_without_numbers, convert_to_numpy=True, show_progress_bar=False)

    def _load_stopwords(self) -> set:
        """
        加载停用词表。
        """
        stopwords_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../stopwords.txt'))
        if os.path.exists(stopwords_path):
            with open(stopwords_path, encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def save_file(self, file_bytes: bytes, filename: str) -> str:
        """
        保存上传的文件到本地存储目录。
        """
        safe_name = filename.replace("..", "_").replace("/", "_")
        save_path = os.path.join(self.storage_dir, safe_name)
        with open(save_path, "wb") as f:
            f.write(file_bytes)
        return save_path

    def start_analysis(self, tender_file_path: str, bid_file_paths: List[str]) -> str:
        """
        启动一次分析任务，将任务加入队列并异步处理。
        """
        task_id = str(uuid.uuid4())
        task_info = {
            "status": "pending",
            "result": None,
            "progress": {"current": 0, "total": 1},
            "created_time": time.time(),
            "file_info": {
                "tender_file": os.path.basename(tender_file_path),
                "bid_files": [os.path.basename(p) for p in bid_file_paths],
                "bid_count": len(bid_file_paths)
            }
        }
        self.tasks[task_id] = task_info
        self.task_queue.append((task_id, tender_file_path, bid_file_paths))
        self._process_queue()
        return task_id

    def _process_queue(self) -> None:
        """
        处理任务队列，控制最大并发数。
        """
        with self.task_lock:
            while self.running_tasks < self.max_concurrent_tasks and self.task_queue:
                task_id, tender_file_path, bid_file_paths = self.task_queue.pop(0)
                self.running_tasks += 1
                thread = Thread(target=self._analyze_task, args=(task_id, tender_file_path, bid_file_paths))
                thread.daemon = True
                thread.start()

    def _extract_and_segment(self, file_path: str) -> List[Dict[str, Any]]:
        """
        文本提取与分段：支持 PDF、Word 文档。
        提取表格单元格并进行单独处理。
        返回：[{page, text, grammar_errors, is_table_cell, row, col, table_idx}]
        """
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            pages = extract_text_from_pdf(file_path)
            segments = []
            for page_data in pages:
                page_num = page_data['page_num']
                
                # 处理普通文本
                for seg in split_text_to_segments(page_data['text']):
                    clean_seg = remove_stopwords(seg, list(self.stopwords))
                    if clean_seg and len(clean_seg) >= self.min_text_length:
                        grammar_errors = detect_grammar_errors(clean_seg)
                        segments.append({
                            'page': page_num,
                            'text': clean_seg,
                            'grammar_errors': grammar_errors,
                            'is_table_cell': False
                        })
                
                # 处理表格
                for table in page_data['tables']:
                    table_idx = table['table_idx']
                    
                    if self.table_processing_mode == 'row':
                        # 按行处理表格
                        rows = {}  # 行索引 -> 行文本
                        for cell in table['cells']:
                            row_idx = cell['row']
                            if row_idx not in rows:
                                rows[row_idx] = []
                            rows[row_idx].append(cell['text'])
                        
                        # 合并每行的单元格文本
                        for row_idx in sorted(rows.keys()):
                            row_text = ' '.join(rows[row_idx])
                            clean_row = remove_stopwords(row_text, list(self.stopwords))
                            if clean_row and len(clean_row) >= self.min_text_length:
                                grammar_errors = detect_grammar_errors(clean_row)
                                segments.append({
                                    'page': page_num,
                                    'text': clean_row,
                                    'grammar_errors': grammar_errors,
                                    'is_table_cell': True,
                                    'row': row_idx,
                                    'table_idx': table_idx
                                })
                    else:
                        # 按单元格处理表格
                        for cell in table['cells']:
                            clean_cell = remove_stopwords(cell['text'], list(self.stopwords))
                            if clean_cell and len(clean_cell) >= self.min_text_length:
                                grammar_errors = detect_grammar_errors(clean_cell)
                                segments.append({
                                    'page': page_num,
                                    'text': clean_cell,
                                    'grammar_errors': grammar_errors,
                                    'is_table_cell': True,
                                    'row': cell['row'],
                                    'col': cell['col'],
                                    'table_idx': table_idx
                                })
            return segments
        elif ext in ['.doc', '.docx']:
            paras = extract_text_from_docx(file_path)
            segments = []
            for i, para in enumerate(paras):
                for seg in split_text_to_segments(para):
                    clean_seg = remove_stopwords(seg, list(self.stopwords))
                    if clean_seg and len(clean_seg) >= self.min_text_length:
                        grammar_errors = detect_grammar_errors(clean_seg)
                        segments.append({
                            'page': i+1,
                            'text': clean_seg,
                            'grammar_errors': grammar_errors,
                            'is_table_cell': False
                        })
            return segments
        else:
            return []

    def _create_faiss_index(self, dim: int) -> faiss.Index:
        """
        创建FAISS索引，根据配置和硬件情况决定是否使用GPU。
        """
        if self.config.ENABLE_GPU and hasattr(faiss, 'StandardGpuResources') and torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                return faiss.GpuIndexFlatIP(res, dim)
            except Exception:
                # 如果GPU初始化失败，回退到CPU
                return faiss.IndexFlatIP(dim)
        else:
            return faiss.IndexFlatIP(dim)

    def _faiss_max_sim(self, query_vecs: np.ndarray, base_vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 faiss 进行最大相似度检索，返回每个查询向量的最高相似度及其索引。
        """
        if len(base_vecs) == 0 or len(query_vecs) == 0:
            return np.array([]), np.array([])
        dim = base_vecs.shape[1]
        
        index = self._create_faiss_index(dim)
        faiss.normalize_L2(base_vecs)
        faiss.normalize_L2(query_vecs)
        index.add(base_vecs)
        sims, idxs = index.search(query_vecs, self.similarity_top_k)
        return sims.flatten(), idxs.flatten()

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        分批向量化文本，防止内存溢出（OOM）。
        优化点：使用预分配的ndarray而不是列表收集结果，减少内存分配开销
        """
        if not texts:
            # 使用空列表测试获取模型的实际输出维度
            sample_vec = self._encode_texts([])
            dim = sample_vec.shape[1] if len(sample_vec.shape) > 1 else 384
            return np.zeros((0, dim))
        
        # 获取模型的实际输出维度
        first_batch = texts[:1]  # 使用第一个文本作为样本来确定维度
        sample_vec = self._encode_texts(first_batch)
        dim = sample_vec.shape[1] if len(sample_vec.shape) > 1 else 384
        
        # 预分配结果数组，避免多次内存分配
        total_len = len(texts)
        result = np.zeros((total_len, dim), dtype=np.float32)
        
        # 先处理第一个批次（已经计算过了）
        result[0] = sample_vec[0] if sample_vec.shape[0] > 0 else np.zeros(dim)
        
        # 处理剩余批次
        for i in range(1, total_len, self.batch_size):
            end_idx = min(i + self.batch_size, total_len)
            batch = texts[i:end_idx]
            vecs = self._encode_texts(batch)
            result[i:end_idx] = vecs
            
            # 定期清理内存，但避免过于频繁
            if (i // self.batch_size) % self.memory_cleanup_interval == 0:
                self._cleanup_memory()
        
        return result

    def _cleanup_memory(self) -> None:
        """
        清理内存和GPU缓存。
        优化点：使用计数器减少gc调用频率，提高性能
        """
        self.cleanup_counter += 1
        if self.cleanup_counter >= self.cleanup_interval:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.cleanup_counter = 0

    def _log_resource(self, tag: str, extra: Optional[dict] = None) -> None:
        """
        记录当前进程的内存和 CPU 占用，便于监控。
        """
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024  # MB
        cpu = process.cpu_percent(interval=0.1)
        log_msg = f"[{tag}] mem={mem:.1f}MB cpu={cpu:.1f}%"
        if extra:
            log_msg += f" | {extra}"
        logging.info(log_msg)

    def _analyze_task(self, task_id: str, tender_file_path: str, bid_file_paths: List[str]) -> None:
        """
        任务主流程：
        1. 招标文件分段并向量化
        2. 投标文件分段并向量化
        3. 剔除与招标文件高度相似的投标片段
        4. 投标文件间两两比对，检测相似片段及规避行为
        5. 统计语法错误，找出多份文件中相同错误
        """
        start_time = time.time()
        self._log_resource('TASK_START', {'task_id': task_id})
        timeout_flag = {'timeout': False}
        
        # 设置超时计时器
        def timeout_callback() -> None:
            timeout_flag['timeout'] = True
            self.tasks[task_id]["status"] = "timeout"
            self.tasks[task_id]["result"] = {"error": f"任务执行超时 ({self.max_task_timeout}秒)"}
            self._log_resource('TASK_TIMEOUT', {'task_id': task_id})
            
        timer = threading.Timer(self.max_task_timeout, timeout_callback)
        timer.daemon = True
        timer.start()
        
        try:
            # 初始化进度跟踪
            self.tasks[task_id]["status"] = "running"
            total_comparisons = len(bid_file_paths) * (len(bid_file_paths) - 1) // 2
            self.tasks[task_id]["progress"] = {"current": 0, "total": total_comparisons}

            # 1. 处理招标文件
            tender_segments, tender_vecs = self._process_tender_document(tender_file_path)
            
            # 2. 处理投标文件
            bid_segments_list, bid_vecs_list = self._process_bid_documents(bid_file_paths)
            
            # 3. 剔除与招标文件高度相似的投标片段
            filtered_bid_segments, filtered_bid_vecs = self._filter_bid_segments(
                bid_segments_list, bid_vecs_list, tender_vecs
            )
            
            # 4. 投标文件间两两比对，检测相似片段及规避行为
            details = self._compare_bid_documents(
                filtered_bid_segments, filtered_bid_vecs, bid_file_paths, task_id
            )
            
            # 5. 统计所有投标文件分段的语法错误
            common_grammar_errors = self._collect_common_grammar_errors(
                filtered_bid_segments, bid_file_paths
            )
            
            # 6. 生成分析结果
            self._generate_result(
                task_id, details, common_grammar_errors, filtered_bid_segments, 
                bid_file_paths, start_time
            )
            
            if timeout_flag['timeout']:
                return
        except Exception as e:
            self.tasks[task_id]["status"] = "error"
            self.tasks[task_id]["result"] = {"error": str(e), "details": str(e.__traceback__)}
            self._log_resource('TASK_ERROR', {'task_id': task_id, 'error': str(e)})
            # 记录异常堆栈
            logging.error(f"Task {task_id} failed with error: {str(e)}", exc_info=True)
        finally:
            try:
                timer.cancel()
            except Exception:
                pass
            # 最终清理内存
            self._cleanup_memory()
            with self.task_lock:
                self.running_tasks -= 1
                self._process_queue()
            self._log_resource('TASK_FINALLY', {'task_id': task_id})

    def _process_tender_document(self, tender_file_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        处理招标文件：提取文本并向量化
        """
        tender_segments = self._extract_and_segment(tender_file_path)
        tender_texts = [seg['text'] for seg in tender_segments]
        tender_vecs = self._batch_encode(tender_texts) if tender_texts else np.zeros((1, 384))
        if isinstance(tender_vecs, list):
            tender_vecs = np.array(tender_vecs)
        return tender_segments, tender_vecs
        
    def _process_bid_documents(self, bid_file_paths: List[str]) -> Tuple[List[List[Dict[str, Any]]], List[np.ndarray]]:
        """
        处理投标文件：提取文本并向量化
        """
        bid_segments_list = []
        bid_vecs_list = []
        for bid_path in bid_file_paths:
            segs = self._extract_and_segment(bid_path)
            texts = [s['text'] for s in segs]
            vecs = self._batch_encode(texts) if texts else np.zeros((1, 384))
            if isinstance(vecs, list):
                vecs = np.array(vecs)
            bid_segments_list.append(segs)
            bid_vecs_list.append(vecs)
            del segs, texts, vecs
            self._cleanup_memory()
        return bid_segments_list, bid_vecs_list
        
    def _filter_bid_segments(self, 
                            bid_segments_list: List[List[Dict[str, Any]]],
                            bid_vecs_list: List[np.ndarray],
                            tender_vecs: np.ndarray) -> Tuple[List[List[Dict[str, Any]]], List[np.ndarray]]:
        """
        剔除投标文件中与招标文件高度相似的片段
        """
        filtered_bid_segments = []
        filtered_bid_vecs = []
        
        for segs, vecs in zip(bid_segments_list, bid_vecs_list):
            if isinstance(vecs, list):
                vecs = np.array(vecs)
            keep_idx = []
            
            # 如果招标文件向量为空，则保留所有投标片段
            if tender_vecs.shape[0] == 0:
                keep_idx = list(range(len(segs)))
            else:
                for i in range(0, len(segs), self.batch_size):
                    batch_vecs = vecs[i:i+self.batch_size]
                    if batch_vecs.shape[0] > 0:
                        sims, _ = self._faiss_max_sim(batch_vecs, tender_vecs)
                        # 只保留与招标文件相似度低于阈值的片段
                        max_seg_idx = len(segs) - 1
                        keep_idx.extend([min(i+ii, max_seg_idx) for ii, sim in enumerate(sims) if sim < self.tender_similarity_threshold])
                    # 定期清理内存
                    if (i // self.batch_size) % self.memory_cleanup_interval == 0:
                        self._cleanup_memory()
            
            # 处理空keep_idx的情况，并确保索引有效
            if keep_idx:
                # 确保所有索引都在有效范围内
                valid_keep_idx = [idx for idx in keep_idx if 0 <= idx < len(segs)]
                filtered_bid_segments.append([segs[idx] for idx in valid_keep_idx])
            else:
                filtered_bid_segments.append([])
            
            filtered_bid_vecs.append(vecs[keep_idx] if len(keep_idx) > 0 else np.zeros((1, 384)))
            del segs, vecs
            self._cleanup_memory()
        
        return filtered_bid_segments, filtered_bid_vecs
        
    def _compare_bid_documents(self, 
                              filtered_bid_segments: List[List[Dict[str, Any]]],
                              filtered_bid_vecs: List[np.ndarray],
                              bid_file_paths: List[str],
                              task_id: str) -> List[Dict[str, Any]]:
        """
        投标文件间两两比对，检测相似片段及规避行为
        """
        details = []
        progress_cnt = 0
        total_comparisons = len(filtered_bid_segments) * (len(filtered_bid_segments) - 1) // 2
        self.tasks[task_id]["progress"] = {"current": 0, "total": total_comparisons}
        
        for i in range(len(filtered_bid_segments)):
            for j in range(i+1, len(filtered_bid_segments)):
                segs_i, vecs_i = filtered_bid_segments[i], filtered_bid_vecs[i]
                segs_j, vecs_j = filtered_bid_segments[j], filtered_bid_vecs[j]
                
                if isinstance(vecs_i, list):
                    vecs_i = np.array(vecs_i)
                if isinstance(vecs_j, list):
                    vecs_j = np.array(vecs_j)
                
                if vecs_i.shape[0] == 0 or vecs_j.shape[0] == 0:
                    progress_cnt += 1
                    self.tasks[task_id]["progress"]["current"] = progress_cnt
                    continue
                
                # 构建索引一次，多次查询
                dim = vecs_j.shape[1]
                index = self._create_faiss_index(dim)
                faiss.normalize_L2(vecs_j)
                index.add(vecs_j)
                
                for k in range(0, vecs_i.shape[0], self.batch_size):
                    batch_vecs = vecs_i[k:k+self.batch_size]
                    faiss.normalize_L2(batch_vecs)
                    sims, idxs = index.search(batch_vecs, self.similarity_top_k)
                    
                    for idx_b in range(len(batch_vecs)):
                        for top_k in range(self.similarity_top_k):
                            sim = sims[idx_b][top_k]
                            max_idx = idxs[idx_b][top_k]
                            idx_i = k + idx_b
                            
                            # 确保索引有效
                            if max_idx >= len(segs_j):
                                continue
                            
                            text1 = segs_i[idx_i]['text']
                            text2 = segs_j[max_idx]['text']
                            
                            # 表格元素识别与分类
                            is_table_cell_i = segs_i[idx_i].get('is_table_cell', False)
                            is_table_cell_j = segs_j[max_idx].get('is_table_cell', False)
                            is_table_row_i = is_table_cell_i and 'col' not in segs_i[idx_i]
                            is_table_row_j = is_table_cell_j and 'col' not in segs_j[max_idx]
                            is_table_cell_only_i = is_table_cell_i and 'col' in segs_i[idx_i]
                            is_table_cell_only_j = is_table_cell_j and 'col' in segs_j[max_idx]
                              
                            # 判断是否为有效相似片段
                            is_valid = self._is_valid_similarity(
                                sim, text1, text2, 
                                is_table_cell_i, is_table_cell_j,
                                is_table_row_i, is_table_row_j,
                                is_table_cell_only_i, is_table_cell_only_j,
                                segs_i[idx_i], segs_j[max_idx]
                            )
                            
                            if is_valid:
                                # 检测规避行为
                                evade_results = self._detect_evasion_behavior(text1, text2, sim)
                                
                                # 构建相似片段详情
                                detail = self._build_similarity_detail(
                                    bid_file_paths[i], bid_file_paths[j],
                                    segs_i[idx_i], segs_j[max_idx],
                                    sim, evade_results,
                                    is_table_cell_i, is_table_cell_j,
                                    top_k
                                )
                                
                                details.append(detail)
                        # 定期清理内存
                        if (k // self.batch_size) % self.memory_cleanup_interval == 0:
                            self._cleanup_memory()
                    
                    progress_cnt += 1
                    self.tasks[task_id]["progress"]["current"] = progress_cnt
                    
                    # 清理索引资源
                    try:
                        del index
                    except:
                        pass
        
        return details
        
    def _is_valid_similarity(self, sim: float, text1: str, text2: str,
                           is_table_cell_i: bool, is_table_cell_j: bool,
                           is_table_row_i: bool, is_table_row_j: bool,
                           is_table_cell_only_i: bool, is_table_cell_only_j: bool,
                           seg_i: Dict[str, Any], seg_j: Dict[str, Any]) -> bool:
        """
        判断两个文本片段是否为有效相似
        """
        # 表格元素匹配条件
        row_match = False
        col_match = False
        table_match = False
        if (is_table_row_i and is_table_row_j) or (is_table_cell_only_i and is_table_cell_only_j):
            row_i = seg_i.get('row')
            row_j = seg_j.get('row')
            table_i = seg_i.get('table_idx')
            table_j = seg_j.get('table_idx')
            row_match = row_i == row_j
            table_match = table_i == table_j
            if is_table_cell_only_i and is_table_cell_only_j:
                col_i = seg_i.get('col')
                col_j = seg_j.get('col')
                col_match = col_i == col_j
          
        # 确定当前阈值
        if is_table_row_i and is_table_row_j:
            current_threshold = self.table_row_threshold
        elif is_table_cell_only_i and is_table_cell_only_j:
            current_threshold = self.table_cell_threshold
        else:
            current_threshold = self.bid_similarity_threshold
          
        # 判断是否为有效相似片段
        is_valid = (sim > current_threshold and 
                    len(text1) >= self.min_text_length and 
                    len(text2) >= self.min_text_length and 
                    # 表格元素需要满足相应的匹配条件
                    ((not is_table_cell_i or not is_table_cell_j) or 
                     (is_table_row_i and is_table_row_j and row_match and table_match) or 
                     (is_table_cell_only_i and is_table_cell_only_j and row_match and col_match and table_match)))
        
        # 额外过滤：排除相似度极高但实际是常见模板文本的情况
        if is_valid and sim > self.config.HIGH_SIMILARITY_THRESHOLD:
            # 检查是否包含大量相同的专业术语或固定表述
            term_count = sum(1 for term in COMMON_TERMS if term in text1 and term in text2)
            if term_count > self.config.COMMON_TERM_COUNT_THRESHOLD:
                # 降低极高相似度的通用模板文本的阈值要求
                is_valid = sim > self.config.VERY_HIGH_SIMILARITY_THRESHOLD
        
        return is_valid
        
    def _detect_evasion_behavior(self, text1: str, text2: str, sim: float) -> Dict[str, bool]:
        """
        检测各种规避行为
        """
        # 检测语序变更规避
        order_changed = is_order_changed(text1, text2)
        
        # 检测停用词插入规避
        stopword_evade = is_stopword_evade(text1, text2, list(self.stopwords))
        
        # 检测同义词替换规避
        synonym_evade = False
        if not stopword_evade and sim < 0.95:
            synonym_evade = is_synonym_evade(text1, text2)
        
        # 检测语义保持但词汇变化较大的情况
        semantic_evade = False
        if sim > self.config.SEMANTIC_EVADE_LOWER_THRESHOLD and \
           sim < self.config.SEMANTIC_EVADE_UPPER_THRESHOLD and \
           not order_changed and not stopword_evade and not synonym_evade:
            # 可能存在更高级的规避行为
            semantic_evade = True
        
        return {
            'order_changed': order_changed,
            'stopword_evade': stopword_evade,
            'synonym_evade': synonym_evade,
            'semantic_evade': semantic_evade
        }
        
    def _build_similarity_detail(self, bid_file_i: str, bid_file_j: str,
                               seg_i: Dict[str, Any], seg_j: Dict[str, Any],
                               sim: float, evade_results: Dict[str, bool],
                               is_table_cell_i: bool, is_table_cell_j: bool,
                               top_k: int) -> Dict[str, Any]:
        """
        构建相似片段详情字典
        """
        detail = {
            'bid_file': os.path.basename(bid_file_i),
            'page': seg_i['page'],
            'text': seg_i['text'],
            'grammar_errors': seg_i.get('grammar_errors', []),
            'similar_with': os.path.basename(bid_file_j),
            'similar_page': seg_j['page'],
            'similarity': float(f'{sim:.4f}'),
            'similar_text': seg_j['text'],
            'similar_grammar_errors': seg_j.get('grammar_errors', []),
            'order_changed': evade_results['order_changed'],
            'stopword_evade': evade_results['stopword_evade'],
            'synonym_evade': evade_results['synonym_evade'],
            'semantic_evade': evade_results['semantic_evade'],
            'is_table_element': is_table_cell_i,
            'similar_is_table_element': is_table_cell_j,
            'rank': top_k + 1
        }
        
        # 如果是表格元素，添加额外信息
        if is_table_cell_i:
            table_info = {
                'row': seg_i.get('row'),
                'table_idx': seg_i.get('table_idx')
            }
            # 只有按单元格处理时才添加列信息
            if 'col' in seg_i:
                table_info['col'] = seg_i.get('col')
            detail.update(table_info)
        
        if is_table_cell_j:
            similar_table_info = {
                'similar_row': seg_j.get('row'),
                'similar_table_idx': seg_j.get('table_idx')
            }
            # 只有按单元格处理时才添加列信息
            if 'col' in seg_j:
                similar_table_info['similar_col'] = seg_j.get('col')
            detail.update(similar_table_info)
        
        return detail
        
    def _collect_common_grammar_errors(self, 
                                     filtered_bid_segments: List[List[Dict[str, Any]]],
                                     bid_file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        收集多份文件中相同的语法错误
        """
        grammar_error_map = {}
        for file_idx, segs in enumerate(filtered_bid_segments):
            for seg in segs:
                for err in seg.get('grammar_errors', []):
                    key = (err, seg['text'])
                    if key not in grammar_error_map:
                        grammar_error_map[key] = []
                    grammar_error_map[key].append({
                        'bid_file': os.path.basename(bid_file_paths[file_idx]),
                        'page': seg['page'],
                        'text': seg['text']
                    })
        
        # 只保留出现在多份文件的相同语法错误
        common_grammar_errors = [
            {'error': k[0], 'text': k[1], 'locations': v}
            for k, v in grammar_error_map.items() if len(v) > 1
        ]
        
        return common_grammar_errors
        
    def _generate_result(self, task_id: str, details: List[Dict[str, Any]],
                       common_grammar_errors: List[Dict[str, Any]],
                       filtered_bid_segments: List[List[Dict[str, Any]]],
                       bid_file_paths: List[str], start_time: float) -> None:
        """
        生成分析结果并更新任务状态
        """
        # 统计文本和表格相似度数量
        text_similarity_count = sum(1 for d in details if not d.get('is_table_element', False))
        table_similarity_count = sum(1 for d in details if d.get('is_table_element', False))
        
        summary = f'分析完成，发现{text_similarity_count}处文本高相似度片段，{table_similarity_count}处表格单元格高相似度，{len(common_grammar_errors)}组相同语法错误。'
        
        # 添加更多结果指标
        total_bid_files = len(bid_file_paths)
        total_segments_processed = sum(len(segs) for segs in filtered_bid_segments)
        avg_similarity_score = float(f'{np.mean([d["similarity"] for d in details]):.4f}') if details else 0.0
        max_similarity_score = float(f'{max([d["similarity"] for d in details]):.4f}') if details else 0.0
        
        self.tasks[task_id]["status"] = "done"
        self.tasks[task_id]["result"] = {
            "summary": summary, 
            "details": details, 
            "grammar_errors": common_grammar_errors,
            "text_similarity_count": text_similarity_count,
            "table_similarity_count": table_similarity_count,
            "total_bid_files": total_bid_files,
            "total_segments_processed": total_segments_processed,
            "avg_similarity_score": avg_similarity_score,
            "max_similarity_score": max_similarity_score,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        elapsed = time.time() - start_time
        self._log_resource('TASK_DONE', {'task_id': task_id, 'elapsed': f'{elapsed:.1f}s'})

    def get_result(self, task_id: str) -> Dict[str, Any]:
        """
        查询指定任务的结果。
        """
        if task_id not in self.tasks:
            return {"status": "not_found", "result": None}
        return self.tasks[task_id]

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        获取所有任务的简要信息列表。
        """
        return [
            {
                "task_id": task_id,
                "status": info["status"],
                "created_time": info["created_time"],
                "file_info": info["file_info"],
                "progress": info.get("progress", {"current": 0, "total": 1})
            }
            for task_id, info in self.tasks.items()
        ]

    def cancel_task(self, task_id: str) -> bool:
        """
        取消队列中的待处理任务。
        """
        if task_id in self.tasks and self.tasks[task_id]["status"] == "pending":
            self.tasks[task_id]["status"] = "cancelled"
            # 从队列中移除
            self.task_queue = [(tid, tender, bids) for tid, tender, bids in self.task_queue if tid != task_id]
            return True
        return False

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> None:
        """
        清理超过指定时长的历史任务。
        """
        current_time = time.time()
        expired_tasks = [
            task_id for task_id, info in self.tasks.items()
            if current_time - info["created_time"] > max_age_hours * 3600
        ]
        for task_id in expired_tasks:
            del self.tasks[task_id]
