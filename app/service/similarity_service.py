import os
import uuid
import time
from typing import List, Dict, Any, Optional
from threading import Thread, Lock
import numpy as np
from sentence_transformers import SentenceTransformer
from app.service.text_utils import extract_text_from_pdf, extract_text_from_docx, split_text_to_segments, remove_stopwords, detect_grammar_errors, is_order_changed, is_stopword_evade  # type: ignore
import faiss
import gc
import torch
import logging
import psutil
import threading

if __name__ == "__main__":
    # 配置日志格式
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# =============================
# 相似度分析服务主类
# =============================
class SimilarityService:
    def __init__(self, storage_dir: str = "tmp_files", max_concurrent_tasks: int = 3):
        """
        初始化服务，包括：
        - 创建存储目录
        - 加载本地文本向量模型
        - 加载停用词表
        - 初始化任务队列和并发控制
        """
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.tasks = {}  # 任务ID -> {status, result, progress, created_time, file_info}
        self.model = self._load_text2vec_model()
        self.stopwords = self._load_stopwords()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running_tasks = 0
        self.task_lock = Lock()
        self.task_queue = []

    def _load_text2vec_model(self):
        """
        加载本地 text2vec 语义向量模型，优先使用 GPU。
        """
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../local_text2vec_model'))
        model = SentenceTransformer(model_path)
        if torch.cuda.is_available():
            model = model.to('cuda')
        return model

    def _encode_texts(self, texts: List[str]):
        """
        对文本列表进行向量化。
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def _load_stopwords(self):
        """
        加载停用词表。
        """
        stopwords_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../stopwords.txt'))
        if os.path.exists(stopwords_path):
            with open(stopwords_path, encoding='utf-8') as f:
                return set([line.strip() for line in f if line.strip()])
        return set()

    def save_file(self, file_bytes, filename: str) -> str:
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

    def _process_queue(self):
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

    def _extract_and_segment(self, file_path: str) -> List[Dict]:
        """
        文本提取与分段：支持 PDF、Word 文档。
        每段去除停用词并检测语法错误。
        返回：[{page, text, grammar_errors}]
        """
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            pages = extract_text_from_pdf(file_path)
            segments = []
            for i, page_text in enumerate(pages):
                for seg in split_text_to_segments(page_text):
                    clean_seg = remove_stopwords(seg, list(self.stopwords))
                    if clean_seg:
                        grammar_errors = detect_grammar_errors(clean_seg)
                        segments.append({'page': i+1, 'text': clean_seg, 'grammar_errors': grammar_errors})
            return segments
        elif ext in ['.doc', '.docx']:
            paras = extract_text_from_docx(file_path)
            segments = []
            for i, para in enumerate(paras):
                for seg in split_text_to_segments(para):
                    clean_seg = remove_stopwords(seg, list(self.stopwords))
                    if clean_seg:
                        grammar_errors = detect_grammar_errors(clean_seg)
                        segments.append({'page': i+1, 'text': clean_seg, 'grammar_errors': grammar_errors})
            return segments
        else:
            return []
    
    # cpu only
    def _faiss_max_sim(self, query_vecs, base_vecs):
       """
       使用 faiss 进行最大相似度检索（CPU 版本）。
       """
       if len(base_vecs) == 0 or len(query_vecs) == 0:
           return [], []
       dim = base_vecs.shape[1]
       index = faiss.IndexFlatIP(dim)
       faiss.normalize_L2(base_vecs)
       faiss.normalize_L2(query_vecs)
       index.add(base_vecs)  # type: ignore
       sims, idxs = index.search(query_vecs, 1)  # type: ignore
       return sims.flatten(), idxs.flatten()
    
    # gpu支持
    # def _faiss_max_sim(self, query_vecs, base_vecs):
    #     """
    #     使用 faiss 进行最大相似度检索（支持 GPU）。
    #     """
    #     if len(base_vecs) == 0 or len(query_vecs) == 0:
    #         return [], []
    #     dim = base_vecs.shape[1]
    #     try:
    #         # 尝试用 GPU
    #         import faiss
    #         if hasattr(faiss, 'StandardGpuResources'):
    #             res = faiss.StandardGpuResources()
    #             index = faiss.GpuIndexFlatIP(res, dim)
    #         else:
    #             raise ImportError
    #     except (ImportError, AttributeError):
    #         # 回退到 CPU
    #         import faiss
    #         index = faiss.IndexFlatIP(dim)
    #     faiss.normalize_L2(base_vecs)
    #     faiss.normalize_L2(query_vecs)
    #     index.add(base_vecs)  # type: ignore
    #     sims, idxs = index.search(query_vecs, 1)  # type: ignore
    #     return sims.flatten(), idxs.flatten()

    def _batch_encode(self, texts: List[str], batch_size: int = 1000):
        """
        分批向量化文本，防止内存溢出（OOM）。
        """
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vecs = self._encode_texts(batch)
            all_vecs.append(vecs)
        return np.vstack(all_vecs) if all_vecs else np.zeros((0, 384))

    def _log_resource(self, tag: str, extra: Optional[dict] = None):
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

    def _analyze_task(self, task_id: str, tender_file_path: str, bid_file_paths: List[str], timeout_sec: int = 1200):
        """
        任务主流程：
        1. 招标文件分段并向量化
        2. 投标文件分段并向量化
        3. 剔除与招标文件高度相似的投标片段
        4. 投标文件间两两比对，检测相似片段及规避行为
        5. 统计语法错误，找出多份文件中相同错误
        支持超时自动终止。
        """
        start_time = time.time()
        self._log_resource('TASK_START', {'task_id': task_id})
        timeout_flag = {'timeout': False}
        def timeout_handler():
            timeout_flag['timeout'] = True
            self.tasks[task_id]["status"] = "error"
            self.tasks[task_id]["result"] = {"error": "任务超时自动终止"}
            self._log_resource('TASK_TIMEOUT', {'task_id': task_id})
        timer = threading.Timer(timeout_sec, timeout_handler)
        timer.start()
        try:
            self.tasks[task_id]["status"] = "running"
            # 1. 招标文件分段并向量化（分批）
            tender_segments = self._extract_and_segment(tender_file_path)
            tender_texts = [seg['text'] for seg in tender_segments]
            tender_vecs = self._batch_encode(tender_texts) if tender_texts else np.zeros((1, 384))
            if isinstance(tender_vecs, list):
                tender_vecs = np.array(tender_vecs)

            # 2. 投标文件分段并向量化（分批）
            bid_segments_list = []
            bid_vecs_list = []
            total_segments = 0
            for bid_path in bid_file_paths:
                segs = self._extract_and_segment(bid_path)
                texts = [s['text'] for s in segs]
                vecs = self._batch_encode(texts) if texts else np.zeros((1, 384))
                if isinstance(vecs, list):
                    vecs = np.array(vecs)
                bid_segments_list.append(segs)
                bid_vecs_list.append(vecs)
                total_segments += len(segs)
                del segs, texts, vecs
                gc.collect()
            self.tasks[task_id]["progress"] = {"current": 0, "total": total_segments}

            # 3. 剔除投标文件中与招标文件高度相似的片段（faiss加速，分批）
            filtered_bid_segments = []
            filtered_bid_vecs = []
            for segs, vecs in zip(bid_segments_list, bid_vecs_list):
                if isinstance(vecs, list):
                    vecs = np.array(vecs)
                keep_idx = []
                batch_size = 1000
                for i in range(0, len(segs), batch_size):
                    batch_vecs = vecs[i:i+batch_size]
                    if tender_vecs.shape[0] > 0 and batch_vecs.shape[0] > 0:
                        sims, _ = self._faiss_max_sim(batch_vecs, tender_vecs)
                        # 只保留与招标文件相似度低于0.8的片段
                        keep_idx.extend([i+ii for ii, sim in enumerate(sims) if sim < 0.8])
                    else:
                        keep_idx.extend(list(range(i, min(i+batch_size, len(segs)))))
                filtered_bid_segments.append([segs[i] for i in keep_idx])
                filtered_bid_vecs.append(vecs[keep_idx] if len(keep_idx) > 0 else np.zeros((1, 384)))
                del segs, vecs
                gc.collect()

            # 4. 投标文件间两两比对，faiss加速，分批
            details = []
            progress_cnt = 0
            for i in range(len(filtered_bid_segments)):
                for j in range(i+1, len(filtered_bid_segments)):
                    segs_i, vecs_i = filtered_bid_segments[i], filtered_bid_vecs[i]
                    segs_j, vecs_j = filtered_bid_segments[j], filtered_bid_vecs[j]
                    if isinstance(vecs_i, list):
                        vecs_i = np.array(vecs_i)
                    if isinstance(vecs_j, list):
                        vecs_j = np.array(vecs_j)
                    if vecs_i.shape[0] == 0 or vecs_j.shape[0] == 0:
                        continue
                    batch_size = 1000
                    for k in range(0, vecs_i.shape[0], batch_size):
                        batch_vecs = vecs_i[k:k+batch_size]
                        sims, idxs = self._faiss_max_sim(batch_vecs, vecs_j)
                        for idx_b, (sim, max_idx) in enumerate(zip(sims, idxs)):
                            idx_i = k + idx_b
                            if sim > 0.85:
                                text1 = segs_i[idx_i]['text']
                                text2 = segs_j[max_idx]['text']
                                # 检测规避行为
                                order_changed = is_order_changed(text1, text2)
                                stopword_evade = is_stopword_evade(text1, text2, list(self.stopwords))
                                details.append({
                                    'bid_file': os.path.basename(bid_file_paths[i]),
                                    'page': segs_i[idx_i]['page'],
                                    'text': text1,
                                    'grammar_errors': segs_i[idx_i].get('grammar_errors', []),
                                    'similar_with': os.path.basename(bid_file_paths[j]),
                                    'similar_page': segs_j[max_idx]['page'],
                                    'similarity': float(f'{sim:.4f}'),
                                    'similar_text': text2,
                                    'similar_grammar_errors': segs_j[max_idx].get('grammar_errors', []),
                                    'order_changed': order_changed,
                                    'stopword_evade': stopword_evade
                                })
                            progress_cnt += 1
                            self.tasks[task_id]["progress"]["current"] = min(progress_cnt, self.tasks[task_id]["progress"]["total"])
                        del batch_vecs, sims, idxs
                        gc.collect()

            # 5. 统计所有投标文件分段的语法错误，找出多份文件中出现相同语法错误的片段
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

            summary = f'分析完成，发现{len(details)}处高相似度片段，{len(common_grammar_errors)}组相同语法错误。'
            self.tasks[task_id]["status"] = "done"
            self.tasks[task_id]["result"] = {"summary": summary, "details": details, "grammar_errors": common_grammar_errors}
            if timeout_flag['timeout']:
                return
            elapsed = time.time() - start_time
            self._log_resource('TASK_DONE', {'task_id': task_id, 'elapsed': f'{elapsed:.1f}s'})
        except Exception as e:
            self.tasks[task_id]["status"] = "error"
            self.tasks[task_id]["result"] = {"error": str(e)}
            self._log_resource('TASK_ERROR', {'task_id': task_id, 'error': str(e)})
        finally:
            try:
                timer.cancel()
            except Exception:
                pass
            with self.task_lock:
                self.running_tasks -= 1
                self._process_queue()

    def get_result(self, task_id: str) -> Dict[str, Any]:
        """
        查询指定任务的结果。
        """
        if task_id not in self.tasks:
            return {"status": "not_found", "result": None}
        return self.tasks[task_id]

    def get_all_tasks(self) -> List[Dict]:
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

    def cleanup_old_tasks(self, max_age_hours: int = 24):
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
