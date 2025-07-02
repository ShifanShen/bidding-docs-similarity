# 投标文件相似度检测系统

本项目基于 FastAPI + 本地 text2vec 语义模型，支持大体量招标/投标文件的智能雷同分析，自动检测语义雷同、规避行为（如语序调整、无意义词插入）、语法错误等，支持结果导出为 JSON/Excel。

---

## 主要功能

- 支持上传招标文件和多份投标文件（PDF/Word）
- 自动分段、去除无意义词、语法错误检测
- 剔除投标文件中参考/摘抄招标文件内容
- 投标文件间语义雷同分析，定位页码、内容、相似文件、相似度等
- 检测规避行为（如语序调整、无意义词插入）
- 检测多份文件中相同语法错误
- 前端进度条、动态动画
- 分析结果一键导出为 JSON/Excel

---

## 安装与运行

### 1. 安装依赖

建议使用 Python 3.8+，并提前创建虚拟环境：

```bash
pip install -r requirements.txt
```

### 2. 下载/准备本地 text2vec 模型

将本地模型（如 HuggingFace sentence-transformers 格式）放在 `local_text2vec_model/` 目录下。

### 3. 启动服务

```bash
python -m app.main
```

默认监听 `http://localhost:8000/`，API 文档见 `http://localhost:8000/docs`

---

## 使用说明

1. 访问 `http://localhost:8000/`，进入前端页面。
2. 上传招标文件和多份投标文件（支持 PDF/Word，单个文件可达千页）。
3. 点击“开始分析”，系统自动分析并展示进度。
4. 分析完成后，页面展示雷同片段、语法错误、规避行为等详细结果。
5. 可一键导出分析结果为 JSON 或 Excel。


## 接口文档

1. 访问 `http://localhost:8000/docs` 进入api文档

---

## 主要依赖

- fastapi
- uvicorn
- python-multipart
- pdfplumber
- python-docx
- sentence-transformers
- torch
- numpy
- language_tool_python
- openpyxl

---

## 目录结构

```
bidding-docs-similarity/
├─ app/
│   ├─ main.py              # FastAPI 启动入口
│   ├─ router/
│   │    └─ similarity.py   # 路由
│   ├─ service/
│   │    ├─ similarity_service.py  # 业务逻辑
│   │    └─ text_utils.py   # 文本处理工具
│   └─ static/
│        └─ index.html      # 前端页面
├─ local_text2vec_model/    # 本地模型目录
├─ requirements.txt
├─ stopwords.txt            # 无意义词表
└─ README.md
```

---

## 常见问题

- **模型加载报错**：请确保 `sentence-transformers` 版本与模型兼容，建议 `pip install -U sentence-transformers`。
- **大文件性能**：系统已做分块与进度反馈，建议服务器内存充足。
- **/docs 无法访问**：请确认服务已正常启动且无端口冲突。

---

## 联系与支持

如有问题或建议，欢迎 issue 或联系作者。
