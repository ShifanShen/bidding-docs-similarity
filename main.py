from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
import tempfile
from docx import Document
import PyPDF2
import fitz  # PyMuPDF
import shutil
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn



# 加载预训练的 BERT 模型和分词器
# 使用本地BERT模型
local_model_path = "./local_bert_model"
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
model = AutoModel.from_pretrained(local_model_path, local_files_only=True)

# 初始化 FastAPI 应用
app = FastAPI(
    title="文档相似度检测 API",
    description="提供文档上传和相似度检测功能",
    version="1.0.0",
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 创建临时目录
TEMP_DIR = Path(tempfile.gettempdir()) / "bidding_docs"
TEMP_DIR.mkdir(exist_ok=True)

# 文档解析函数
async def extract_text_from_file(file_path: str) -> str:
    """从不同类型的文件中提取文本内容"""
    if file_path.endswith('.docx'):
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif file_path.endswith('.pdf'):
        # 使用 PyMuPDF 提取 PDF 文本
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"不支持的文件类型: {file_path.split('.')[-1]}")


# 定义函数来获取文本的嵌入向量
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用 [CLS] 标记的输出作为句子的嵌入
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

# 定义文件上传和相似度检测的 API 端点
@app.post("/upload-and-compare")
async def upload_and_compare(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # 保存上传的文件到临时目录
    file1_path = TEMP_DIR / file1.filename
    file2_path = TEMP_DIR / file2.filename

    with open(file1_path, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
    with open(file2_path, "wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)

    # 提取文本内容
    text1 = await extract_text_from_file(str(file1_path))
    text2 = await extract_text_from_file(str(file2_path))

    # 获取文本嵌入
    embeddings1 = get_embeddings(text1)
    embeddings2 = get_embeddings(text2)

    # 计算余弦相似度
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]

    # 确保相似度值是 Python 内置的 float 类型
    similarity = float(similarity)

    # 清理临时文件
    os.remove(file1_path)
    os.remove(file2_path)

    return {"similarity": similarity}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)