# 应用程序主模块
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import Config
from transformers import AutoTokenizer, AutoModel
import torch

# 初始化FastAPI应用
app = FastAPI(
    title="文档相似度检测 API",
    description="提供文档上传和相似度检测功能",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型和分词器加载
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.LOCAL_MODEL_PATH)
    model = AutoModel.from_pretrained(Config.LOCAL_MODEL_PATH)

# 初始化时加载模型
load_model()

__all__ = ["app"]