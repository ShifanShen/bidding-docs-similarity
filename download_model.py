# from sentence_transformers import SentenceTransformer
#
# # 定义模型名称和保存目录
# model_name = 'shibing624/text2vec-base-chinese'
# save_directory = 'local_text2vec_model'
#
# # 下载并保存模型
# model = SentenceTransformer(model_name)
# model.save(save_directory)
#
# print(f"模型已成功下载并保存到 {save_directory} 目录。")

import os
import warnings
from huggingface_hub import snapshot_download, HfApi
from sentence_transformers import SentenceTransformer

# ====================== 1. 强制配置国内镜像（核心修复） ======================
# 方案A：直接修改huggingface_hub的默认端点（优先级最高）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 方案B：手动指定API的基础URL
HfApi().endpoint = 'https://hf-mirror.com'

# 忽略无关警告
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ====================== 2. 先手动下载模型文件到本地，再加载 ======================
model_name = 'shibing624/text2vec-base-chinese'
save_directory = 'local_text2vec_model'

try:
    # 步骤1：使用huggingface-hub的snapshot_download下载模型（强制用镜像）
    print(f"正在从国内镜像下载模型 {model_name}...")
    model_dir = snapshot_download(
        repo_id=model_name,
        repo_type='model',
        endpoint='https://hf-mirror.com'  # 显式指定镜像端点
    )

    # 步骤2：加载模型并保存到指定目录
    model = SentenceTransformer(model_dir)
    model.save(save_directory)
    print(f"✅ 模型已成功下载并保存到 {save_directory} 目录。")

except Exception as e:
    print(f"❌ 下载失败：{str(e)[:200]}")
    print(f"🔍 尝试加载本地目录 {save_directory} 中的模型...")
    try:
        model = SentenceTransformer(save_directory)
        print(f"✅ 成功加载本地模型 {save_directory}！")
    except Exception as e2:
        print(f"❌ 本地加载失败：{str(e2)[:200]}")
        print("\n💡 请执行解决方案2：手动下载模型。")