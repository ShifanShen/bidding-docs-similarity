"""
招标文档相似度分析系统主程序
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from app.router import similarity, ocr, upload, extract, analyze, config
from app.config.log_config import log_config

# 初始化日志
log_config.setup_logging()

# 创建FastAPI应用
app = FastAPI(
    title="招标文档相似度分析系统 API",
    version="1.0.0"
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 注册路由（新API结构）
app.include_router(upload.router)  # 文件上传
app.include_router(extract.router)  # 文本提取
app.include_router(analyze.router)  # 相似度分析
app.include_router(config.router)  # 配置管理

# 保留旧路由以保持向后兼容
app.include_router(similarity.router)  # 旧版相似度分析API
app.include_router(ocr.router)  # 旧版OCR API

@app.get("/")
def root():
    """首页重定向到前端页面"""
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    import uvicorn
    # 支持通过环境变量配置端口，默认8001
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)