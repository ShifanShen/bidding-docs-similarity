"""
招标文档相似度分析系统主程序
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from app.router import similarity, ocr, highlight
from app.config.log_config import log_config

# 初始化日志
log_config.setup_logging()

# 创建FastAPI应用
app = FastAPI(
    title="招标文档相似度分析系统",
    description="基于AI的招标文档相似度检测和规避行为分析",
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

# 注册路由
app.include_router(similarity.router)
app.include_router(ocr.router)
app.include_router(highlight.router)

@app.get("/")
def root():
    """首页重定向到前端页面"""
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)