"""
招标文档相似度分析系统主程序
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from app.router import ocr, upload, extract, analyze, config, entity, health, highlight
from app.config.log_config import log_config
from app.lifespan import lifespan

# 初始化日志
log_config.setup_logging()

# 创建FastAPI应用
# 配置Swagger UI使用本地资源，避免CDN不稳定问题
app = FastAPI(
    title="招标文档相似度分析系统 API",
    version="1.0.0",
    # 使用本地Swagger UI资源，提高稳定性
    swagger_ui_parameters={
        "persistAuthorization": True,
    },
    # 生命周期管理：在启动和关闭时执行初始化/清理
    lifespan=lifespan,
    # 可选：如果需要完全离线，可以禁用Swagger UI的CDN
    # 但需要手动下载Swagger UI资源到static目录
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

# 注册路由（统一的新API结构）
app.include_router(health.router)  # 健康检查（最先注册，用于监控）
app.include_router(upload.router)  # 文件上传
app.include_router(extract.router)  # 文本提取（统一提取接口，支持OCR和pdfplumber）
app.include_router(analyze.router)  # 相似度分析（新版统一接口）
app.include_router(config.router)  # 配置管理
app.include_router(entity.router)  # 实体识别
app.include_router(highlight.router)  # 高亮标记
app.include_router(ocr.router)  # OCR服务状态检查（仅保留状态接口）

@app.get("/")
def root():
    """首页重定向到前端页面"""
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run("app.main:app", host="0.0.0.0", port=8020, reload=True) # 开发reload方便调试

    uvicorn.run("app.main:app", host="0.0.0.0", port=8020,reload=False, workers=1)