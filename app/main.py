from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from app.router import similarity
import os

app = FastAPI(title="Bidding Docs Similarity System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源请求（可以根据需求修改为指定域名列表）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有请求方法（POST, GET, PUT, DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)

# 挂载静态文件（前端页面）
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 挂载 similarity 路由
app.include_router(similarity.router)

# 首页路由，返回前端页面
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)