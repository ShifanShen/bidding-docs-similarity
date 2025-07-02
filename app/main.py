from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from app.router import similarity
import os

app = FastAPI(title="Bidding Docs Similarity System")

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