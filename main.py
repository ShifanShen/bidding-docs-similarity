
# 应用程序入口
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app import app as main_app
from router.upload_router import router as upload_router

# 挂载路由
main_app.include_router(upload_router, prefix="/api")



# 挂载静态文件
main_app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:main_app", host="0.0.0.0", port=8000, reload=True)
    # 惩罚系数
    LENGTH_RATIO_LOW_PENALTY = 0.3
    LONG_DOC_BASE_PENALTY = 0.8