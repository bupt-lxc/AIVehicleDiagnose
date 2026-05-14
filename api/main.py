from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from api.routes import diagnostic, health
from app.config import Settings

def create_app(settings: Settings) -> FastAPI:
    """创建并配置 FastAPI 应用实例"""
    app = FastAPI(
        title="AutoDiagnose API",
        description="车辆电检报告智能诊断平台",
        version="2.0.0",
    )
    
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    app.add_middleware(
        CORSMiddleware, 
        allow_origins=ALLOWED_ORIGINS, 
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )
    
    # 注册路由
    app.include_router(health.router)
    app.include_router(diagnostic.router)
    
    return app
