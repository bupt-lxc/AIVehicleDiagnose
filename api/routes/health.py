from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """基础存活状态检查"""
    return {"status": "healthy", "service": "autodiagnose"}


@router.get("/ready")
async def readiness_check():
    """就绪状态检查，包括外部依赖连接检查"""
    # TODO: 检查数据库连接
    # TODO: 检查 LLM 连接
    # TODO: 检查 MCP 连接
    return {"status": "ready"}
