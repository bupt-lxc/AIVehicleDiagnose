import uuid
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from app.diagnostic.state import FinalReport, create_initial_state, DiagnosticState
from app.diagnostic.graph import AutoDiagnoseGraph
from app.diagnostic.hitl.handler import HITLHandler
import asyncio
from loguru import logger

REQUEST_TIMEOUT_SECONDS = 120

# FUNC-001: 任务状态追踪
_task_status = {}

router = APIRouter(prefix="/api/v1/diagnostic", tags=["diagnostic"])


class DiagnosticRequest(BaseModel):
    """诊断请求模型"""
    raw_input: str = Field(..., description="原始电检报告文本")
    user_context: str = Field(default="", description="用户补充上下文")
    thread_id: str | None = Field(default=None, description="会话ID，用于多轮对话")
    config: dict | None = Field(default=None, description="可选配置，如LLM profile选择等")


class DiagnosticResponse(BaseModel):
    """诊断响应模型"""
    thread_id: str = Field(..., description="当前会话ID")
    run_id: str = Field(..., description="当前运行实例ID")
    status: Literal["running", "completed", "failed", "pending_hitl"] = Field(
        ..., description="诊断任务状态"
    )
    final_report: FinalReport | None = Field(default=None, description="最终诊断报告")
    hitl_required: bool = Field(default=False, description="是否需要人工介入")
    hitl_question: str | None = Field(default=None, description="人工介入问题内容")
    hitl_options: list[str] | None = Field(default=None, description="人工介入可选项")
    operator_trace: list[dict] | None = Field(default=None, description="最近可展示的分析轨迹")


class HITLSubmission(BaseModel):
    """HITL 提交模型"""
    choice: str = Field(..., description="用户选择")
    user_input: str | None = Field(default=None, description="用户额外输入")


def _build_response(state: dict) -> DiagnosticResponse:
    status = "running"
    
    if state.get("hitl_required"):
        status = "pending_hitl"
    elif state.get("final_report") and state.get("report_approved"):
         status = "completed"
    elif state.get("error_message") or state.get("exit_code", 0) != 0:
         status = "failed"
    
    return DiagnosticResponse(
        thread_id=state.get("thread_id", ""),
        run_id=state.get("run_id", ""),
        status=status,
        final_report=state.get("final_report"),
        hitl_required=state.get("hitl_required", False),
        hitl_question=state.get("hitl_question"),
        hitl_options=state.get("hitl_options"),
        operator_trace=list(state.get("operator_trace", []))[-20:],
    )


async def _run_diagnosis_background(thread_id: str, request: DiagnosticRequest):
    _task_status[thread_id] = {"status": "running", "progress": 0, "run_id": ""}
    
    initial_state = create_initial_state(
        raw_input=request.raw_input,
        user_context=request.user_context,
        thread_id=thread_id
    )
    _task_status[thread_id]["run_id"] = initial_state.get("run_id", "")
    
    async with AutoDiagnoseGraph() as graph_runner:
        try:
            try:
                async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                    final_state = await graph_runner.run(initial_state)
                    
                _task_status[thread_id]["status"] = "completed"
                _task_status[thread_id]["progress"] = 100
                _task_status[thread_id]["final_state"] = final_state
                
            except asyncio.TimeoutError:
                logger.error(f"Diagnostic request timed out for thread {thread_id}")
                _task_status[thread_id]["status"] = "failed"
                _task_status[thread_id]["error"] = f"Request exceeded {REQUEST_TIMEOUT_SECONDS}s timeout"
        except Exception as e:
            logger.error(f"Diagnostic failed for thread {thread_id}: {str(e)}", exc_info=True)
            _task_status[thread_id]["status"] = "failed"
            _task_status[thread_id]["error"] = str(e)

@router.post("/analyze", response_model=DiagnosticResponse)
async def analyze(request: Request, body: DiagnosticRequest, background_tasks: BackgroundTasks):
    """启动诊断分析任务"""
    thread_id = body.thread_id or f"diag-{uuid.uuid4().hex[:12]}"
    
    background_tasks.add_task(_run_diagnosis_background, thread_id, body)
    
    return DiagnosticResponse(
        thread_id=thread_id,
        run_id="",
        status="running",
        final_report=None,
        hitl_required=False,
        hitl_question=None,
        hitl_options=None,
        operator_trace=None,
    )


@router.post("/hitl/{thread_id}", response_model=DiagnosticResponse)
async def submit_hitl(thread_id: str, submission: HITLSubmission):
    """提交人工干预 (HITL) 选择并继续诊断"""
    async with AutoDiagnoseGraph() as graph_runner:
        try:
            # 1. Get current state
            config = {"configurable": {"thread_id": thread_id}}
            current_state_snapshot = await graph_runner.graph.aget_state(config)
            
            if not current_state_snapshot.values:
                raise HTTPException(status_code=404, detail="Session not found or expired")
                
            current_state = current_state_snapshot.values
            
            # 2. Process feedback
            hitl_handler = HITLHandler()
            
            updated_state = hitl_handler.process_feedback(
                current_state,
                submission.choice,
                submission.user_input
            )
            
            # 3. Update state in graph
            await graph_runner.graph.aupdate_state(config, updated_state)
            
            # 4. Resume execution
            # Invoke with None input to resume from current state
            final_state = await graph_runner.graph.ainvoke(None, config)
            
            return _build_response(final_state)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to submit HITL feedback for thread {thread_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error while processing feedback")


@router.get("/status/{thread_id}", response_model=DiagnosticResponse)
async def get_status(thread_id: str):
    """获取诊断任务当前状态"""
    async with AutoDiagnoseGraph() as graph_runner:
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state_snapshot = await graph_runner.graph.aget_state(config)
            
            if not state_snapshot.values:
                if thread_id in _task_status:
                    task = _task_status[thread_id]
                    if task.get("status") == "failed":
                        return DiagnosticResponse(
                            thread_id=thread_id,
                            run_id=task.get("run_id", ""),
                            status="failed",
                            hitl_required=False,
                            hitl_question=task.get("error"),
                            operator_trace=None,
                        )
                    return DiagnosticResponse(
                        thread_id=thread_id,
                        run_id=task.get("run_id", ""),
                        status="running",
                        hitl_required=False,
                        operator_trace=None,
                    )
                raise HTTPException(status_code=404, detail="Session not found")
                
            return _build_response(state_snapshot.values)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get status for thread {thread_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error while fetching status")
