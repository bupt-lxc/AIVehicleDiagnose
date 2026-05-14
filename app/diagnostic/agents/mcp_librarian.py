from __future__ import annotations

import asyncio
from loguru import logger

from app.diagnostic.state import DiagnosticState, RAGKnowledge
from app.rag_client import RAGClient


class MCPLibrarianAgent:
    NODE_NAME = "mcp_librarian"
    
    def __init__(self, rag_client: RAGClient, config: dict | None = None):
        self._rag_client = rag_client
        self._dtc_top_k = config.get("dtc_top_k", 3) if config else 3
        self._case_top_k = config.get("case_top_k", 5) if config else 5
        logger.info("[MCPLibrarianAgent] 初始化完成")
    
    async def retrieve(self, state: DiagnosticState) -> dict:
        structured_data = state.get("structured_data", {})
        focused_dtcs = state.get("focused_dtcs", []) or structured_data.get("dtc_codes", [])
        dtc_codes = [d["code"] for d in focused_dtcs if d.get("code")]
        symptoms = self._build_symptom_query(state)
        
        if not dtc_codes and not symptoms:
            logger.warning("[MCPLibrarianAgent] 无 DTC 和症状信息，跳过知识检索")
            return {
                "rag_knowledge": RAGKnowledge(
                    dtc_definitions=[],
                    case_studies=[],
                    repair_procedures=[]
                ),
                "current_node": self.NODE_NAME,
            }
            
        try:
            dtc_definitions, case_studies = await asyncio.gather(
                self._get_dtc_definitions(dtc_codes),
                self._get_case_studies(symptoms),
            )
            
            rag_knowledge = RAGKnowledge(
                dtc_definitions=dtc_definitions,
                case_studies=case_studies,
                repair_procedures=self._extract_repair_procedures(case_studies),
            )
            
            logger.info(f"[MCPLibrarianAgent] 知识检索完成: {len(dtc_definitions)} 条定义, {len(case_studies)} 个案例")
            knowledge_degraded = len(dtc_codes) > 0 and len(dtc_definitions) == 0
            return {
                "rag_knowledge": rag_knowledge,
                "current_node": self.NODE_NAME,
                "knowledge_degraded": knowledge_degraded,
                "hitl_required": bool(knowledge_degraded and dtc_codes),
                "current_hitl_point": "knowledge_retrieval" if knowledge_degraded and dtc_codes else state.get("current_hitl_point"),
                "hitl_question": "知识召回不足，请补充故障现象/优先ECU以提升检索质量。" if knowledge_degraded and dtc_codes else state.get("hitl_question", ""),
                "hitl_options": ["A", "D"] if knowledge_degraded and dtc_codes else state.get("hitl_options", []),
            }
        except Exception as e:
            logger.error(f"[MCPLibrarianAgent] 知识检索失败: {e}")
            return {
                "rag_knowledge": RAGKnowledge(
                    dtc_definitions=[],
                    case_studies=[],
                    repair_procedures=[]
                ),
                "current_node": self.NODE_NAME,
                "error_message": f"知识检索失败: {str(e)}",
                "risk_flags": state.get("risk_flags", []) + ["knowledge_retrieval_failed"],
                "knowledge_degraded": True
            }
    
    def _build_symptom_query(self, state: DiagnosticState) -> str:
        parts = []
        
        if state.get("user_context"):
            parts.append(state["user_context"])
            
        if state.get("user_description"):
            parts.append(state["user_description"])
            
        structured_data = state.get("structured_data", {})
        for dtc in structured_data.get("dtc_codes", []):
            if dtc.get("description"):
                parts.append(f"DTC {dtc.get('code')}: {dtc.get('description')}")
                
        return " ".join(parts)
        
    async def _get_dtc_definitions(self, dtc_codes: list[str]) -> list[dict]:
        definitions = []
        for code in dtc_codes:
            query_str = f"What is the definition and common causes for DTC {code}?"
            result = await self._rag_client.query(
                question=query_str,
                top_k=self._dtc_top_k,
                collection="dtc_definitions"
            )
            
            if not result.get("is_error"):
                definitions.append({
                    "code": code,
                    "definition": result.get("content", ""),
                    "source": "mcp_knowledge_hub"
                })
        return definitions
        
    async def _get_case_studies(self, symptoms: str) -> list[dict]:
        if not symptoms:
            return []
            
        query_str = f"Find similar diagnostic case studies for these symptoms: {symptoms}"
        result = await self._rag_client.query(
            question=query_str,
            top_k=self._case_top_k,
            collection="case_studies"
        )
        
        if not result.get("is_error"):
            # Simple wrapper since we just have string content from the tool
            return [{
                "symptoms_query": symptoms,
                "content": result.get("content", ""),
                "source": "mcp_knowledge_hub"
            }]
        return []
        
    def _extract_repair_procedures(self, case_studies: list[dict]) -> list[str]:
        # For a more advanced implementation, this could use an LLM to extract procedures.
        # For now, we return the raw content if it looks like procedures are included.
        procedures = []
        for case in case_studies:
            content = case.get("content", "")
            if "repair" in content.lower() or "procedure" in content.lower() or "fix" in content.lower():
                procedures.append(content)
        return procedures
