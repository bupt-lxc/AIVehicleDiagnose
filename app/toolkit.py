from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

def build_diagnostic_tool(graph):
    # Avoid circular import
    from .graph_pipeline import DiagnosticGraph
    
    @tool("rag_diagnostic_tool", description="Use RAG context and LLM to generate a diagnostic analysis")
    def rag_diagnostic_tool(question: str, top_k: int = 5, collection: str = "") -> str:
        # Check if graph is of correct type if needed, but for now just trust duck typing or explicit hint
        collection_name = collection.strip() or None
        result: dict[str, Any] = graph.run(question=question, top_k=top_k, collection=collection_name)
        logger.info("Tool called: top_k=%s collection=%s", top_k, collection_name)
        return json.dumps(result, ensure_ascii=False, indent=2)

    return rag_diagnostic_tool
