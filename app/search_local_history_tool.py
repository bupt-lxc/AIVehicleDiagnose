from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class SearchLocalHistoryInput(BaseModel):
    query: str = Field(
        ...,
        description="核心检索词。可输入主题词、问题句或关键词组合，例如“budget review”。",
        min_length=1,
    )
    start_date: str | None = Field(
        default=None,
        description="起始日期，格式 YYYY-MM-DD。为空表示不限制开始时间。",
    )
    end_date: str | None = Field(
        default=None,
        description="结束日期，格式 YYYY-MM-DD。为空表示不限制结束时间。",
    )
    mode: Literal["semantic", "keyword", "hybrid"] = Field(
        default="hybrid",
        description="检索模式：semantic=向量检索，keyword=BM25关键词检索，hybrid=混合重排。",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="返回结果数量上限。小问题建议 3-5，开放检索建议 10-20。",
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def _validate_date(cls, value: str | None) -> str | None:
        if value is None or value.strip() == "":
            return None
        datetime.strptime(value, "%Y-%m-%d")
        return value

    @field_validator("query")
    @classmethod
    def _normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query 不能为空")
        return normalized

    @model_validator(mode="after")
    def _validate_date_range(self) -> "SearchLocalHistoryInput":
        if self.start_date and self.end_date:
            start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
            end = datetime.strptime(self.end_date, "%Y-%m-%d").date()
            if start > end:
                raise ValueError("start_date 不能晚于 end_date")
        return self


class PendingAction(BaseModel):
    status: Literal["pending_approval"] = "pending_approval"
    action_type: str = Field(..., description="待确认动作类型，如 delete 或 external_api_call")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="风险等级")
    proposed_args: dict[str, Any] = Field(default_factory=dict, description="拟执行参数")
    message: str = Field(..., description="给人类审阅的执行说明")


@dataclass(slots=True)
class LocalHistorySearchService:
    history_root: Path
    collection_name: str = "conversation_history"

    def search(self, payload: SearchLocalHistoryInput) -> dict[str, Any]:
        try:
            import chromadb
        except Exception:
            return {
                "status": "error",
                "error_type": "dependency_missing",
                "message": "未检测到 chromadb，请先安装依赖后再执行本地历史检索。",
                "results": [],
            }

        chroma_dir = self.history_root / "multimodal" / "chroma"
        if not chroma_dir.exists():
            return {
                "status": "error",
                "error_type": "history_not_initialized",
                "message": f"历史向量库目录不存在: {chroma_dir}",
                "results": [],
            }

        client = None
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collection = client.get_collection(self.collection_name)
        except Exception as exc:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            return {
                "status": "error",
                "error_type": "db_unavailable",
                "message": f"历史向量库不可用或未初始化: {exc}",
                "results": [],
            }

        try:
            semantic_rows = self._semantic_search(collection, payload)
            keyword_rows = self._keyword_search(collection, payload)
            merged = self._merge_and_rank(payload.mode, semantic_rows, keyword_rows)
            limited = merged[: payload.limit]
            return {
                "status": "ok",
                "mode": payload.mode,
                "query": payload.query,
                "date_filter": {"start_date": payload.start_date, "end_date": payload.end_date},
                "total_candidates": len(merged),
                "results": limited,
            }
        except Exception as exc:
            return {
                "status": "error",
                "error_type": "search_failed",
                "message": f"本地历史检索失败: {exc}",
                "results": [],
            }
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _semantic_search(self, collection: Any, payload: SearchLocalHistoryInput) -> list[dict[str, Any]]:
        fetch_k = max(payload.limit * 8, 40)
        rows: list[dict[str, Any]] = []
        try:
            result = collection.query(
                query_texts=[payload.query],
                n_results=fetch_k,
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            return rows
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for idx, item_id in enumerate(ids):
            metadata = metas[idx] if idx < len(metas) else {}
            if not self._match_date(metadata, payload.start_date, payload.end_date):
                continue
            distance = float(dists[idx]) if idx < len(dists) and dists[idx] is not None else 1.0
            score = 1.0 / (1.0 + max(distance, 0.0))
            rows.append(
                {
                    "id": item_id,
                    "document": docs[idx] if idx < len(docs) else "",
                    "metadata": metadata,
                    "semantic_score": score,
                    "keyword_score": 0.0,
                }
            )
        return rows

    def _keyword_search(self, collection: Any, payload: SearchLocalHistoryInput) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            result = collection.get(include=["metadatas", "documents"])
        except Exception:
            return rows
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        corpus: list[dict[str, Any]] = []
        for idx, item_id in enumerate(ids):
            metadata = metas[idx] if idx < len(metas) else {}
            if not self._match_date(metadata, payload.start_date, payload.end_date):
                continue
            corpus.append(
                {
                    "id": item_id,
                    "document": docs[idx] if idx < len(docs) else "",
                    "metadata": metadata,
                }
            )
        scores = self._bm25_scores(payload.query, [item["document"] for item in corpus])
        for idx, item in enumerate(corpus):
            rows.append(
                {
                    "id": item["id"],
                    "document": item["document"],
                    "metadata": item["metadata"],
                    "semantic_score": 0.0,
                    "keyword_score": scores[idx],
                }
            )
        return rows

    def _merge_and_rank(
        self,
        mode: Literal["semantic", "keyword", "hybrid"],
        semantic_rows: list[dict[str, Any]],
        keyword_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if mode == "semantic":
            return self._sort_rows(semantic_rows, mode)
        if mode == "keyword":
            return self._sort_rows(keyword_rows, mode)
        merged: dict[str, dict[str, Any]] = {}
        for item in semantic_rows:
            merged[item["id"]] = dict(item)
        for item in keyword_rows:
            if item["id"] not in merged:
                merged[item["id"]] = dict(item)
            else:
                merged[item["id"]]["keyword_score"] = item["keyword_score"]
        rows = list(merged.values())
        max_semantic = max((item["semantic_score"] for item in rows), default=1.0) or 1.0
        max_keyword = max((item["keyword_score"] for item in rows), default=1.0) or 1.0
        for item in rows:
            semantic_norm = item["semantic_score"] / max_semantic
            keyword_norm = item["keyword_score"] / max_keyword
            item["score"] = 0.65 * semantic_norm + 0.35 * keyword_norm
        rows.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return [self._normalize_result(item, "hybrid") for item in rows]

    def _sort_rows(self, rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
        key_name = "semantic_score" if mode == "semantic" else "keyword_score"
        ordered = sorted(rows, key=lambda x: x.get(key_name, 0.0), reverse=True)
        return [self._normalize_result(item, mode) for item in ordered]

    def _normalize_result(self, row: dict[str, Any], mode: str) -> dict[str, Any]:
        metadata = row.get("metadata", {}) or {}
        return {
            "id": row.get("id", ""),
            "score": float(row.get("score", row.get("semantic_score", row.get("keyword_score", 0.0)))),
            "mode": mode,
            "document": row.get("document", ""),
            "created_at": metadata.get("created_at"),
            "trace_id": metadata.get("trace_id"),
            "conversation_id": metadata.get("conversation_id"),
            "message_id": metadata.get("message_id"),
            "metadata": metadata,
        }

    def _match_date(self, metadata: dict[str, Any], start_date: str | None, end_date: str | None) -> bool:
        if not start_date and not end_date:
            return True
        created_at = str(metadata.get("created_at", "")).strip()
        if not created_at:
            return False
        created_day = self._safe_parse_day(created_at)
        if created_day is None:
            return False
        start_day = date.min if not start_date else datetime.strptime(start_date, "%Y-%m-%d").date()
        end_day = date.max if not end_date else datetime.strptime(end_date, "%Y-%m-%d").date()
        return start_day <= created_day <= end_day

    def _safe_parse_day(self, raw_value: str) -> date | None:
        candidate = raw_value.strip().replace("Z", "+00:00")
        try:
            if "T" in candidate:
                return datetime.fromisoformat(candidate).date()
            return datetime.strptime(candidate[:10], "%Y-%m-%d").date()
        except ValueError:
            return None

    def _bm25_scores(self, query: str, docs: list[str]) -> list[float]:
        tokenized_docs = [self._tokenize(item) for item in docs]
        query_tokens = self._tokenize(query)
        if not docs or not query_tokens:
            return [0.0 for _ in docs]
        avg_len = sum(len(tokens) for tokens in tokenized_docs) / max(len(tokenized_docs), 1)
        k1 = 1.5
        b = 0.75
        doc_freq: dict[str, int] = {}
        for tokens in tokenized_docs:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        scores: list[float] = []
        total_docs = len(tokenized_docs)
        for tokens in tokenized_docs:
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            length = max(len(tokens), 1)
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if freq == 0:
                    continue
                df = doc_freq.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1 - b + b * (length / max(avg_len, 1)))
                score += idf * ((freq * (k1 + 1)) / max(denom, 1e-9))
            scores.append(score)
        return scores

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", text.lower())


def search_local_history(payload: SearchLocalHistoryInput, service: LocalHistorySearchService) -> dict[str, Any]:
    return service.search(payload)


def build_search_local_history_tool(service: LocalHistorySearchService) -> StructuredTool:
    def _runner(
        query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        mode: Literal["semantic", "keyword", "hybrid"] = "hybrid",
        limit: int = 5,
    ) -> dict[str, Any]:
        try:
            payload = SearchLocalHistoryInput(
                query=query,
                start_date=start_date,
                end_date=end_date,
                mode=mode,
                limit=limit,
            )
        except ValidationError as exc:
            return {
                "status": "error",
                "error_type": "invalid_arguments",
                "message": "参数校验失败，请检查 query/date/mode/limit。",
                "details": exc.errors(),
                "results": [],
            }
        return search_local_history(payload, service)

    tool = StructuredTool.from_function(
        func=_runner,
        name="search_local_history",
        description=(
            "检索本地历史对话，支持语义检索、关键词检索、日期范围过滤与混合排序。"
            "优先用于查找历史问答、trace 相关上下文和会话证据。"
        ),
        args_schema=SearchLocalHistoryInput,
        return_direct=False,
    )
    return tool


def build_default_search_local_history_tool(
    history_root: str | Path = Path("logs/history"),
    collection_name: str = "conversation_history",
) -> StructuredTool:
    root_path = Path(history_root)
    service = LocalHistorySearchService(history_root=root_path, collection_name=collection_name)
    return build_search_local_history_tool(service)


def get_tool_definition() -> dict[str, Any]:
    # 如果未来演变为删除类工具，不直接执行副作用；应先返回 pending_approval，由图状态机驱动 HITL。
    # 例如删除 c:\Users\ADMIN\Desktop\AI Agent Projects\AI Disgnostic\file2delete 时，先返回待审批动作。
    return {
        "name": "search_local_history",
        "description": (
            "Search local conversation history from ChromaDB with semantic/keyword/hybrid ranking and optional date filters."
        ),
        "schema": SearchLocalHistoryInput.model_json_schema(),
        "hitl": {
            "read_only": True,
            "requires_approval": False,
            "future_write_protocol": {
                "status": "pending_approval",
                "action": "delete",
                "example_path": r"c:\Users\ADMIN\Desktop\AI Agent Projects\AI Disgnostic\file2delete",
                "approve_action": "execute side effect",
                "modify_action": "resubmit with updated args",
                "reject_action": "abort flow",
            },
        },
    }
