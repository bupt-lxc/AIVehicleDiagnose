from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class CompressionInput:
    previous_summary: str
    old_messages: list[dict[str, str]]
    max_tokens: int


@dataclass(slots=True)
class CompressionOutput:
    summary_text: str
    strategy_name: str
    estimated_tokens: int


class CompressionStrategy(Protocol):
    name: str

    def compress(self, payload: CompressionInput) -> CompressionOutput:
        ...


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


class SummaryCompressionStrategy:
    name = "summary"

    def compress(self, payload: CompressionInput) -> CompressionOutput:
        lines: list[str] = []
        if payload.previous_summary.strip():
            lines.append(f"Previous summary: {payload.previous_summary.strip()}")
        for item in payload.old_messages:
            role = item.get("role", "unknown")
            content = item.get("content", "").strip().replace("\n", " ")
            if content:
                lines.append(f"{role}: {content[:320]}")
        merged = " | ".join(lines)
        max_chars = max(400, payload.max_tokens * 4 // 3)
        summary = merged[:max_chars] if len(merged) > max_chars else merged
        return CompressionOutput(
            summary_text=summary,
            strategy_name=self.name,
            estimated_tokens=estimate_tokens(summary),
        )


class SlidingWindowCompressionStrategy:
    name = "sliding_window"

    def compress(self, payload: CompressionInput) -> CompressionOutput:
        combined = []
        if payload.previous_summary.strip():
            combined.append(payload.previous_summary.strip())
        for item in payload.old_messages[-6:]:
            role = item.get("role", "unknown")
            content = item.get("content", "").strip().replace("\n", " ")
            combined.append(f"{role}: {content}")
        text = "\n".join(combined)
        max_chars = max(400, payload.max_tokens * 4 // 2)
        summary = text[-max_chars:] if len(text) > max_chars else text
        return CompressionOutput(
            summary_text=summary,
            strategy_name=self.name,
            estimated_tokens=estimate_tokens(summary),
        )


class KeywordCompressionStrategy:
    name = "vectorized"

    def compress(self, payload: CompressionInput) -> CompressionOutput:
        corpus = [payload.previous_summary]
        corpus.extend(item.get("content", "") for item in payload.old_messages)
        words = re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", " ".join(corpus).lower())
        scores: dict[str, int] = {}
        for token in words:
            if len(token) < 2:
                continue
            scores[token] = scores.get(token, 0) + 1
        top_keywords = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:120]
        
        if not top_keywords:
            summary = "No significant keywords found."
        else:
            summary = " ".join([f"{word}:{count}" for word, count in top_keywords])
            
        return CompressionOutput(
            summary_text=summary,
            strategy_name=self.name,
            estimated_tokens=estimate_tokens(summary),
        )


def load_class(path: str) -> Any:
    module_name, class_name = path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class CompressionRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, CompressionStrategy] = {
            SummaryCompressionStrategy.name: SummaryCompressionStrategy(),
            SlidingWindowCompressionStrategy.name: SlidingWindowCompressionStrategy(),
            KeywordCompressionStrategy.name: KeywordCompressionStrategy(),
        }

    def register(self, name: str, strategy: CompressionStrategy) -> None:
        self._strategies[name] = strategy

    def load_plugins(self, plugins: list[str]) -> None:
        for item in plugins:
            cls = load_class(item)
            strategy = cls()
            name = getattr(strategy, "name", cls.__name__)
            self.register(name, strategy)

    def get(self, name: str) -> CompressionStrategy:
        if name in self._strategies:
            return self._strategies[name]
        raise KeyError(f"Compression strategy not found: {name}")
