from __future__ import annotations

import asyncio
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.run_test4_full_chain import run_full_chain_case


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    xml_path: str
    user_context: str
    enabled: bool
    label_status: str
    expected_root_cause_keywords: list[str]
    expected_topk_keywords: list[list[str]]
    expected_hitl: bool | None


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _keywords_match(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    normalized = _normalize_text(text)
    return all(_normalize_text(keyword) in normalized for keyword in keywords)


def _hypothesis_topk_hit(hypotheses: list[dict[str, Any]], expected_groups: list[list[str]], top_k: int) -> bool:
    if not expected_groups:
        return False
    for item in hypotheses[:top_k]:
        text = " ".join(
            [
                str(item.get("hypothesis", "")),
                str(item.get("supporting_dtc", "")),
                str(item.get("related_ecu", "")),
            ]
        )
        if any(_keywords_match(text, group) for group in expected_groups):
            return True
    return False


def _parse_iso_duration_ms(started_at: str | None, finished_at: str | None) -> float:
    if not started_at or not finished_at:
        return 0.0
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(finished_at)
    return round((end - start).total_seconds() * 1000, 3)


def _compute_tool_cost(tool_calls: list[dict[str, Any]], weights: dict[str, float]) -> tuple[float, int]:
    total_cost = 0.0
    for item in tool_calls:
        total_cost += float(weights.get(item.get("name"), 1.0))
    return round(total_cost, 6), len(tool_calls)


def _is_diagnostic_hitl(final_state: dict[str, Any]) -> bool:
    diagnostic_hitl_nodes = {
        "hitl_parse_confirm",
        "hitl_low_confidence",
        "hitl_audit_conflict",
    }
    transition_nodes = {
        str(item.get("node", "")).lower()
        for item in final_state.get("transition_log", [])
        if isinstance(item, dict)
    }
    current_point = str(final_state.get("current_hitl_point") or "").lower()
    if current_point and current_point not in {"report_approval", "hitl_report_approval"}:
        return True
    return any(node in transition_nodes for node in diagnostic_hitl_nodes)


def _compute_p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 6)
    sorted_values = sorted(values)
    index = math.ceil(0.95 * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return round(sorted_values[index], 6)


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[BenchmarkCase]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = [
        BenchmarkCase(
            case_id=item["case_id"],
            xml_path=item["xml_path"],
            user_context=item["user_context"],
            enabled=bool(item.get("enabled", True)),
            label_status=item.get("label_status", "ground_truth_pending"),
            expected_root_cause_keywords=list(item.get("expected_root_cause_keywords", [])),
            expected_topk_keywords=list(item.get("expected_topk_keywords", [])),
            expected_hitl=item.get("expected_hitl"),
        )
        for item in payload.get("cases", [])
    ]
    return payload, cases


async def _run_case(case: BenchmarkCase, output_dir: Path, weights: dict[str, float], top_k: int) -> dict[str, Any]:
    case_output_dir = output_dir / case.case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)
    result = await run_full_chain_case(
        xml_path=case.xml_path,
        user_context=case.user_context,
        case_id=case.case_id,
        output_dir=case_output_dir,
    )
    final_state = result["final_state"]
    trace = result["trace"]
    root_cause = final_state.get("preliminary_root_cause", "")
    hypotheses = list(final_state.get("hypothesis_candidates", []))
    root_cause_correct = None
    top_k_hit = None
    if case.label_status == "labeled":
        root_cause_correct = _keywords_match(root_cause, case.expected_root_cause_keywords)
        top_k_hit = _hypothesis_topk_hit(hypotheses, case.expected_topk_keywords, top_k)
    duration_ms = _parse_iso_duration_ms(final_state.get("started_at"), final_state.get("finished_at"))
    tool_cost, tool_call_count = _compute_tool_cost(trace.get("tool_call", []), weights)
    hitl_triggered = _is_diagnostic_hitl(final_state)
    audit_intercepted = not bool(final_state.get("audit_passed"))
    case_result = {
        "case_id": case.case_id,
        "label_status": case.label_status,
        "root_cause": root_cause,
        "root_cause_correct": root_cause_correct,
        "top_k_hit": top_k_hit,
        "misdiagnosed": (not root_cause_correct) if root_cause_correct is not None else None,
        "hitl_triggered": hitl_triggered,
        "rca_iterations": int(final_state.get("rca_iteration", 0) or 0),
        "tool_call_cost": tool_cost,
        "tool_call_count": tool_call_count,
        "latency_ms": duration_ms,
        "audit_intercepted": audit_intercepted,
        "audit_passed": bool(final_state.get("audit_passed")),
        "confidence_score": float(final_state.get("confidence_score", 0.0) or 0.0),
        "confidence_breakdown": final_state.get("confidence_breakdown", {}),
        "hypothesis_candidates": hypotheses,
        "report_path": result["report_path"],
        "trace_path": result["trace_path"],
        "frontend_path": result["frontend_path"],
        "audit_path": result["audit_path"],
    }
    (case_output_dir / "benchmark_case_result.json").write_text(
        json.dumps(case_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return case_result


def _aggregate(case_results: list[dict[str, Any]], total_manifest_cases: int, executed_cases: int, top_k: int) -> dict[str, Any]:
    labeled = [item for item in case_results if item["root_cause_correct"] is not None]
    total_labeled = len(labeled)
    root_cause_accuracy = (
        sum(1 for item in labeled if item["root_cause_correct"]) / total_labeled if total_labeled else 0.0
    )
    top_k_hit_rate = (
        sum(1 for item in labeled if item["top_k_hit"]) / total_labeled if total_labeled else 0.0
    )
    misdiagnosis_rate = (
        sum(1 for item in labeled if item["misdiagnosed"]) / total_labeled if total_labeled else 0.0
    )
    hitl_trigger_rate = (
        sum(1 for item in case_results if item["hitl_triggered"]) / executed_cases if executed_cases else 0.0
    )
    average_reasoning_rounds = (
        statistics.mean(item["rca_iterations"] for item in case_results) if case_results else 0.0
    )
    average_tool_call_cost = (
        statistics.mean(item["tool_call_cost"] for item in case_results) if case_results else 0.0
    )
    p95_latency_ms = _compute_p95([item["latency_ms"] for item in case_results])
    audit_intercept_rate = (
        sum(1 for item in case_results if item["audit_intercepted"]) / executed_cases if executed_cases else 0.0
    )
    return {
        "manifest_case_count": total_manifest_cases,
        "executed_case_count": executed_cases,
        "labeled_case_count": total_labeled,
        "label_coverage": round((total_labeled / executed_cases) if executed_cases else 0.0, 6),
        "top_k": top_k,
        "root_cause_accuracy": round(root_cause_accuracy, 6),
        "top_k_hit_rate": round(top_k_hit_rate, 6),
        "misdiagnosis_rate": round(misdiagnosis_rate, 6),
        "hitl_trigger_rate": round(hitl_trigger_rate, 6),
        "average_reasoning_rounds": round(average_reasoning_rounds, 6),
        "average_tool_call_cost": round(average_tool_call_cost, 6),
        "p95_latency_ms": round(p95_latency_ms, 6),
        "audit_intercept_rate": round(audit_intercept_rate, 6),
    }


def _render_markdown(summary: dict[str, Any], case_results: list[dict[str, Any]]) -> str:
    lines = [
        "# Benchmark Report",
        "",
        "## Summary",
        f"- Manifest cases: {summary['manifest_case_count']}",
        f"- Executed cases: {summary['executed_case_count']}",
        f"- Labeled cases: {summary['labeled_case_count']}",
        f"- Label coverage: {summary['label_coverage']}",
        f"- Root cause accuracy: {summary['root_cause_accuracy']}",
        f"- Top-{summary['top_k']} hit rate: {summary['top_k_hit_rate']}",
        f"- Misdiagnosis rate: {summary['misdiagnosis_rate']}",
        f"- HITL trigger rate: {summary['hitl_trigger_rate']}",
        f"- Average reasoning rounds: {summary['average_reasoning_rounds']}",
        f"- Average tool call cost: {summary['average_tool_call_cost']}",
        f"- P95 latency ms: {summary['p95_latency_ms']}",
        f"- Audit intercept rate: {summary['audit_intercept_rate']}",
        "",
        "## Cases",
    ]
    for item in case_results:
        lines.extend(
            [
                f"### {item['case_id']}",
                f"- Label status: {item['label_status']}",
                f"- Root cause correct: {item['root_cause_correct']}",
                f"- Top-K hit: {item['top_k_hit']}",
                f"- Misdiagnosed: {item['misdiagnosed']}",
                f"- HITL triggered: {item['hitl_triggered']}",
                f"- RCA iterations: {item['rca_iterations']}",
                f"- Tool call cost: {item['tool_call_cost']}",
                f"- Latency ms: {item['latency_ms']}",
                f"- Audit intercepted: {item['audit_intercepted']}",
                f"- Confidence score: {item['confidence_score']}",
                f"- Report: {item['report_path']}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


async def main() -> None:
    manifest_path = PROJECT_ROOT / "benchmark" / "cases.json"
    manifest, cases = _load_manifest(manifest_path)
    top_k = int(manifest.get("top_k", 3))
    weights = {str(k): float(v) for k, v in manifest.get("tool_cost_weights", {}).items()}
    output_dir = PROJECT_ROOT / "benchmark" / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    case_results: list[dict[str, Any]] = []
    for case in cases:
        if not case.enabled:
            continue
        case_results.append(await _run_case(case, output_dir, weights, top_k))

    summary = _aggregate(
        case_results=case_results,
        total_manifest_cases=len(cases),
        executed_cases=len(case_results),
        top_k=top_k,
    )
    payload = {
        "manifest_path": str(manifest_path),
        "summary": summary,
        "cases": case_results,
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "benchmark_report.md").write_text(
        _render_markdown(summary, case_results),
        encoding="utf-8",
    )
    print(f"BENCHMARK_OUTPUT_DIR={output_dir.resolve()}")
    print(f"SUMMARY_JSON={(output_dir / 'benchmark_summary.json').resolve()}")
    print(f"REPORT_MD={(output_dir / 'benchmark_report.md').resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
