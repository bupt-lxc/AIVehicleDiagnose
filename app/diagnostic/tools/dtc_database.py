"""DTC 数据库查询工具

从大众/奥迪 DTC 数据库查询故障码详细信息。
"""

import sqlite3
import json
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent.parent.parent / "vehicle_info" / "dtcs_complete.db"


def _normalize_dtc_queries(dtc_code: str) -> tuple[set[str], set[str]]:
    cleaned = "".join(ch for ch in str(dtc_code or "").upper().strip() if ch.isalnum())
    exact_codes: set[str] = set()
    vw5_codes: set[str] = set()
    if not cleaned:
        return exact_codes, vw5_codes
    exact_codes.add(cleaned)
    if len(cleaned) >= 7:
        exact_codes.add(cleaned[:7])
    if len(cleaned) >= 5:
        vw5_codes.add(cleaned[:5])
    if len(cleaned) == 7 and (cleaned.endswith("00") or cleaned.endswith("F0")):
        vw5_codes.add(cleaned[:5])
    return exact_codes, vw5_codes


def _dedupe_results(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for item in rows:
        key = (item.get("dtc_code"), item.get("ecu_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped

def query_dtc(dtc_code: str, limit: int = 5) -> dict[str, Any]:
    """
    根据 DTC 码查询数据库中的详细信息。
    
    Args:
        dtc_code: DTC 故障码 (如 "U112100", "B153100")
        limit: 返回结果数量限制
        
    Returns:
        包含 DTC 详情的字典
    """
    if not dtc_code:
        return {"error": "DTC 码不能为空"}
    
    dtc_code = dtc_code.upper().strip()
    exact_codes, vw5_codes = _normalize_dtc_queries(dtc_code)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        results: list[dict[str, Any]] = []
        
        query_exact = """
            SELECT d.id, d.display_trouble_code, d.text, d.dtc_level, 
                   e.ecu_id, e.shortname, e.longname
            FROM dtcs d
            JOIN ecus e ON d.ecu_id = e.id
            WHERE d.display_trouble_code = ?
               OR d.dtc_id = ?
               OR d.shortname = ?
            LIMIT 200
        """
        query_vw5 = """
            SELECT d.id, d.display_trouble_code, d.text, d.dtc_level, 
                   e.ecu_id, e.shortname, e.longname
            FROM dtcs d
            JOIN ecus e ON d.ecu_id = e.id
            WHERE d.display_trouble_code LIKE ?
               OR d.dtc_id LIKE ?
               OR d.shortname LIKE ?
            LIMIT 200
        """

        def _append_rows(rows):
            for row in rows:
                dtc_id = row[0]
                sdg_data = []
                cursor.execute("""SELECT sdg_content FROM sdgs WHERE dtc_id = ?""", (dtc_id,))
                sdg_rows = cursor.fetchall()
                for sdg_row in sdg_rows:
                    try:
                        sdg = json.loads(sdg_row[0].replace("'", '"'))
                        sdg_data.append(sdg)
                    except Exception:
                        sdg_data.append({"raw": sdg_row[0]})
                results.append(
                    {
                        "dtc_code": row[1],
                        "description": row[2],
                        "dtc_level": row[3],
                        "ecu_id": row[4],
                        "ecu_shortname": row[5],
                        "ecu_longname": row[6],
                        "environment_data": sdg_data[:10] if sdg_data else [],
                    }
                )

        for exact in sorted(exact_codes):
            cursor.execute(query_exact, (exact, exact, exact))
            _append_rows(cursor.fetchall())

        for prefix in sorted(vw5_codes):
            like_value = f"{prefix}%"
            cursor.execute(query_vw5, (like_value, like_value, like_value))
            _append_rows(cursor.fetchall())
        
        conn.close()
        results = _dedupe_results(results, max(1, int(limit or 5)))
        
        if not results:
            return {
                "success": False,
                "error": f"未找到 DTC: {dtc_code}",
                "suggestion": "尝试输入7位通用error type或5位大众DTC码"
            }
        
        return {
            "success": True,
            "query_dtc": dtc_code,
            "result_count": len(results),
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"数据库查询失败: {str(e)}"
        }


def search_dtc_by_keyword(keyword: str, limit: int = 10) -> dict[str, Any]:
    """
    根据关键词搜索 DTC。
    
    Args:
        keyword: 搜索关键词
        limit: 返回结果数量限制
        
    Returns:
        搜索结果列表
    """
    if not keyword:
        return {"error": "关键词不能为空"}
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        query = """
            SELECT d.id, d.display_trouble_code, d.text, d.dtc_level, 
                   e.ecu_id, e.shortname, e.longname
            FROM dtcs d
            JOIN ecus e ON d.ecu_id = e.id
            WHERE d.text LIKE ? OR d.display_trouble_code LIKE ?
            LIMIT ?
        """
        
        pattern = f"%{keyword}%"
        cursor.execute(query, (pattern, pattern, limit))
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                "dtc_code": row[1],
                "description": row[2],
                "dtc_level": row[3],
                "ecu_id": row[4],
                "ecu_shortname": row[5],
                "ecu_longname": row[6],
            })
        
        conn.close()
        
        return {
            "success": True,
            "keyword": keyword,
            "result_count": len(results),
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"搜索失败: {str(e)}"
        }


def get_dtc_summary(dtc_code: str) -> dict[str, Any]:
    """
    获取 DTC 的简要摘要信息（不含环境数据）。
    
    Args:
        dtc_code: DTC 故障码
        
    Returns:
        DTC 摘要信息
    """
    result = query_dtc(dtc_code, limit=3)
    
    if not result.get("success"):
        return result
    
    summary = {
        "dtc_code": result["query_dtc"],
        "found_count": result["result_count"],
        "details": []
    }
    
    for r in result.get("results", []):
        summary["details"].append({
            "code": r["dtc_code"],
            "description": r["description"],
            "level": r["dtc_level"],
            "ecu": r["ecu_shortname"],
            "ecu_name": r["ecu_longname"],
        })
    
    return summary


def register_tools(registry):
    """注册工具到注册表"""
    registry.register(
        name="query_dtc",
        handler=query_dtc,
        description="根据DTC故障码查询数据库中的详细信息，包括故障描述、ECU信息、环境数据等",
        input_schema={
            "type": "object",
            "properties": {
                "dtc_code": {
                    "type": "string", 
                    "description": "DTC故障码 (如 U112100, B153100, C12D7F0)"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制",
                    "default": 5
                }
            },
            "required": ["dtc_code"]
        },
        disclosure_level=1
    )
    
    registry.register(
        name="search_dtc",
        handler=search_dtc_by_keyword,
        description="根据关键词搜索DTC故障码，适用于不确定具体DTC码时",
        input_schema={
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string", 
                    "description": "搜索关键词 (如 'radar', 'communication', 'ESP')"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制",
                    "default": 10
                }
            },
            "required": ["keyword"]
        },
        disclosure_level=2
    )
    
    registry.register(
        name="get_dtc_summary",
        handler=get_dtc_summary,
        description="获取DTC故障码的简要摘要信息（不含详细环境数据）",
        input_schema={
            "type": "object",
            "properties": {
                "dtc_code": {
                    "type": "string", 
                    "description": "DTC故障码"
                }
            },
            "required": ["dtc_code"]
        },
        disclosure_level=0
    )


if __name__ == "__main__":
    print("=" * 70)
    print("DTC 数据库查询测试")
    print("=" * 70)
    
    test_codes = ["U112100", "B153100", "C12D7F0", "U146700"]
    
    for code in test_codes:
        print(f"\n查询 DTC: {code}")
        result = query_dtc(code)
        print(f"  成功: {result.get('success')}")
        if result.get('success'):
            print(f"  结果数: {result['result_count']}")
            for r in result['results'][:2]:
                print(f"    {r['dtc_code']}: {r['description']}")
                print(f"    ECU: {r['ecu_shortname']} - {r['ecu_longname']}")
        else:
            print(f"  错误: {result.get('error')}")
    
    print("\n\n关键词搜索测试:")
    result = search_dtc_by_keyword("radar")
    print(f"  搜索 'radar': 找到 {result['result_count']} 条")
    
    result = search_dtc_by_keyword("no communication")
    print(f"  搜索 'no communication': 找到 {result['result_count']} 条")
