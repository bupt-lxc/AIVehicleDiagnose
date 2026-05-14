"""DTC Query Tool for LangGraph Integration

用于查询 DTC 故障码信息的工具，支持 trouble_code 和 display_trouble_code 查询。
集成 LangGraph 框架，提供标准化的输入输出 Schema。
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field, asdict

DB_PATH = Path(__file__).parent.parent.parent.parent / "vehicle_info" / "dtcs_complete.db"


@dataclass
class DTCInfo:
    """单个 DTC 信息"""
    dtc_id: int
    trouble_code: str
    display_trouble_code: str
    description: str
    dtc_level: int
    ecu_shortname: str
    ecu_longname: str
    sdgs: list = field(default_factory=list)


@dataclass
class DTCQueryOutput:
    """DTC 查询工具输出"""
    success: bool
    query_code: str
    result_count: int
    results: list = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    solution: Optional[str] = None


ERROR_CODES = {
    "EMPTY_INPUT": {
        "message": "输入的故障码为空",
        "solution": "请提供有效的故障码，如 'C101D29' 或 '28'"
    },
    "INVALID_FORMAT": {
        "message": "故障码格式无效",
        "solution": "故障码应为数字或字母数字组合，如 'C101D29', 'U112100', '28'"
    },
    "NOT_FOUND": {
        "message": "未找到匹配的 DTC 记录",
        "solution": "请检查故障码是否正确，或尝试使用 display_trouble_code 格式查询"
    },
    "DB_CONNECTION_ERROR": {
        "message": "数据库连接失败",
        "solution": "请检查数据库文件是否存在：vehicle_info/dtcs_complete.db"
    },
    "QUERY_ERROR": {
        "message": "数据库查询过程中发生错误",
        "solution": "请检查数据库完整性，或联系技术支持"
    }
}


def parse_sdg_content(sdg_str: str) -> dict:
    """解析 SDG 内容字符串为字典"""
    if not sdg_str:
        return {}
    try:
        return json.loads(sdg_str.replace("'", '"'))
    except json.JSONDecodeError:
        return {"raw": sdg_str}


def query_dtc_by_code(code: str, include_sdgs: bool = True, limit: int = 10) -> DTCQueryOutput:
    """
    根据 trouble_code 或 display_trouble_code 查询 DTC 信息
    
    Args:
        code: 故障码（支持 trouble_code 或 display_trouble_code）
        include_sdgs: 是否包含 SDG 环境数据
        limit: 返回结果数量限制
        
    Returns:
        DTCQueryOutput: 标准化的查询结果
    """
    if not code or not code.strip():
        error = ERROR_CODES["EMPTY_INPUT"]
        return DTCQueryOutput(
            success=False,
            query_code=code or "",
            result_count=0,
            error_code="EMPTY_INPUT",
            error_message=error["message"],
            solution=error["solution"]
        )
    
    code = code.strip().upper()
    
    if not all(c.isalnum() for c in code):
        error = ERROR_CODES["INVALID_FORMAT"]
        return DTCQueryOutput(
            success=False,
            query_code=code,
            result_count=0,
            error_code="INVALID_FORMAT",
            error_message=error["message"],
            solution=error["solution"]
        )
    
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
    except sqlite3.Error as e:
        error = ERROR_CODES["DB_CONNECTION_ERROR"]
        return DTCQueryOutput(
            success=False,
            query_code=code,
            result_count=0,
            error_code="DB_CONNECTION_ERROR",
            error_message=f"{error['message']}: {str(e)}",
            solution=error["solution"]
        )
    
    try:
        results = []
        
        query = """
            SELECT d.id, d.trouble_code, d.display_trouble_code, d.text, d.dtc_level,
                   e.shortname, e.longname
            FROM dtcs d
            JOIN ecus e ON d.ecu_id = e.id
            WHERE d.display_trouble_code = ? OR d.trouble_code = ?
            LIMIT ?
        """
        cursor.execute(query, (code, code, limit))
        rows = cursor.fetchall()
        
        if not rows:
            query_like = """
                SELECT d.id, d.trouble_code, d.display_trouble_code, d.text, d.dtc_level,
                       e.shortname, e.longname
                FROM dtcs d
                JOIN ecus e ON d.ecu_id = e.id
                WHERE d.display_trouble_code LIKE ? OR d.trouble_code LIKE ?
                LIMIT ?
            """
            cursor.execute(query_like, (f"{code}%", f"{code}%", limit))
            rows = cursor.fetchall()
        
        for row in rows:
            dtc_id = row[0]
            
            sdgs = []
            if include_sdgs:
                cursor.execute(
                    "SELECT sdg_content FROM sdgs WHERE dtc_id = ?",
                    (dtc_id,)
                )
                sdg_rows = cursor.fetchall()
                sdgs = [parse_sdg_content(s[0]) for s in sdg_rows[:20]]
            
            results.append(DTCInfo(
                dtc_id=dtc_id,
                trouble_code=row[1],
                display_trouble_code=row[2],
                description=row[3] or "",
                dtc_level=row[4],
                ecu_shortname=row[5],
                ecu_longname=row[6],
                sdgs=sdgs
            ))
        
        conn.close()
        
        if not results:
            error = ERROR_CODES["NOT_FOUND"]
            return DTCQueryOutput(
                success=False,
                query_code=code,
                result_count=0,
                error_code="NOT_FOUND",
                error_message=error["message"],
                solution=error["solution"]
            )
        
        return DTCQueryOutput(
            success=True,
            query_code=code,
            result_count=len(results),
            results=results
        )
        
    except sqlite3.Error as e:
        conn.close()
        error = ERROR_CODES["QUERY_ERROR"]
        return DTCQueryOutput(
            success=False,
            query_code=code,
            result_count=0,
            error_code="QUERY_ERROR",
            error_message=f"{error['message']}: {str(e)}",
            solution=error["solution"]
        )


def dtc_query_tool(code: str, include_sdgs: bool = True, limit: int = 10) -> dict:
    """
    DTC 查询工具入口函数，用于 LangGraph 集成
    
    Args:
        code: 故障码（支持 trouble_code 或 display_trouble_code 格式）
        include_sdgs: 是否包含 SDG 环境数据
        limit: 返回结果数量限制
        
    Returns:
        包含查询结果的字典，可直接序列化为 JSON
    """
    result = query_dtc_by_code(code, include_sdgs, limit)
    output = asdict(result)
    output['results'] = [asdict(r) for r in result.results]
    return output


def register_tools(registry):
    """注册工具到注册表"""
    registry.register(
        name="query_dtc_info",
        handler=dtc_query_tool,
        description="查询DTC故障码信息。支持trouble_code(如'28')和display_trouble_code(如'C101D29')两种格式。"
                    "返回故障描述、ECU信息、SDG环境数据等详细信息。错误时返回错误代码、说明和解决方案。",
        input_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "DTC故障码，支持两种格式：trouble_code(如'28')或display_trouble_code(如'C101D29','U112100')"
                },
                "include_sdgs": {
                    "type": "boolean",
                    "description": "是否包含SDG环境数据，默认true。设为false可加快查询速度。",
                    "default": True
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制，默认10，范围1-100",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100
                }
            },
            "required": ["code"]
        },
        disclosure_level=0
    )


if __name__ == "__main__":
    print("=" * 70)
    print("DTC 查询工具测试")
    print("=" * 70)
    
    test_codes = [
        "C101D29",
        "28",
        "U112100",
        "INVALID!",
        "",
        "NOTEXIST"
    ]
    
    for code in test_codes:
        print(f"\n查询故障码: '{code}'")
        result = query_dtc_by_code(code, include_sdgs=False, limit=3)
        
        if result.success:
            print(f"  成功，找到 {result.result_count} 条记录")
            for r in result.results[:2]:
                print(f"    - {r.display_trouble_code}: {r.description[:50]}...")
                print(f"      ECU: {r.ecu_shortname}")
        else:
            print(f"  失败: {result.error_code}")
            print(f"    错误: {result.error_message}")
            print(f"    解决: {result.solution}")
