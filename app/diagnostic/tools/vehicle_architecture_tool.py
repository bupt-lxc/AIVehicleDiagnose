"""Vehicle Architecture Query Tools for LangGraph Integration

用于查询车辆架构信息的工具集，包括：
1. ECU 信息查询工具 - 根据 ECU 地址查询 ECU 信息
2. ECU 邻居查询工具 - 查询 N 跳内可连接的 ECU
3. ECU 跳数查询工具 - 查询两个 ECU 之间的跳数和路径

跳数定义：
- 0跳：ECU自身
- 1跳：同一组内的ECU（main_ecu + subecu_of_main_ecu 或 connected_ecu + subecu_of_connected_ecu）
- 2跳：直接网络连接的另一组ECU
- 3+跳：通过中间ECU组连接
"""

import sqlite3
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from collections import deque

DB_PATH = Path(__file__).parent.parent.parent.parent / "vehicle_info" / "vehicle_architecture.db"


@dataclass
class ECUInfo:
    ecu_address: str
    ecu_name: str
    ecu_description: str
    ecu_domain: str


@dataclass
class ECUInfoOutput:
    success: bool
    ecu_address: str
    ecu_info: Optional[ECUInfo] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    solution: Optional[str] = None


@dataclass
class ECUPathNode:
    ecu_address: str
    ecu_name: str
    hop: int


@dataclass
class ECUNeighborOutput:
    success: bool
    source_ecu: str
    source_ecu_name: str
    max_hops: int
    neighbors_by_hop: dict = field(default_factory=dict)
    total_neighbors: int = 0
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    solution: Optional[str] = None


@dataclass
class ECUHopDistanceOutput:
    success: bool
    source_ecu: str
    source_ecu_name: str
    target_ecu: str
    target_ecu_name: str
    hop_distance: int = -1
    path: list = field(default_factory=list)
    path_edges: list = field(default_factory=list)
    connection_type: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    solution: Optional[str] = None


ERROR_CODES = {
    "EMPTY_INPUT": {
        "message": "输入的ECU地址为空",
        "solution": "请提供有效的ECU地址，如 '0019', '0001'"
    },
    "INVALID_FORMAT": {
        "message": "ECU地址格式无效",
        "solution": "ECU地址应为字母数字组合，如 '0019', '8103'"
    },
    "ECU_NOT_FOUND": {
        "message": "未找到指定的ECU",
        "solution": "请检查ECU地址是否正确，或该ECU不在当前车辆架构中"
    },
    "DB_CONNECTION_ERROR": {
        "message": "数据库连接失败",
        "solution": "请检查数据库文件是否存在：vehicle_info/vehicle_architecture.db"
    },
    "QUERY_ERROR": {
        "message": "数据库查询过程中发生错误",
        "solution": "请检查数据库完整性，或联系技术支持"
    },
    "SAME_ECU": {
        "message": "源ECU和目标ECU相同",
        "solution": "查询两个不同ECU之间的跳数"
    },
    "NOT_REACHABLE": {
        "message": "两个ECU在指定跳数内不可达",
        "solution": "尝试增加最大跳数，或检查ECU地址是否正确"
    }
}


def get_db_connection():
    return sqlite3.connect(str(DB_PATH))


def parse_ecu_list(ecu_str) -> set:
    if not ecu_str or ecu_str == 'NULL':
        return set()
    return set(e.strip() for e in str(ecu_str).split(',') if e.strip() and e.strip() != 'NULL')


def get_ecu_info_from_db(ecu_address: str) -> Optional[ECUInfo]:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ecu_address, ecu_name, ecu_description, ecu_domain FROM ecu_info WHERE ecu_address = ?",
            (ecu_address,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return ECUInfo(
                ecu_address=row[0],
                ecu_name=row[1] or "",
                ecu_description=row[2] or "",
                ecu_domain=row[3] or "Unknown"
            )
        return None
    except sqlite3.Error:
        return None


def build_ecu_groups() -> dict:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT main_ecu, subecu_of_main_ecu, connected_ecu, subecu_of_connected_ecu, connection_protocols FROM topology"
        )
        
        ecu_to_group = {}
        processed_groups = set()
        
        for row in cursor.fetchall():
            main, sub_main, connected, sub_connected, protocols_raw = row
            main_group = frozenset(parse_ecu_list(main) | parse_ecu_list(sub_main))
            connected_group = frozenset(parse_ecu_list(connected) | parse_ecu_list(sub_connected))
            
            if main_group and main_group not in processed_groups:
                for ecu in main_group:
                    ecu_to_group[ecu] = main_group
                processed_groups.add(main_group)
            
            if connected_group and connected_group not in processed_groups:
                for ecu in connected_group:
                    ecu_to_group[ecu] = connected_group
                processed_groups.add(connected_group)
        
        conn.close()
        return ecu_to_group
    except sqlite3.Error:
        return {}


def build_group_adjacency() -> dict:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT main_ecu, subecu_of_main_ecu, connected_ecu, subecu_of_connected_ecu, connection_protocols FROM topology"
        )
        
        group_adjacency = {}
        
        for row in cursor.fetchall():
            main, sub_main, connected, sub_connected, protocols_raw = row
            main_group = frozenset(parse_ecu_list(main) | parse_ecu_list(sub_main))
            connected_group = frozenset(parse_ecu_list(connected) | parse_ecu_list(sub_connected))
            
            if main_group and main_group not in group_adjacency:
                group_adjacency[main_group] = {}
            if connected_group and connected_group not in group_adjacency:
                group_adjacency[connected_group] = {}
            
            if main_group and connected_group:
                protocols = [p.strip() for p in str(protocols_raw or "").split(",") if p.strip()]
                existing_main = set(group_adjacency[main_group].get(connected_group, set()))
                existing_main.update(protocols)
                group_adjacency[main_group][connected_group] = existing_main
                existing_connected = set(group_adjacency[connected_group].get(main_group, set()))
                existing_connected.update(protocols)
                group_adjacency[connected_group][main_group] = existing_connected
        
        conn.close()
        return group_adjacency
    except sqlite3.Error:
        return {}


def query_ecu_info(ecu_address: str) -> ECUInfoOutput:
    if not ecu_address or not ecu_address.strip():
        error = ERROR_CODES["EMPTY_INPUT"]
        return ECUInfoOutput(
            success=False,
            ecu_address=ecu_address or "",
            error_code="EMPTY_INPUT",
            error_message=error["message"],
            solution=error["solution"]
        )
    
    ecu_address = ecu_address.upper().strip()
    
    if not all(c.isalnum() for c in ecu_address):
        error = ERROR_CODES["INVALID_FORMAT"]
        return ECUInfoOutput(
            success=False,
            ecu_address=ecu_address,
            error_code="INVALID_FORMAT",
            error_message=error["message"],
            solution=error["solution"]
        )
    
    try:
        ecu_info = get_ecu_info_from_db(ecu_address)
        
        if not ecu_info:
            error = ERROR_CODES["ECU_NOT_FOUND"]
            return ECUInfoOutput(
                success=False,
                ecu_address=ecu_address,
                error_code="ECU_NOT_FOUND",
                error_message=error["message"],
                solution=error["solution"]
            )
        
        return ECUInfoOutput(
            success=True,
            ecu_address=ecu_address,
            ecu_info=ecu_info
        )
        
    except Exception as e:
        error = ERROR_CODES["QUERY_ERROR"]
        return ECUInfoOutput(
            success=False,
            ecu_address=ecu_address,
            error_code="QUERY_ERROR",
            error_message=f"{error['message']}: {str(e)}",
            solution=error["solution"]
        )


def get_ecu_neighbors(ecu_address: str, max_hops: int = 2) -> ECUNeighborOutput:
    if not ecu_address or not ecu_address.strip():
        error = ERROR_CODES["EMPTY_INPUT"]
        return ECUNeighborOutput(
            success=False,
            source_ecu=ecu_address or "",
            source_ecu_name="",
            max_hops=max_hops,
            error_code="EMPTY_INPUT",
            error_message=error["message"],
            solution=error["solution"]
        )
    
    ecu_address = ecu_address.upper().strip()
    
    source_info = get_ecu_info_from_db(ecu_address)
    source_name = source_info.ecu_name if source_info else ""
    
    ecu_to_group = build_ecu_groups()
    
    if ecu_address not in ecu_to_group:
        error = ERROR_CODES["ECU_NOT_FOUND"]
        return ECUNeighborOutput(
            success=False,
            source_ecu=ecu_address,
            source_ecu_name=source_name,
            max_hops=max_hops,
            error_code="ECU_NOT_FOUND",
            error_message=error["message"],
            solution=error["solution"]
        )
    
    source_group = ecu_to_group[ecu_address]
    group_adjacency = build_group_adjacency()
    
    neighbors_by_hop = {}
    
    group_members = source_group - {ecu_address}
    if group_members:
        hop_1_neighbors = []
        for ecu in group_members:
            info = get_ecu_info_from_db(ecu)
            hop_1_neighbors.append({
                "ecu_address": ecu,
                "ecu_name": info.ecu_name if info else "",
                "via_protocols": []
            })
        neighbors_by_hop["1"] = sorted(hop_1_neighbors, key=lambda x: x["ecu_address"])
    
    visited_groups = {source_group}
    current_groups = {source_group}
    
    for hop in range(2, max_hops + 1):
        next_groups = set()
        hop_neighbors = []
        
        for grp in current_groups:
            for neighbor_group, protocols in group_adjacency.get(grp, {}).items():
                if neighbor_group not in visited_groups:
                    next_groups.add(neighbor_group)
                    visited_groups.add(neighbor_group)
                    
                    for ecu in neighbor_group:
                        info = get_ecu_info_from_db(ecu)
                        hop_neighbors.append({
                            "ecu_address": ecu,
                            "ecu_name": info.ecu_name if info else "",
                            "via_protocols": sorted(list(protocols)),
                        })
        
        if hop_neighbors:
            unique_neighbors = {n["ecu_address"]: n for n in hop_neighbors}
            neighbors_by_hop[str(hop)] = sorted(
                list(unique_neighbors.values()), 
                key=lambda x: x["ecu_address"]
            )
        
        current_groups = next_groups
        
        if not current_groups:
            break
    
    total = sum(len(v) for v in neighbors_by_hop.values())
    
    return ECUNeighborOutput(
        success=True,
        source_ecu=ecu_address,
        source_ecu_name=source_name,
        max_hops=max_hops,
        neighbors_by_hop=neighbors_by_hop,
        total_neighbors=total
    )


def check_ecu_hop_distance(source_ecu: str, target_ecu: str, max_hops: int = 10) -> ECUHopDistanceOutput:
    if not source_ecu or not source_ecu.strip():
        error = ERROR_CODES["EMPTY_INPUT"]
        return ECUHopDistanceOutput(
            success=False,
            source_ecu=source_ecu or "",
            source_ecu_name="",
            target_ecu=target_ecu or "",
            target_ecu_name="",
            error_code="EMPTY_INPUT",
            error_message=error["message"],
            solution=error["solution"]
        )
    
    if not target_ecu or not target_ecu.strip():
        error = ERROR_CODES["EMPTY_INPUT"]
        return ECUHopDistanceOutput(
            success=False,
            source_ecu=source_ecu,
            source_ecu_name="",
            target_ecu=target_ecu or "",
            target_ecu_name="",
            error_code="EMPTY_INPUT",
            error_message="目标ECU地址为空",
            solution="请提供有效的目标ECU地址"
        )
    
    source_ecu = source_ecu.upper().strip()
    target_ecu = target_ecu.upper().strip()
    
    source_info = get_ecu_info_from_db(source_ecu)
    target_info = get_ecu_info_from_db(target_ecu)
    
    source_name = source_info.ecu_name if source_info else ""
    target_name = target_info.ecu_name if target_info else ""
    
    if source_ecu == target_ecu:
        return ECUHopDistanceOutput(
            success=True,
            source_ecu=source_ecu,
            source_ecu_name=source_name,
            target_ecu=target_ecu,
            target_ecu_name=target_name,
            hop_distance=0,
            path=[ECUPathNode(ecu_address=source_ecu, ecu_name=source_name, hop=0)],
            connection_type="self"
        )
    
    ecu_to_group = build_ecu_groups()
    
    if source_ecu not in ecu_to_group:
        error = ERROR_CODES["ECU_NOT_FOUND"]
        return ECUHopDistanceOutput(
            success=False,
            source_ecu=source_ecu,
            source_ecu_name=source_name,
            target_ecu=target_ecu,
            target_ecu_name=target_name,
            error_code="ECU_NOT_FOUND",
            error_message=f"源ECU {source_ecu} 未找到",
            solution=error["solution"]
        )
    
    if target_ecu not in ecu_to_group:
        error = ERROR_CODES["ECU_NOT_FOUND"]
        return ECUHopDistanceOutput(
            success=False,
            source_ecu=source_ecu,
            source_ecu_name=source_name,
            target_ecu=target_ecu,
            target_ecu_name=target_name,
            error_code="ECU_NOT_FOUND",
            error_message=f"目标ECU {target_ecu} 未找到",
            solution=error["solution"]
        )
    
    source_group = ecu_to_group[source_ecu]
    target_group = ecu_to_group[target_ecu]
    
    if source_group == target_group:
        group_ecus = sorted(list(source_group))
        path = [
            ECUPathNode(ecu_address=source_ecu, ecu_name=source_name, hop=0),
            ECUPathNode(ecu_address=target_ecu, ecu_name=target_name, hop=1)
        ]
        return ECUHopDistanceOutput(
            success=True,
            source_ecu=source_ecu,
            source_ecu_name=source_name,
            target_ecu=target_ecu,
            target_ecu_name=target_name,
            hop_distance=1,
            path=path,
            connection_type="same_group"
        )
    
    group_adjacency = build_group_adjacency()
    
    visited_groups = {source_group: None}
    queue = deque([(source_group, 2)])
    
    while queue:
        current_group, hops = queue.popleft()
        
        if hops > max_hops:
            continue
        
        for neighbor_group, protocols in group_adjacency.get(current_group, {}).items():
            if neighbor_group not in visited_groups:
                visited_groups[neighbor_group] = (current_group, sorted(list(protocols)))
                
                if neighbor_group == target_group:
                    group_path = []
                    g = target_group
                    while g is not None:
                        group_path.append(g)
                        parent_info = visited_groups[g]
                        if parent_info is None:
                            g = None
                        else:
                            g = parent_info[0]
                    group_path.reverse()
                    
                    ecu_path = [ECUPathNode(ecu_address=source_ecu, ecu_name=source_name, hop=0)]
                    path_edges = []
                    for i, grp in enumerate(group_path):
                        if i == 0:
                            continue
                        parent_group = group_path[i - 1]
                        link_protocols = sorted(list(group_adjacency.get(parent_group, {}).get(grp, set())))
                        representative = min(grp)
                        info = get_ecu_info_from_db(representative)
                        ecu_path.append(ECUPathNode(
                            ecu_address=representative,
                            ecu_name=info.ecu_name if info else "",
                            hop=i
                        ))
                        path_edges.append(
                            {
                                "from_ecu": min(parent_group),
                                "to_ecu": representative,
                                "protocols": link_protocols,
                            }
                        )
                    ecu_path.append(ECUPathNode(ecu_address=target_ecu, ecu_name=target_name, hop=len(group_path)))
                    
                    connection_type = "direct_connection" if hops == 2 else "multi_hop"
                    
                    return ECUHopDistanceOutput(
                        success=True,
                        source_ecu=source_ecu,
                        source_ecu_name=source_name,
                        target_ecu=target_ecu,
                        target_ecu_name=target_name,
                        hop_distance=hops,
                        path=ecu_path,
                        path_edges=path_edges,
                        connection_type=connection_type
                    )
                
                queue.append((neighbor_group, hops + 1))
    
    error = ERROR_CODES["NOT_REACHABLE"]
    return ECUHopDistanceOutput(
        success=False,
        source_ecu=source_ecu,
        source_ecu_name=source_name,
        target_ecu=target_ecu,
        target_ecu_name=target_name,
        error_code="NOT_REACHABLE",
        error_message=error["message"],
        solution=error["solution"]
    )


def ecu_info_tool(ecu_address: str) -> dict:
    result = query_ecu_info(ecu_address)
    output = asdict(result)
    if result.ecu_info:
        output['ecu_info'] = asdict(result.ecu_info)
    return output


def ecu_neighbors_tool(ecu_address: str, max_hops: int = 2) -> dict:
    result = get_ecu_neighbors(ecu_address, max_hops)
    return asdict(result)


def ecu_hop_distance_tool(source_ecu: str, target_ecu: str, max_hops: int = 10) -> dict:
    result = check_ecu_hop_distance(source_ecu, target_ecu, max_hops)
    output = asdict(result)
    output['path'] = [asdict(p) for p in result.path]
    return output


def register_tools(registry):
    registry.register(
        name="query_ecu_info",
        handler=ecu_info_tool,
        description="根据ECU地址查询ECU信息，包括ECU名称、描述和所属域。",
        input_schema={
            "type": "object",
            "properties": {
                "ecu_address": {
                    "type": "string",
                    "description": "ECU地址，如 '0019', '0001', '8103'"
                }
            },
            "required": ["ecu_address"]
        },
        disclosure_level=0
    )
    
    registry.register(
        name="get_ecu_neighbors",
        handler=ecu_neighbors_tool,
        description="查询指定ECU在N跳范围内可连接的所有ECU，返回各跳的ECU列表及其名称。"
                    "跳数定义：1跳=同组ECU，2跳=直接连接的ECU组，3+跳=通过中间组连接。",
        input_schema={
            "type": "object",
            "properties": {
                "ecu_address": {
                    "type": "string",
                    "description": "ECU地址，如 '0019', '0001'"
                },
                "max_hops": {
                    "type": "integer",
                    "description": "最大跳数，默认2",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["ecu_address"]
        },
        disclosure_level=0
    )
    
    registry.register(
        name="check_ecu_hop_distance",
        handler=ecu_hop_distance_tool,
        description="查询两个ECU之间的跳数距离和连接路径。"
                    "跳数定义：0跳=自身，1跳=同组ECU，2跳=直接连接的ECU组，3+跳=通过中间组连接。",
        input_schema={
            "type": "object",
            "properties": {
                "source_ecu": {
                    "type": "string",
                    "description": "源ECU地址，如 '0019'"
                },
                "target_ecu": {
                    "type": "string",
                    "description": "目标ECU地址，如 '0075'"
                },
                "max_hops": {
                    "type": "integer",
                    "description": "最大搜索跳数，默认10",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["source_ecu", "target_ecu"]
        },
        disclosure_level=0
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Vehicle Architecture Query Tools 测试")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("测试 1: ECU 信息查询")
    print("=" * 70)
    
    test_ecus = ["0019", "0001", "8103", "NOTEXIST", ""]
    for ecu in test_ecus:
        print(f"\n查询 ECU: '{ecu}'")
        result = query_ecu_info(ecu)
        if result.success:
            print(f"  成功")
            print(f"    名称: {result.ecu_info.ecu_name}")
            print(f"    域: {result.ecu_info.ecu_domain}")
        else:
            print(f"  失败: {result.error_code}")
            print(f"    错误: {result.error_message}")
            print(f"    解决: {result.solution}")
    
    print("\n" + "=" * 70)
    print("测试 2: ECU 邻居查询")
    print("=" * 70)
    
    test_cases = [("0019", 2), ("0001", 3), ("NOTEXIST", 2)]
    for ecu, hops in test_cases:
        print(f"\n查询 ECU '{ecu}' 在 {hops} 跳内的邻居:")
        result = get_ecu_neighbors(ecu, hops)
        if result.success:
            print(f"  成功，共 {result.total_neighbors} 个邻居")
            for hop, neighbors in result.neighbors_by_hop.items():
                print(f"    {hop}跳: {len(neighbors)} 个")
                for n in neighbors[:3]:
                    print(f"      - {n['ecu_address']}: {n['ecu_name']}")
        else:
            print(f"  失败: {result.error_code}")
            print(f"    错误: {result.error_message}")
    
    print("\n" + "=" * 70)
    print("测试 3: ECU 跳数查询")
    print("=" * 70)
    
    test_pairs = [
        ("0019", "0019"),
        ("0019", "07E0"),
        ("0019", "8119"),
        ("0019", "0075"),
        ("0001", "0075"),
        ("0019", "NOTEXIST"),
        ("NOTEXIST", "0001"),
    ]
    for source, target in test_pairs:
        print(f"\n查询 {source} -> {target}:")
        result = check_ecu_hop_distance(source, target)
        if result.success:
            print(f"  成功，跳数: {result.hop_distance}")
            print(f"  连接类型: {result.connection_type}")
            path_str = " -> ".join([f"{p.ecu_address}({p.ecu_name[:15]})" for p in result.path])
            print(f"  路径: {path_str}")
        else:
            print(f"  失败: {result.error_code}")
            print(f"    错误: {result.error_message}")
            print(f"    解决: {result.solution}")
    
    print("\n" + "=" * 70)
    print("所有测试完成")
    print("=" * 70)
