"""Topology Hop Checker Tool

用于检查两个ECU之间需要几跳才能连接起来。

跳数定义：
- 0跳：ECU自身
- 1跳：同一组内的ECU（main_ecu + subecu_of_main_ecu 或 connected_ecu + subecu_of_connected_ecu）
- 2跳：直接网络连接的另一组ECU
- 3+跳：通过中间ECU组连接

数据结构说明：
- main_ecu + subecu_of_main_ecu: 真正在一起的ECU（同一模块/物理集成）
- connected_ecu + subecu_of_connected_ecu: 另一组真正在一起的ECU
- 这两组之间通过网络协议连接
"""

import sqlite3
from collections import deque
from typing import Optional
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent.parent / "vehicle_info" / "vehicle_architecture.db"


def parse_ecu_list(ecu_str) -> set:
    if not ecu_str or ecu_str == 'NULL':
        return set()
    ecu_str = str(ecu_str)
    return set(e.strip() for e in ecu_str.split(',') if e.strip() and e.strip() != 'NULL')


def build_ecu_groups() -> dict:
    """
    构建ECU组信息
    返回: {ecu_address: group_id}
    """
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    ecu_to_group = {}
    group_id = 0
    
    cursor.execute("""
        SELECT main_ecu, subecu_of_main_ecu, connected_ecu, subecu_of_connected_ecu 
        FROM topology
    """)
    
    processed_groups = set()
    
    for row in cursor.fetchall():
        main_ecu, subecu_of_main, connected_ecu, subecu_of_connected = row
        
        main_group = frozenset(parse_ecu_list(main_ecu) | parse_ecu_list(subecu_of_main))
        connected_group = frozenset(parse_ecu_list(connected_ecu) | parse_ecu_list(subecu_of_connected))
        
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


def build_group_adjacency() -> dict:
    """
    构建组之间的邻接关系
    返回: {frozenset(group_ecus): {connected_group: set(protocols)}}
    """
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    group_adjacency = {}
    
    cursor.execute("""
        SELECT main_ecu, subecu_of_main_ecu, connected_ecu, subecu_of_connected_ecu, connection_protocols 
        FROM topology
    """)
    
    for row in cursor.fetchall():
        main_ecu, subecu_of_main, connected_ecu, subecu_of_connected, protocols = row
        
        main_group = frozenset(parse_ecu_list(main_ecu) | parse_ecu_list(subecu_of_main))
        connected_group = frozenset(parse_ecu_list(connected_ecu) | parse_ecu_list(subecu_of_connected))
        
        if main_group and main_group not in group_adjacency:
            group_adjacency[main_group] = {}
        if connected_group and connected_group not in group_adjacency:
            group_adjacency[connected_group] = {}
        
        if main_group and connected_group:
            protocol_set = {p.strip() for p in str(protocols or "").split(",") if p.strip()}
            existing_main = set(group_adjacency[main_group].get(connected_group, set()))
            existing_main.update(protocol_set)
            group_adjacency[main_group][connected_group] = existing_main
            existing_connected = set(group_adjacency[connected_group].get(main_group, set()))
            existing_connected.update(protocol_set)
            group_adjacency[connected_group][main_group] = existing_connected
    
    conn.close()
    return group_adjacency


def check_ecu_hop_distance(source_ecu: str, target_ecu: str, max_hops: int = 10) -> dict:
    """
    检查两个ECU之间的跳数距离
    
    跳数定义：
    - 0跳：ECU自身
    - 1跳：同一组内的ECU（真正在一起的ECU）
    - 2跳：直接网络连接的另一组ECU
    - 3+跳：通过中间ECU组连接
    
    Args:
        source_ecu: 源ECU地址
        target_ecu: 目标ECU地址
        max_hops: 最大搜索跳数（防止无限搜索）
        
    Returns:
        {
            'source_ecu': str,
            'target_ecu': str,
            'hop_distance': int,  # -1表示不可达
            'path': list,  # 最短路径
            'connection_type': str,  # 'self', 'same_group', 'direct_connection', 'multi_hop'
            'reachable': bool,
            'message': str
        }
    """
    source_ecu = source_ecu.upper().strip()
    target_ecu = target_ecu.upper().strip()
    
    if source_ecu == target_ecu:
        return {
            'source_ecu': source_ecu,
            'target_ecu': target_ecu,
            'hop_distance': 0,
            'path': [source_ecu],
            'connection_type': 'self',
            'reachable': True,
            'message': f'ECU {source_ecu} 到自身为 0 跳'
        }
    
    ecu_to_group = build_ecu_groups()
    
    if source_ecu not in ecu_to_group:
        return {
            'source_ecu': source_ecu,
            'target_ecu': target_ecu,
            'hop_distance': -1,
            'path': [],
            'connection_type': 'not_found',
            'reachable': False,
            'message': f'源ECU {source_ecu} 不在拓扑数据库中'
        }
    
    if target_ecu not in ecu_to_group:
        return {
            'source_ecu': source_ecu,
            'target_ecu': target_ecu,
            'hop_distance': -1,
            'path': [],
            'connection_type': 'not_found',
            'reachable': False,
            'message': f'目标ECU {target_ecu} 不在拓扑数据库中'
        }
    
    source_group = ecu_to_group[source_ecu]
    target_group = ecu_to_group[target_ecu]
    
    if source_group == target_group:
        group_ecus = sorted(list(source_group))
        return {
            'source_ecu': source_ecu,
            'target_ecu': target_ecu,
            'hop_distance': 1,
            'path': [source_ecu, target_ecu],
            'connection_type': 'same_group',
            'reachable': True,
            'group_members': group_ecus,
            'message': f'ECU {source_ecu} 和 {target_ecu} 在同一组内，为 1 跳（组内成员: {", ".join(group_ecus)}）'
        }
    
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
                    
                    ecu_path = [source_ecu]
                    for i, grp in enumerate(group_path):
                        if i == 0:
                            continue
                        representative = min(grp)
                        ecu_path.append(representative)
                    ecu_path.append(target_ecu)
                    
                    connection_type = 'direct_connection' if hops == 2 else 'multi_hop'
                    path_edges = []
                    for i in range(1, len(group_path)):
                        prev_group = group_path[i - 1]
                        curr_group = group_path[i]
                        path_edges.append(
                            {
                                "from_ecu": min(prev_group),
                                "to_ecu": min(curr_group),
                                "protocols": sorted(list(group_adjacency.get(prev_group, {}).get(curr_group, set()))),
                            }
                        )
                    
                    return {
                        'source_ecu': source_ecu,
                        'target_ecu': target_ecu,
                        'hop_distance': hops,
                        'path': ecu_path,
                        'path_edges': path_edges,
                        'connection_type': connection_type,
                        'reachable': True,
                        'message': f'ECU {source_ecu} 到 {target_ecu} 需要 {hops} 跳'
                    }
                
                queue.append((neighbor_group, hops + 1))
    
    return {
        'source_ecu': source_ecu,
        'target_ecu': target_ecu,
        'hop_distance': -1,
        'path': [],
        'connection_type': 'unreachable',
        'reachable': False,
        'message': f'ECU {source_ecu} 和 {target_ecu} 在 {max_hops} 跳内不可达'
    }


def get_ecu_group_info(ecu_address: str) -> dict:
    """
    获取ECU所在的组信息
    
    Args:
        ecu_address: ECU地址
        
    Returns:
        {
            'ecu_address': str,
            'group_members': list,
            'connected_groups': list,
            'message': str
        }
    """
    ecu_address = ecu_address.upper().strip()
    ecu_to_group = build_ecu_groups()
    
    if ecu_address not in ecu_to_group:
        return {
            'ecu_address': ecu_address,
            'group_members': [],
            'connected_groups': [],
            'message': f'ECU {ecu_address} 不在拓扑数据库中'
        }
    
    group = ecu_to_group[ecu_address]
    group_members = sorted(list(group))
    
    group_adjacency = build_group_adjacency()
    connected_groups = []
    
    for connected_group, protocols in group_adjacency.get(group, {}).items():
        connected_groups.append(
            {
                "ecus": sorted(list(connected_group)),
                "protocols": sorted(list(protocols)),
            }
        )
    
    return {
        'ecu_address': ecu_address,
        'group_members': group_members,
        'connected_groups': connected_groups,
        'message': f'ECU {ecu_address} 所在组有 {len(group_members)} 个成员，连接了 {len(connected_groups)} 个其他组'
    }


def get_ecu_neighbors(ecu_address: str, hop_distance: int = 2) -> dict:
    """
    获取指定ECU在N跳范围内的所有邻居ECU
    
    Args:
        ecu_address: ECU地址
        hop_distance: 跳数范围
        
    Returns:
        {
            'ecu_address': str,
            'hop_distance': int,
            'neighbors_by_hop': {1: [...], 2: [...], ...},
            'total_neighbors': int
        }
    """
    ecu_address = ecu_address.upper().strip()
    ecu_to_group = build_ecu_groups()
    
    if ecu_address not in ecu_to_group:
        return {
            'ecu_address': ecu_address,
            'hop_distance': hop_distance,
            'neighbors_by_hop': {},
            'total_neighbors': 0,
            'message': f'ECU {ecu_address} 不在拓扑数据库中'
        }
    
    source_group = ecu_to_group[ecu_address]
    group_adjacency = build_group_adjacency()
    
    neighbors_by_hop = {}
    
    group_members = source_group - {ecu_address}
    if group_members:
        neighbors_by_hop[1] = [{"ecu_address": ecu, "via_protocols": []} for ecu in sorted(list(group_members))]
    
    visited_groups = {source_group}
    current_groups = {source_group}
    
    for hop in range(2, hop_distance + 1):
        next_groups = set()
        hop_neighbors = set()
        
        for grp in current_groups:
            for neighbor_group, protocols in group_adjacency.get(grp, {}).items():
                if neighbor_group not in visited_groups:
                    next_groups.add(neighbor_group)
                    visited_groups.add(neighbor_group)
                    for ecu in neighbor_group:
                        hop_neighbors.add((ecu, tuple(sorted(list(protocols)))))
        
        if hop_neighbors:
            neighbors_by_hop[hop] = [
                {"ecu_address": item[0], "via_protocols": list(item[1])}
                for item in sorted(list(hop_neighbors), key=lambda x: x[0])
            ]
        
        current_groups = next_groups
        
        if not current_groups:
            break
    
    total = sum(len(v) for v in neighbors_by_hop.values())
    
    return {
        'ecu_address': ecu_address,
        'hop_distance': hop_distance,
        'neighbors_by_hop': neighbors_by_hop,
        'total_neighbors': total,
        'message': f'ECU {ecu_address} 在 {hop_distance} 跳范围内共有 {total} 个邻居ECU'
    }


def register_tools(registry):
    """注册工具到注册表"""
    registry.register(
        name="check_ecu_hop_distance",
        handler=check_ecu_hop_distance,
        description="检查两个ECU之间需要几跳才能连接起来。0跳=自身，1跳=同组ECU（真正在一起），2跳=直接网络连接，3+跳=通过中间组连接。",
        input_schema={
            "type": "object",
            "properties": {
                "source_ecu": {
                    "type": "string",
                    "description": "源ECU地址，如 '0019', '0001'"
                },
                "target_ecu": {
                    "type": "string",
                    "description": "目标ECU地址，如 '0075', '8103'"
                },
                "max_hops": {
                    "type": "integer",
                    "description": "最大搜索跳数，默认10",
                    "default": 10
                }
            },
            "required": ["source_ecu", "target_ecu"]
        },
        disclosure_level=2
    )
    
    registry.register(
        name="get_ecu_group_info",
        handler=get_ecu_group_info,
        description="获取ECU所在的组信息，包括同组成员和连接的其他组。",
        input_schema={
            "type": "object",
            "properties": {
                "ecu_address": {
                    "type": "string",
                    "description": "ECU地址，如 '0019', '0001'"
                }
            },
            "required": ["ecu_address"]
        },
        disclosure_level=2
    )
    
    registry.register(
        name="get_ecu_neighbors",
        handler=get_ecu_neighbors,
        description="获取指定ECU在N跳范围内的所有邻居ECU列表。",
        input_schema={
            "type": "object",
            "properties": {
                "ecu_address": {
                    "type": "string",
                    "description": "ECU地址，如 '0019', '0001'"
                },
                "hop_distance": {
                    "type": "integer",
                    "description": "跳数范围，默认2",
                    "default": 2
                }
            },
            "required": ["ecu_address"]
        },
        disclosure_level=3
    )


if __name__ == '__main__':
    print("=== 测试拓扑跳数检查工具（新跳数定义）===")
    print("\n跳数定义:")
    print("  0跳: ECU自身")
    print("  1跳: 同一组内的ECU（真正在一起的ECU）")
    print("  2跳: 直接网络连接的另一组ECU")
    print("  3+跳: 通过中间ECU组连接")
    
    test_cases = [
        ('0019', '0019'),
        ('0019', '07E0'),
        ('0019', '07E2'),
        ('0019', '8119'),
        ('0019', '0075'),
        ('07E0', '07E2'),
        ('07E0', '8119'),
        ('0001', '0075'),
    ]
    
    for source, target in test_cases:
        result = check_ecu_hop_distance(source, target)
        print(f"\n{source} -> {target}:")
        print(f"  跳数: {result['hop_distance']}")
        print(f"  连接类型: {result.get('connection_type', 'N/A')}")
        print(f"  可达: {result['reachable']}")
        if result['path']:
            print(f"  路径: {' -> '.join(result['path'])}")
        if 'group_members' in result:
            print(f"  组成员: {result['group_members']}")
    
    print("\n" + "="*60)
    print("=== 测试组信息查询 ===")
    
    group_info = get_ecu_group_info('0019')
    print(f"\nECU 0019 组信息:")
    print(f"  组成员: {group_info['group_members']}")
    print(f"  连接组数: {len(group_info['connected_groups'])}")
    
    print("\n" + "="*60)
    print("=== 测试邻居查询 ===")
    
    neighbors = get_ecu_neighbors('0019', hop_distance=3)
    print(f"\nECU 0019 的邻居:")
    for hop, ecus in neighbors['neighbors_by_hop'].items():
        print(f"  {hop}跳: {ecus[:10]}{'...' if len(ecus) > 10 else ''}")
    print(f"  总计: {neighbors['total_neighbors']} 个")
