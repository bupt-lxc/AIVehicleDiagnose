from typing import List, Optional
from .database import get_db_connection
from .models import ECUModel, ConnectionModel, TopologyGraphModel


def _row_to_ecu(row: dict) -> ECUModel:
    """将实际数据库行转换为 ECUModel"""
    ecu_id = row.get("ecu_address") or row.get("ecu_id") or ""
    domain = row.get("ecu_domain") or row.get("domain") or "Unknown"
    power_segment = row.get("power_segment") or ""
    is_gateway = bool(row.get("is_gateway") or 0)
    return ECUModel(ecu_id=ecu_id, domain=domain, power_segment=power_segment, is_gateway=is_gateway)


def _row_to_connection(row: dict) -> ConnectionModel:
    """将实际数据库行转换为 ConnectionModel"""
    # 适配两种表结构
    if "source_id" in row:
        source = row["source_id"]
        target = row["target_id"]
        protocol = row.get("protocol", "Unknown")
        gateway = row.get("gateway_id")
    else:
        # 使用 topology 表
        source = row.get("main_ecu") or row.get("source_id")
        target = row.get("connected_ecu") or row.get("target_id")
        protocol = row.get("connection_protocols") or row.get("protocol", "Unknown")
        gateway = None
    
    # 统一转为字符串
    source = str(source).strip() if source else ""
    target = str(target).strip() if target else ""
    
    # 规范化协议名
    protocol_lower = (protocol or "Unknown").lower()
    if "canfd" in protocol_lower:
        protocol = "CAN"
    elif "can" in protocol_lower:
        protocol = "CAN"
    elif "lin" in protocol_lower:
        protocol = "LIN"
    elif "eth" in protocol_lower:
        protocol = "ETH"
    elif "flexray" in protocol_lower:
        protocol = "FlexRay"
    else:
        protocol = "Unknown"
    
    return ConnectionModel(
        source_id=source,
        target_id=target,
        protocol=protocol,
        gateway_id=str(gateway) if gateway else None
    )


def get_ecu(ecu_id: str) -> Optional[ECUModel]:
    """获取指定 ECU 的信息"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ecu_info WHERE ecu_address = ?", (ecu_id,))
        row = cursor.fetchone()
        if row:
            return _row_to_ecu(dict(row))
        # Fallback: try old table name
        try:
            cursor.execute("SELECT * FROM ecu WHERE ecu_id = ?", (ecu_id,))
            row = cursor.fetchone()
            if row:
                return _row_to_ecu(dict(row))
        except Exception:
            pass
        return None


def get_upstream_ecus(ecu_id: str) -> List[ECUModel]:
    """获取上游 ECU"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Try topology table first
        try:
            cursor.execute("""
                SELECT DISTINCT e.* FROM ecu_info e
                JOIN topology t ON e.ecu_address = t.main_ecu
                WHERE t.connected_ecu = ?
            """, (ecu_id,))
            rows = cursor.fetchall()
            if rows:
                return [_row_to_ecu(dict(r)) for r in rows]
        except Exception:
            pass
        
        # Fallback to old schema
        try:
            cursor.execute("""
                SELECT e.* FROM ecu e
                JOIN connection c ON e.ecu_id = c.source_id
                WHERE c.target_id = ?
            """, (ecu_id,))
            return [_row_to_ecu(dict(r)) for r in cursor.fetchall()]
        except Exception:
            return []


def get_downstream_ecus(ecu_id: str) -> List[ECUModel]:
    """获取下游 ECU"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Try topology table first
        try:
            cursor.execute("""
                SELECT DISTINCT e.* FROM ecu_info e
                JOIN topology t ON e.ecu_address = t.connected_ecu
                WHERE t.main_ecu = ?
            """, (ecu_id,))
            rows = cursor.fetchall()
            if rows:
                return [_row_to_ecu(dict(r)) for r in rows]
        except Exception:
            pass
        
        # Fallback to old schema
        try:
            cursor.execute("""
                SELECT e.* FROM ecu e
                JOIN connection c ON e.ecu_id = c.target_id
                WHERE c.source_id = ?
            """, (ecu_id,))
            return [_row_to_ecu(dict(r)) for r in cursor.fetchall()]
        except Exception:
            return []


from collections import deque


def get_sub_topology(center_ecu_id: str, depth: int = 2) -> TopologyGraphModel:
    """获取指定深度的 ECU 子拓扑结构"""
    nodes = {}
    edges = []
    visited = set()
    MAX_VISITED_NODES = 1000
    
    queue = deque([(center_ecu_id, 0)])
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Detect which table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cursor.fetchall()}
        has_new = "ecu_info" in tables and "topology" in tables
        
        while queue and len(visited) < MAX_VISITED_NODES:
            current_ecu, current_depth = queue.popleft()
            
            if current_ecu in visited or current_depth > depth:
                continue
                
            visited.add(current_ecu)
            
            # Query ECU
            if has_new:
                cursor.execute("SELECT * FROM ecu_info WHERE ecu_address = ?", (current_ecu,))
            else:
                cursor.execute("SELECT * FROM ecu WHERE ecu_id = ?", (current_ecu,))
            
            row = cursor.fetchone()
            if row:
                nodes[current_ecu] = _row_to_ecu(dict(row))
            
            if current_depth < depth:
                # Query connections
                if has_new:
                    cursor.execute(
                        "SELECT * FROM topology WHERE main_ecu = ? OR connected_ecu = ?",
                        (current_ecu, current_ecu)
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM connection WHERE source_id = ? OR target_id = ?",
                        (current_ecu, current_ecu)
                    )
                
                for row in cursor.fetchall():
                    edge = _row_to_connection(dict(row))
                    if edge.source_id and edge.target_id:
                        edges.append(edge)
                        
                        if edge.source_id not in visited:
                            queue.append((edge.source_id, current_depth + 1))
                        if edge.target_id not in visited:
                            queue.append((edge.target_id, current_depth + 1))
    
    # 去重 edges
    unique_edges = []
    seen = set()
    for edge in edges:
        identifier = f"{edge.source_id}-{edge.target_id}-{edge.protocol}"
        if identifier not in seen:
            seen.add(identifier)
            unique_edges.append(edge)
            
    return TopologyGraphModel(
        nodes=list(nodes.values()),
        edges=unique_edges,
        depth=depth,
        center_ecu=center_ecu_id
    )


def add_ecu(ecu: ECUModel) -> None:
    """添加 ECU 记录"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Try new table first
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO ecu_info (ecu_address, ecu_domain, ecu_name, ecu_description)
                VALUES (?, ?, ?, ?)
            """, (ecu.ecu_id, ecu.domain, "", ""))
        except Exception:
            cursor.execute("""
                INSERT OR REPLACE INTO ecu (ecu_id, domain, power_segment, is_gateway)
                VALUES (?, ?, ?, ?)
            """, (ecu.ecu_id, ecu.domain, ecu.power_segment, ecu.is_gateway))


def add_connection(connection: ConnectionModel) -> None:
    """添加连接记录"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Try new table first
        try:
            cursor.execute("""
                INSERT INTO topology (main_ecu, connected_ecu, connection_protocols)
                VALUES (?, ?, ?)
            """, (connection.source_id, connection.target_id, connection.protocol))
        except Exception:
            cursor.execute("""
                INSERT INTO connection (source_id, target_id, protocol, gateway_id)
                VALUES (?, ?, ?, ?)
            """, (connection.source_id, connection.target_id, connection.protocol, connection.gateway_id))


def clear_db() -> None:
    """清理数据库"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for table in ["topology", "connection"]:
            try:
                cursor.execute(f"DELETE FROM {table}")
            except Exception:
                pass
        for table in ["ecu_info", "ecu"]:
            try:
                cursor.execute(f"DELETE FROM {table}")
            except Exception:
                pass
