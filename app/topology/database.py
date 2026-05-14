import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager
import threading
from queue import Queue, Empty

def _get_db_path_env() -> str:
    """Get topology DB path from environment, read at call time not import time"""
    return os.getenv("TOPOLOGY_DB_PATH", "./logs/topology.sqlite")

def get_db_path() -> Path:
    path = Path(_get_db_path_env())
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

class ConnectionPool:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, max_connections=10):
        self.pool = Queue(maxsize=max_connections)
        self.max_connections = max_connections
        for _ in range(max_connections):
            conn = sqlite3.connect(get_db_path(), check_same_thread=False, timeout=15.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            self.pool.put(conn)
            
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
        
    def acquire(self):
        try:
            return self.pool.get(timeout=15.0)
        except Empty:
            raise RuntimeError("Database connection pool exhausted")
            
    def release(self, conn):
        self.pool.put(conn)

@contextmanager
def get_db_connection():
    """
    提供一个上下文管理器，用于获取 SQLite 数据库连接。
    在结束时自动提交并释放连接回连接池。
    """
    import time
    pool = ConnectionPool.get_instance()
    
    # Retry logic for acquiring connection
    max_retries = 3
    conn = None
    last_error = None
    
    for attempt in range(max_retries):
        try:
            conn = pool.acquire()
            break
        except RuntimeError as e:
            if "pool exhausted" in str(e):
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
            raise
            
    if conn is None:
        raise last_error or RuntimeError("Failed to acquire database connection")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.release(conn)

def init_db():
    """
    初始化拓扑数据库的表结构。
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 创建 ECU 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ecu (
                ecu_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                power_segment TEXT NOT NULL,
                is_gateway BOOLEAN NOT NULL DEFAULT 0
            )
        """)
        
        # 创建 Connection 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS connection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                protocol TEXT NOT NULL,
                gateway_id TEXT,
                FOREIGN KEY(source_id) REFERENCES ecu(ecu_id),
                FOREIGN KEY(target_id) REFERENCES ecu(ecu_id),
                FOREIGN KEY(gateway_id) REFERENCES ecu(ecu_id)
            )
        """)

if __name__ == "__main__":
    init_db()
