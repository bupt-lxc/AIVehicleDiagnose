
import sqlite3
import os

db_path = r"d:\AI Agent Projects\AI Disgnostic\vehicle_info\vehicle_architecture.db"

if not os.path.exists(db_path):
    print(f"Error: DB not found at {db_path}")
else:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables:", tables)
        
        # 如果有 ecu 表，查前几条
        if any('ecu' in t for t in tables):
            cursor.execute("SELECT * FROM ecu LIMIT 5")
            rows = cursor.fetchall()
            print("\nECU Sample:")
            for row in rows:
                print(row)
                
        conn.close()
    except Exception as e:
        print(f"Query failed: {e}")
