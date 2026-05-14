import sqlite3

print("=== dtcs_complete.db ===")
conn = sqlite3.connect("vehicle_info/dtcs_complete.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", tables)

for table in ['ecus', 'dtcs', 'sdgs']:
    print(f"\n--- {table} ---")
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    print(f"  Count: {cursor.fetchone()[0]}")
    cursor.execute(f"PRAGMA table_info({table})")
    cols = cursor.fetchall()
    print("  Schema:")
    for c in cols:
        print(f"    {c[1]} ({c[2]})")
    cursor.execute(f"SELECT * FROM {table} LIMIT 3")
    for r in cursor.fetchall():
        print(f"  Row: {r}")

conn.close()