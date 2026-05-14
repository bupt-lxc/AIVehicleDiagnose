"""Debug topology query"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

topology_db = os.path.abspath("vehicle_info/vehicle_architecture.db")
os.environ["TOPOLOGY_DB_PATH"] = str(topology_db)

# Reset connection pool singleton
from app.topology.database import ConnectionPool
ConnectionPool._instance = None

from app.topology.queries import get_sub_topology, get_ecu
from app.topology.database import get_db_path

print(f"DB Path: {get_db_path()}")

# Test get_ecu
ecu = get_ecu("0015")
print(f"ECU 0015: {ecu}")

# Test get_sub_topology
subgraph = get_sub_topology("0015", depth=1)
print(f"Subgraph for 0015: nodes={len(subgraph.nodes)}, edges={len(subgraph.edges)}")
if subgraph.nodes:
    for n in subgraph.nodes[:5]:
        print(f"  Node: {n.ecu_id} ({n.domain})")
if subgraph.edges:
    for e in subgraph.edges[:5]:
        print(f"  Edge: {e.source_id} -> {e.target_id} ({e.protocol})")
