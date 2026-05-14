"""Debug Surveyor issue - more detailed"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set env var BEFORE any import
topology_db = os.path.abspath("vehicle_info/vehicle_architecture.db")
os.environ["TOPOLOGY_DB_PATH"] = str(topology_db)

# Reset connection pool singleton
from app.topology.database import ConnectionPool
ConnectionPool._instance = None

from app.topology.queries import get_sub_topology, get_ecu
import traceback

# Test all 12 focused ECU IDs from the actual run
# From the log: 12 focused DTCs
# Let's test some common ECU IDs
ecu_ids = ["0009", "0015", "0017", "0019", "0042", "0075", "008B", "00A5", "8103", "810C", "8124", "812C"]

for ecu_id in ecu_ids:
    try:
        ecu = get_ecu(ecu_id)
        if ecu:
            print(f"get_ecu({ecu_id}): {ecu.ecu_id} ({ecu.domain})")
        else:
            print(f"get_ecu({ecu_id}): None")
        
        subgraph = get_sub_topology(ecu_id, depth=1)
        print(f"  sub_topology: {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")
        
    except Exception as e:
        print(f"\nERROR testing {ecu_id}: {e}")
        traceback.print_exc()
        break
