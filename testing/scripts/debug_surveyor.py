"""Debug Surveyor issue"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set env var BEFORE any import
topology_db = os.path.abspath("vehicle_info/vehicle_architecture.db")
os.environ["TOPOLOGY_DB_PATH"] = str(topology_db)

# Reset connection pool singleton
from app.topology.database import ConnectionPool
ConnectionPool._instance = None

from app.diagnostic.agents.surveyor_agent import SurveyorAgent
from app.diagnostic.state import create_initial_state
from app.diagnostic.tools.xml_report_parser import parse_diagnostic_report, convert_to_structured_data
from app.diagnostic.prioritizer import DTCSeverity

# Create a state with parsed data
xml_path = "testing/test1.xml"
parsed_report = parse_diagnostic_report(xml_path)
structured_data = convert_to_structured_data(parsed_report)

# Add priority scores
for dtc in structured_data["dtc_codes"]:
    dtc["priority_score"] = 0

state = create_initial_state(raw_input=xml_path, user_context="test")
state["structured_data"] = structured_data

print(f"ECU states count: {len(structured_data.get('ecu_states', []))}")
print(f"DTC codes count: {len(structured_data.get('dtc_codes', []))}")

# Check ECU ID types
for dtc in structured_data.get('dtc_codes', [])[:3]:
    print(f"  DTC ECU ID: {repr(dtc.get('ecu_id'))} (type: {type(dtc.get('ecu_id'))})")

for ecu in structured_data.get('ecu_states', [])[:3]:
    print(f"  ECU ID: {repr(ecu.get('ecu_id'))} (type: {type(ecu.get('ecu_id'))})")

# Try surveyor
surveyor = SurveyorAgent(config={"topology_depth": 1})
try:
    result = surveyor.survey(state)
    print(f"\nSurveyor result: {result.get('topology_graph')}")
except Exception as e:
    import traceback
    print(f"\nSurveyor failed: {e}")
    traceback.print_exc()
