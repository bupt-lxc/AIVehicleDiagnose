from __future__ import annotations

from typing import Any
from loguru import logger

from app.diagnostic.state import DiagnosticState, TopologyGraph, TopologyNode, TopologyEdge
from app.topology.queries import get_sub_topology


class SurveyorAgent:
    NODE_NAME = "surveyor"
    DEFAULT_TOPOLOGY_DEPTH = 1
    MAX_TOPOLOGY_DEPTH = 5
    
    def __init__(self, config: dict | None = None):
        self._config = config or {}
        depth = self._config.get("topology_depth", self.DEFAULT_TOPOLOGY_DEPTH)
        self._topology_depth = min(depth, self.MAX_TOPOLOGY_DEPTH)
    
    def survey(self, state: DiagnosticState) -> dict:
        ecu_ids = self._extract_ecu_ids(
            state.get("structured_data", {}),
            state.get("focused_dtcs", []),
        )
        
        if not ecu_ids:
            logger.info("[SurveyorAgent] 无可用的 ECU ID 进行拓扑查询")
            return {
                "topology_graph": None,
                "current_node": self.NODE_NAME,
            }
        
        try:
            topology_graph = self._build_topology_graph(ecu_ids)
            logger.info(f"[SurveyorAgent] 成功构建拓扑图，节点数: {len(topology_graph.get('nodes', []))}")
            
            # Update ECU states with domain info
            structured_data = state.get("structured_data", {})
            ecu_states = structured_data.get("ecu_states", [])
            domain_map = {n["ecu_id"]: n["domain"] for n in topology_graph.get("nodes", [])}
            
            for ecu in ecu_states:
                if ecu.get("ecu_id") in domain_map:
                    ecu["domain"] = domain_map[ecu["ecu_id"]]
                    
            return {
                "topology_graph": topology_graph,
                "structured_data": structured_data,
                "current_node": self.NODE_NAME,
            }
        except Exception as e:
            logger.error(f"[SurveyorAgent] 构建拓扑图失败: {e}")
            return {
                "topology_graph": None,
                "current_node": self.NODE_NAME,
                "error_message": f"构建拓扑图失败: {str(e)}",
                "exit_code": 1  # Indicate failure
            }
    
    def _extract_ecu_ids(self, structured_data: dict, focused_dtcs: list[dict] | None = None) -> list[str]:
        ecu_ids = set()
        
        # Extract from DTCs
        for dtc in (focused_dtcs or structured_data.get("dtc_codes", [])):
            if dtc.get("ecu_id"):
                ecu_ids.add(str(dtc["ecu_id"]))
        
        # Extract from ECU states
        for ecu in structured_data.get("ecu_states", []):
            if ecu.get("ecu_id"):
                ecu_ids.add(str(ecu["ecu_id"]))
                
        return list(ecu_ids)
    
    def _build_topology_graph(self, ecu_ids: list[str]) -> TopologyGraph:
        all_nodes = {}
        all_edges = []
        
        for ecu_id in ecu_ids:
            subgraph_model = get_sub_topology(ecu_id, depth=self._topology_depth)
            if not subgraph_model:
                continue
                
            # Convert models to dictionaries
            for node in subgraph_model.nodes:
                all_nodes[node.ecu_id] = {
                    "ecu_id": node.ecu_id,
                    "domain": node.domain,
                    "power_segment": node.power_segment,
                    "gateway": node.is_gateway
                }
                
            for edge in subgraph_model.edges:
                all_edges.append({
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "protocol": edge.protocol,
                    "gateway_id": edge.gateway_id
                })
        
        nodes_list = list(all_nodes.values())
        edges_list = self._deduplicate_edges(all_edges)
        
        return {
            "nodes": nodes_list,
            "edges": edges_list,
            "depth": self._topology_depth,
            "center_ecu": ecu_ids[0] if ecu_ids else ""
        }
        
    def _deduplicate_edges(self, edges: list[dict]) -> list[dict]:
        unique_edges = []
        seen = set()
        for edge in edges:
            identifier = f"{edge['source_id']}-{edge['target_id']}-{edge['protocol']}"
            if identifier not in seen:
                seen.add(identifier)
                unique_edges.append(edge)
        return unique_edges
