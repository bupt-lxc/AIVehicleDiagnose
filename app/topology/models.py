from pydantic import BaseModel, Field
from typing import Optional, Literal, List

class ECUModel(BaseModel):
    """
    ECU 节点信息模型
    """
    ecu_id: str = Field(..., description="ECU 标识符")
    domain: str = Field(..., description="所属域，如 Powertrain/Chassis/Body/Infotainment/ADAS")
    power_segment: str = Field(..., description="电源段")
    is_gateway: bool = Field(default=False, description="是否为网关")

class ConnectionModel(BaseModel):
    """
    ECU 连接信息模型
    """
    source_id: str = Field(..., description="源 ECU ID")
    target_id: str = Field(..., description="目标 ECU ID")
    protocol: Literal["CAN", "LIN", "ETH", "FlexRay", "Unknown"] = Field(..., description="通信协议")
    gateway_id: Optional[str] = Field(default=None, description="经过的网关 ECU ID")

class TopologyGraphModel(BaseModel):
    """
    拓扑图信息模型
    """
    nodes: List[ECUModel] = Field(default_factory=list, description="节点列表")
    edges: List[ConnectionModel] = Field(default_factory=list, description="边列表")
    depth: int = Field(default=0, description="拓扑深度")
    center_ecu: str = Field(default="", description="中心 ECU ID")
