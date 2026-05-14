# app/diagnostic/config.py

from dataclasses import dataclass
from app.config import Settings

@dataclass
class DiagnosticConfig:
    enable_local_llm: bool = True
    enable_cloud_fallback: bool = True
    enable_hitl: bool = True
    topology_depth: int = 2
    min_confidence: float = 0.7
    max_retries: int = 3
    
    @classmethod
    def from_settings(cls, settings: Settings) -> "DiagnosticConfig":
        return cls(
            enable_local_llm=bool(settings.local_profile.api_key),
            enable_cloud_fallback=bool(settings.online_profile.api_key),
        )
