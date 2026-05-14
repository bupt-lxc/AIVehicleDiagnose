from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol
import importlib.util
import json


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]
    
    def execute(self, **kwargs) -> Any: ...


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    handler: Callable[..., Any]
    input_schema: dict[str, Any]
    disclosure_level: int = 0
    dependencies: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass(slots=True)
class ToolRegistry:
    _tools: dict[str, ToolDefinition] = field(default_factory=dict)
    _disclosure_order: list[str] = field(default_factory=list)
    _plugin_paths: list[Path] = field(default_factory=list)
    _context: dict[str, Any] = field(default_factory=dict)
    
    def register(
        self,
        name: str,
        handler: Callable,
        description: str,
        input_schema: dict,
        disclosure_level: int = 0,
        dependencies: list[str] | None = None,
    ) -> None:
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            handler=handler,
            input_schema=input_schema,
            disclosure_level=disclosure_level,
            dependencies=dependencies or [],
        )
        self._update_disclosure_order()
    
    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            self._update_disclosure_order()
            return True
        return False
    
    def _update_disclosure_order(self) -> None:
        self._disclosure_order = sorted(
            self._tools.keys(),
            key=lambda x: self._tools[x].disclosure_level,
        )
    
    def get(self, name: str) -> Callable:
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"Tool not found: {name}")
        if not tool.enabled:
            raise RuntimeError(f"Tool is disabled: {name}")
        return tool.handler
    
    def get_definition(self, name: str) -> ToolDefinition:
        return self._tools.get(name)
    
    def set_context(self, key: str, value: Any) -> None:
        self._context[key] = value
    
    def get_context(self, key: str) -> Any:
        return self._context.get(key)
    
    def inject_context_to_handler(self, name: str, **context_overrides) -> Callable:
        import inspect
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"Tool not found: {name}")
        
        merged_context = {**self._context, **context_overrides}
        
        def wrapped_handler(**kwargs):
            # Only pass context variables that the tool handler accepts
            sig = inspect.signature(tool.handler)
            accepted_params = sig.parameters.keys()
            filtered_context = {k: v for k, v in merged_context.items() if k in accepted_params}
            return tool.handler(**kwargs, **filtered_context)
        
        wrapped_handler.__name__ = name
        return wrapped_handler
    
    def list_openai_tools(
        self,
        max_disclosure_level: int | None = None,
        selected_names: list[str] | None = None,
    ) -> list[dict]:
        tools = []
        
        for name in self._disclosure_order:
            if selected_names and name not in selected_names:
                continue
            
            tool = self._tools[name]
            
            if max_disclosure_level is not None:
                if tool.disclosure_level > max_disclosure_level:
                    continue
            
            if not tool.enabled:
                continue
            
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            })
        
        return tools
    
    def get_disclosure_snapshot(self, level: int) -> dict[str, Any]:
        return {
            "level": level,
            "available_tools": [
                name for name in self._disclosure_order
                if self._tools[name].disclosure_level <= level
            ],
            "tool_count": sum(
                1 for t in self._tools.values()
                if t.disclosure_level <= level and t.enabled
            ),
        }
    
    def load_plugins(self, plugin_paths: list[str | Path]) -> None:
        for path_str in plugin_paths:
            path = Path(path_str)
            self._plugin_paths.append(path)
            
            if path.is_file() and path.suffix == ".py":
                self._load_tool_file(path)
            elif path.is_dir():
                for py_file in path.glob("*.py"):
                    self._load_tool_file(py_file)
    
    def _load_tool_file(self, path: Path) -> None:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, "register_tools"):
                module.register_tools(self)
    
    def enable_tool(self, name: str) -> None:
        if name in self._tools:
            self._tools[name].enabled = True
    
    def disable_tool(self, name: str) -> None:
        if name in self._tools:
            self._tools[name].enabled = False
    
    def monitor_snapshot(self) -> dict[str, Any]:
        return {
            "total_tools": len(self._tools),
            "enabled_tools": sum(1 for t in self._tools.values() if t.enabled),
            "disclosure_levels": {
                level: sum(1 for t in self._tools.values() if t.disclosure_level == level)
                for level in set(t.disclosure_level for t in self._tools.values())
            },
            "plugin_paths": [str(p) for p in self._plugin_paths],
        }


@dataclass(slots=True)
class ProgressiveDisclosureController:
    _tool_registry: ToolRegistry
    _current_level: int = 0
    _level_thresholds: dict[int, dict] = field(default_factory=dict)
    
    def __post_init__(self):
        self._level_thresholds = {
            0: {"max_confidence": 1.0, "description": "收敛验证工具"},
            1: {"max_confidence": 0.75, "description": "扩展检索工具"},
            2: {"max_confidence": 0.55, "description": "拓扑分析工具"},
            3: {"max_confidence": 0.35, "description": "深度探索工具"},
        }
    
    def update_level(self, confidence: float, state: dict) -> int:
        bounded = max(0.0, min(1.0, float(confidence or 0.0)))
        new_level = 0
        if bounded <= self._level_thresholds[3]["max_confidence"]:
            new_level = 3
        elif bounded <= self._level_thresholds[2]["max_confidence"]:
            new_level = 2
        elif bounded <= self._level_thresholds[1]["max_confidence"]:
            new_level = 1
        else:
            new_level = 0
        
        if new_level != self._current_level:
            self._current_level = new_level
        
        return self._current_level
    
    def get_available_tools(self) -> list[dict]:
        return self._tool_registry.list_openai_tools(
            max_disclosure_level=self._current_level
        )
    
    def get_current_snapshot(self) -> dict[str, Any]:
        return {
            "current_level": self._current_level,
            "level_description": self._level_thresholds[self._current_level]["description"],
            "tools": self._tool_registry.get_disclosure_snapshot(self._current_level),
        }
