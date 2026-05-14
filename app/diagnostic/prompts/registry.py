from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol
import importlib.util
import json


class PromptTemplate(Protocol):
    name: str
    version: str
    system_prompt: str
    user_prompt_template: str
    
    def render(self, **kwargs) -> str: ...


@dataclass(slots=True)
class PromptRegistry:
    _prompts: dict[str, PromptTemplate] = field(default_factory=dict)
    _versions: dict[str, list[str]] = field(default_factory=dict)
    _active_versions: dict[str, str] = field(default_factory=dict)
    _plugin_paths: list[Path] = field(default_factory=list)
    
    def register(self, prompt: PromptTemplate, set_active: bool = True) -> None:
        key = prompt.name
        version = prompt.version
        
        full_key = f"{key}@{version}"
        self._prompts[full_key] = prompt
        
        if key not in self._versions:
            self._versions[key] = []
        self._versions[key].append(version)
        
        if set_active:
            self._active_versions[key] = version
    
    def get(
        self,
        name: str,
        version: str | None = None,
        **inject_vars,
    ) -> str:
        if version:
            full_key = f"{name}@{version}"
        else:
            active_version = self._active_versions.get(name)
            if not active_version:
                raise KeyError(f"No active version for prompt: {name}")
            full_key = f"{name}@{active_version}"
        
        prompt = self._prompts.get(full_key)
        if not prompt:
            raise KeyError(f"Prompt not found: {full_key}")
        
        return prompt.render(**inject_vars)
    
    def get_system_prompt(self, name: str, version: str | None = None) -> str:
        if version:
            full_key = f"{name}@{version}"
        else:
            active_version = self._active_versions.get(name)
            if not active_version:
                 # Fallback logic or raise error
                 # If we have any version, maybe pick latest?
                 # For now, let's be strict but clear
                 raise KeyError(f"No active version defined for prompt: {name}")
            full_key = f"{name}@{active_version}"
        
        prompt = self._prompts.get(full_key)
        if not prompt:
            raise KeyError(f"Prompt not found: {full_key}")
        
        return prompt.system_prompt
    
    def load_plugins(self, plugin_paths: list[str | Path]) -> None:
        for path_str in plugin_paths:
            path = Path(path_str)
            self._plugin_paths.append(path)
            
            if path.is_file() and path.suffix == ".py":
                self._load_prompt_file(path)
            elif path.is_dir():
                for py_file in path.glob("*.py"):
                    self._load_prompt_file(py_file)
    
    def _load_prompt_file(self, path: Path) -> None:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, PromptTemplate):
                    self.register(attr)
    
    def list_available(self) -> dict[str, list[str]]:
        return dict(self._versions)
    
    def set_active_version(self, name: str, version: str) -> None:
        if name not in self._versions:
            raise KeyError(f"Prompt not found: {name}")
        if version not in self._versions[name]:
            raise ValueError(f"Version {version} not found for prompt {name}")
        self._active_versions[name] = version


@dataclass(slots=True)
class DynamicPromptTemplate:
    name: str
    version: str
    system_prompt: str
    user_prompt_template: str
    injectable_vars: list[str] = field(default_factory=list)
    validators: dict[str, Callable[[Any], bool]] = field(default_factory=dict)
    
    def render(self, **kwargs) -> str:
        self._validate_injection_vars(kwargs)
        return self.user_prompt_template.format(**kwargs)
    
    def _validate_injection_vars(self, vars: dict[str, Any]) -> None:
        for var_name in self.injectable_vars:
            if var_name not in vars:
                raise ValueError(f"Missing required injection variable: {var_name}")
            
            if var_name in self.validators:
                if not self.validators[var_name](vars[var_name]):
                    raise ValueError(f"Validation failed for variable: {var_name}")
