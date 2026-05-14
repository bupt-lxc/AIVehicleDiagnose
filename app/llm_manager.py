from __future__ import annotations

import logging
from typing import Dict

from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import SecretStr

from .config import LLMProfile, Settings


class LLMManager:
    def __init__(self, settings: Settings) -> None:
        self._profiles: Dict[str, LLMProfile] = {
            "local": settings.local_profile,
            "online": settings.online_profile,
        }
        self._active_profile = settings.default_profile if settings.default_profile in self._profiles else "local"
        logger.info("LLM profile initialized: %s", self._active_profile)

    @property
    def active_profile_name(self) -> str:
        return self._active_profile

    @property
    def active_profile(self) -> LLMProfile:
        return self._profiles[self._active_profile]

    def switch_profile(self, profile_name: str, model: str | None = None) -> LLMProfile:
        normalized = profile_name.lower().strip()
        if normalized not in self._profiles:
            raise ValueError(f"Unsupported profile: {profile_name}")
        self._active_profile = normalized
        if model:
            self._profiles[normalized].model = model.strip()
        logger.info(
            "LLM profile switched: profile=%s model=%s base_url=%s",
            self._active_profile,
            self._profiles[normalized].model,
            self._profiles[normalized].base_url,
        )
        return self._profiles[normalized]

    def get_llm(self, temperature: float = 0.1) -> ChatOpenAI:
        profile = self.active_profile
        if not profile.api_key:
            raise ValueError(f"API key is empty for profile {profile.name}")
        return ChatOpenAI(
            model=profile.model,
            api_key=SecretStr(profile.api_key),
            base_url=profile.base_url,
            temperature=temperature,
        )
