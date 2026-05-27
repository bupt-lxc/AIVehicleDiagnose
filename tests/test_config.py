from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_deepseek_api_key_alias_is_used(self) -> None:
        env = {key: value for key, value in os.environ.items() if key not in {"ONLINE_LLM_API_KEY", "DASHSCOPE_API_KEY"}}
        env["DEEPSEEK_API_KEY"] = "deepseek-test-key"
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings()

        self.assertEqual(settings.online_profile.api_key, "deepseek-test-key")
        self.assertEqual(settings.default_profile, "online")


if __name__ == "__main__":
    unittest.main()
