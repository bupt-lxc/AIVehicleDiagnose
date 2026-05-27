import unittest
from unittest.mock import patch
from unittest.mock import Mock

import httpx

from app.config import LLMProfile
from app.providers import OpenAICompatibleProvider


class ProviderErrorDetailTests(unittest.TestCase):
    def test_http_status_error_includes_response_body_preview(self):
        profile = LLMProfile(name="online", base_url="https://api.example.test", api_key="x", model="demo")
        provider = OpenAICompatibleProvider(profile)
        response = Mock()
        response.status_code = 400
        response.text = '{"error":{"message":"tools is not supported"}}'
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "bad request",
            request=Mock(),
            response=response,
        )
        provider._client = Mock()
        provider._client.post.return_value = response

        with self.assertRaisesRegex(RuntimeError, "HTTP Error 400: .*tools is not supported"):
            provider.generate([{"role": "user", "content": "hello"}])

    def test_deepseek_profile_ignores_qwen_fallback_models(self):
        profile = LLMProfile(name="online", base_url="https://api.deepseek.com", api_key="x", model="deepseek-v4-pro")
        provider = OpenAICompatibleProvider(profile)

        with patch.dict("os.environ", {"ONLINE_LLM_FALLBACK_MODELS": "qwen-plus,qwen-turbo,deepseek-v4-flash"}):
            self.assertEqual(provider._candidate_models(), ["deepseek-v4-pro", "deepseek-v4-flash"])

    def test_normalize_messages_preserves_assistant_reasoning_content(self):
        profile = LLMProfile(name="online", base_url="https://api.deepseek.com", api_key="x", model="deepseek-v4-pro")
        provider = OpenAICompatibleProvider(profile)

        normalized = provider._normalize_messages(
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "Thought: inspect\nAction: query_dtc",
                    "reasoning_content": "private thinking payload",
                },
            ]
        )

        self.assertEqual(normalized[-1]["reasoning_content"], "private thinking payload")


if __name__ == "__main__":
    unittest.main()
