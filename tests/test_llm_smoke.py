"""Smoke test for the LLM interface — confirms the API key works end to end.

Asks one obvious causal question ("Does smoking cause lung cancer?") using
Asia-style wording and checks the model votes 1 (Yes). This makes a real API
call, so it is skipped automatically when no OPENAI_API_KEY is configured.

Run from the repo root:
    python -m unittest tests.test_llm_smoke -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

from src.llm import interface  # noqa: E402

load_dotenv(interface.ENV_PATH)


@unittest.skipUnless(
    os.getenv("OPENAI_API_KEY"),
    "OPENAI_API_KEY not set (copy .env.example to .env) — skipping live test",
)
class TestLLMSmoke(unittest.TestCase):
    def test_obvious_causal_question(self):
        vote = interface.llm_vote("Does smoking cause lung cancer?")
        self.assertEqual(vote, 1, "expected the model to answer Yes (1)")


if __name__ == "__main__":
    unittest.main()
