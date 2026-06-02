from __future__ import annotations

import unittest

from aviation_rag.config import Settings
from aviation_rag.runtime import build_run_state


class RuntimeStateTests(unittest.TestCase):
    def test_build_run_state_with_overrides(self):
        settings = Settings(default_top_k=10, default_strategy="hybrid")
        state = build_run_state(
            settings,
            query_raw="engine warning",
            top_k=5,
            strategy="metadata_first",
            allow_local_fallback=False,
            write_phase1_artifact=True,
        )
        self.assertEqual(state["query_raw"], "engine warning")
        self.assertFalse(state["allow_local_fallback"])
        self.assertTrue(state["write_phase1_artifact"])
        self.assertEqual(state["retrieval_plan_override"]["top_k"], 5)
        self.assertEqual(state["retrieval_plan_override"]["strategy"], "metadata_first")

    def test_build_run_state_uses_defaults(self):
        settings = Settings(default_top_k=10, default_strategy="hybrid")
        state = build_run_state(settings, query_raw="weather turbulence")
        self.assertNotIn("retrieval_plan_override", state)


if __name__ == "__main__":
    unittest.main()
