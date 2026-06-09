from __future__ import annotations

import unittest

from aviation_rag.streamlit_bootstrap import (
    BOOTSTRAP_ENV_VAR,
    build_streamlit_command,
    should_bootstrap_streamlit,
)


class StreamlitBootstrapTests(unittest.TestCase):
    def test_bootstrap_needed_for_plain_python_launch(self):
        self.assertTrue(
            should_bootstrap_streamlit(
                module_name="__main__",
                has_streamlit_context=False,
                env={},
            )
        )

    def test_bootstrap_skips_when_already_in_streamlit(self):
        self.assertFalse(
            should_bootstrap_streamlit(
                module_name="__main__",
                has_streamlit_context=True,
                env={},
            )
        )

    def test_bootstrap_skips_after_relaunch_marker(self):
        self.assertFalse(
            should_bootstrap_streamlit(
                module_name="__main__",
                has_streamlit_context=False,
                env={BOOTSTRAP_ENV_VAR: "1"},
            )
        )

    def test_build_streamlit_command_preserves_user_args(self):
        command = build_streamlit_command(
            python_executable="python",
            script_path="streamlit_app.py",
            args=["--server.port", "8502"],
        )
        self.assertEqual(command[:4], ["python", "-m", "streamlit", "run"])
        self.assertTrue(command[4].endswith("streamlit_app.py"))
        self.assertEqual(command[5:], ["--server.port", "8502"])


if __name__ == "__main__":
    unittest.main()
