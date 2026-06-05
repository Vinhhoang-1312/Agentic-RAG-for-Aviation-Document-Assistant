from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

BOOTSTRAP_ENV_VAR = "AVIATION_RAG_STREAMLIT_BOOTSTRAPPED"


def should_bootstrap_streamlit(
    *,
    module_name: str,
    has_streamlit_context: bool,
    env: Mapping[str, str] | None = None,
) -> bool:
    current_env = env or os.environ
    return (
        module_name == "__main__"
        and not has_streamlit_context
        and current_env.get(BOOTSTRAP_ENV_VAR) != "1"
    )


def build_streamlit_command(
    *,
    python_executable: str,
    script_path: str,
    args: Sequence[str] | None = None,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "streamlit",
        "run",
        str(Path(script_path).resolve()),
    ]
    if args:
        command.extend(args)
    return command


def ensure_streamlit_runtime(
    *,
    module_name: str,
    script_path: str,
    args: Sequence[str] | None = None,
) -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return

    has_streamlit_context = get_script_run_ctx() is not None
    if not should_bootstrap_streamlit(
        module_name=module_name,
        has_streamlit_context=has_streamlit_context,
    ):
        return

    command = build_streamlit_command(
        python_executable=sys.executable,
        script_path=script_path,
        args=args,
    )
    env = os.environ.copy()
    env[BOOTSTRAP_ENV_VAR] = "1"
    print(
        "Launching the Streamlit UI with `python -m streamlit run` for proper app mode...",
        flush=True,
    )
    raise SystemExit(subprocess.run(command, env=env, check=False).returncode)
