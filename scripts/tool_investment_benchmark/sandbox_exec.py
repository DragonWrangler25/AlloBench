"""Sandboxed script execution used by the R3 write_script/run_script tools and the
R2c code-emission grader.

- has_solve(code)   -> True iff `code` defines a function named `solve`
- _run(code)        -> (stdout, None) on a clean exit, else (None, error)
- _parse_answer(s)  -> the last `ANSWER:` value in stdout, else the last number
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile

_NUM = re.compile(r"-?\$?\s*\d[\d,]*(?:\.\d+)?")
_ANSWER_LINE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)


def has_solve(code: str) -> bool:
    """True iff the code defines a function named `solve`."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False
    return any(isinstance(n, ast.FunctionDef) and n.name == "solve"
               for n in ast.walk(tree))


def _to_float(token: str) -> float | None:
    token = token.replace("$", "").replace(",", "").strip()
    try:
        return float(token)
    except ValueError:
        return None


def _parse_answer(stdout: str) -> float | None:
    """Prefer the last `ANSWER:` line; else the last number anywhere in stdout."""
    ans_lines = _ANSWER_LINE.findall(stdout or "")
    for raw in reversed(ans_lines):
        nums = _NUM.findall(raw)
        if nums:
            v = _to_float(nums[-1])
            if v is not None:
                return v
    nums = _NUM.findall(stdout or "")
    return _to_float(nums[-1]) if nums else None


def _limit_resources():  # pragma: no cover - runs only in the child process
    """Best-effort CPU + address-space caps in the subprocess (preexec_fn)."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))  # 1 GiB
    except Exception:
        pass


def _run(code: str, timeout: float = 5.0) -> tuple[str | None, str | None]:
    """Run `code` in a sandboxed subprocess (isolated, CPU/mem-limited, temp cwd).
    Returns (stdout, None) on a clean exit, or (None, error) on timeout / spawn
    failure / non-zero exit."""
    if not (code or "").strip():
        return None, "empty code"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True, text=True, timeout=timeout,
                cwd=tmp, preexec_fn=_limit_resources,
            )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:  # spawn failure
        return None, f"{type(e).__name__}: {e}"
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        return None, f"exit {proc.returncode}: {err[-1] if err else ''}"
    return proc.stdout, None
