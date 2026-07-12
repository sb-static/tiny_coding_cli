#!/usr/bin/env python3
"""
Tiny coding CLI with structured file tools.

Runs a coding task against an OpenAI-compatible /v1/chat/completions endpoint
such as your MoA proxy. The model gets terminal + file tools and can work inside
the --workdir workspace.

Example:
  python tiny_coding_cli_file_tools.py \
    --task "Create hello.py and run it" \
    --workdir /tmp/project \
    --base-url http://localhost:8000/v1 \
    --api-key test-key \
    --model moa --tools all --verbose
"""

from __future__ import annotations

import argparse
import html as html_lib
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

COMPLETION_MARKER = "CODING_FINAL_OUTPUT"
DEFAULT_MAX_OUTPUT_CHARS = 24000
PROJECT_INSTRUCTION_FILES = ("AGENTS.md",)  # checked in workspace root, in order

DEFAULT_WEB_TIMEOUT = 15.0
DEFAULT_WEB_MAX_RESULTS = 6
DEFAULT_WEB_MAX_CHARS = 20000
WEB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# Web search uses the `ddgs` package (pip install ddgs), a metasearch library that
# queries DuckDuckGo and other backends with built-in rotation. We keep a light
# throttle + retries on top, and an optional SearXNG instance as a final fallback.
WEB_SEARCH_MIN_INTERVAL = 1.0   # seconds between searches
WEB_SEARCH_MAX_ATTEMPTS = 3     # ddgs attempts before falling back to SearXNG

_WEB_SESSION = requests.Session()  # used by the SearXNG fallback and web_fetch helpers
_LAST_SEARCH_TS = 0.0
_SEARCH_THROTTLE_LOCK = threading.Lock()

try:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
    _DDGS_IMPORT_ERROR = ""
except ImportError as _exc:  # pragma: no cover - depends on environment
    DDGS = None  # type: ignore[assignment]
    DDGSException = RatelimitException = TimeoutException = Exception  # type: ignore[assignment,misc]
    _DDGS_IMPORT_ERROR = str(_exc)

_DDGS_CLIENT = None  # lazily created, reused across searches


def _get_ddgs_client(timeout: float):
    global _DDGS_CLIENT
    if _DDGS_CLIENT is None:
        try:
            _DDGS_CLIENT = DDGS(timeout=timeout)
        except TypeError:  # older/newer ddgs without a timeout kwarg
            _DDGS_CLIENT = DDGS()
    return _DDGS_CLIENT

BASE_SYSTEM_PROMPT = """You are a coding agent. You complete the user's task by working directly in a local \
workspace using the tools provided. You are autonomous: keep working until the task is fully done.

How to work:
1. Explore first. Inspect the workspace and read relevant files before changing anything.
2. Plan briefly. State the concrete steps you will take in one or two sentences.
3. Act in small steps. Make one focused change at a time, then check the result.
4. Verify. After meaningful edits, run a test or syntax check when tools allow it.
5. If something fails, read the actual error, fix the real cause, and retry. Do not guess blindly.

Operating rules:
- The workspace root is given below. Treat all relative paths as relative to it. Stay inside it.
- If the workspace provides project instructions (e.g. an AGENTS.md), they are included with the \
task below. Follow them as project conventions, but the operating, safety, and finishing rules \
in this message still take precedence if they ever conflict.
- Make reasonable assumptions and proceed; do not stop to ask the user questions. Only the final \
message is shown to the user, so do not address them mid-task.
- Prefer one tool call per turn so you can react to each result, unless calls are clearly independent.
- Mind your context limits. Check a file's size with list_files before reading it. For large \
files, read a small part first to learn the structure, then narrow with line ranges or search \
instead of loading the whole thing. If a read or command output comes back truncated, it is \
incomplete: narrow your next read rather than assuming you saw everything.
- Use only non-interactive commands. Never run editors (vim, nano), pagers (less, more), \
long-running servers, watchers, or anything that waits for input.

Finishing:
- You are done when the task is complete and verified. To finish, reply with a normal text \
message and NO tool calls. That message is your final answer: a short summary of what you did \
and how it was verified. Do not call any tool in the same turn you finish.""".strip()


TERMINAL_TOOL_NAMES = {"terminal"}
WEB_TOOL_NAMES = {"web_search", "web_fetch"}
FILE_TOOL_NAMES = {
    "get_workdir",
    "list_files",
    "read_file",
    "search_files",
    "write_file",
    "edit_file",
    "create_directory",
    "delete_path",
}


def tool_name(tool: Dict[str, Any]) -> str:
    return str((tool.get("function") or {}).get("name") or "")


def resolve_enabled_tool_names(spec: str) -> List[str]:
    """Resolve --tools into an ordered list of enabled tool names.

    Supported values:
      terminal  -> only terminal
      files     -> file tools only, no terminal
      web       -> web_search + web_fetch only
      all       -> terminal + file + web tools
      none      -> no tools; model can only answer text
      a,b,c     -> custom comma-separated subset
    """
    spec = (spec or "all").strip().lower()
    all_names = [tool_name(tool) for tool in TOOLS]

    if spec in {"all", "default", "terminal+files", "files+terminal"}:
        return all_names
    if spec in {"terminal", "shell"}:
        return [name for name in all_names if name in TERMINAL_TOOL_NAMES]
    if spec in {"files", "file", "file-tools", "file_tools"}:
        return [name for name in all_names if name in FILE_TOOL_NAMES]
    if spec in {"web", "internet", "search"}:
        return [name for name in all_names if name in WEB_TOOL_NAMES]
    if spec in {"none", "off", "no-tools", "no_tools"}:
        return []

    requested = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [name for name in requested if name not in set(all_names)]
    if unknown:
        raise ValueError(f"Unknown tool name(s) in --tools: {unknown}. Available: {all_names}")
    return [name for name in all_names if name in set(requested)]


def filter_tools(enabled_names: List[str]) -> List[Dict[str, Any]]:
    enabled = set(enabled_names)
    return [tool for tool in TOOLS if tool_name(tool) in enabled]


def build_system_prompt(enabled_names: List[str], workspace_root: str = "") -> str:
    enabled = set(enabled_names)
    has_terminal = bool(enabled & TERMINAL_TOOL_NAMES)
    has_files = bool(enabled & FILE_TOOL_NAMES)

    mode_lines: List[str] = []
    if has_terminal and has_files:
        mode_lines.extend([
            "Tools available: structured file tools + a terminal.",
            "- Use read_file before edit_file. Use write_file to create new files or fully rewrite one.",
            "- To locate code, use search_files (returns clean path:line matches); use terminal grep/find",
            "  only when you need shell features like pipes. Read whole files only when you must.",
            "- Use the terminal for tests, builds, git, package installs, and inspection.",
            "- When data is too big to read directly (large logs, CSVs, datasets), write a small",
            "  script that processes it and prints only the summary or answer you need, then run it.",
            "- If edit_file reports old_text was not found, re-read the file and retry with a smaller,",
            "  exact snippet copied from what you just read (including indentation).",
            "",
            "Example of a correct edit cycle:",
            '  1) read_file {"path": "app.py"}',
            '  2) edit_file {"path": "app.py", "old_text": "    return 1", "new_text": "    return 2"}',
            '  3) terminal {"command": "python -m py_compile app.py"}',
        ])
    elif has_terminal:
        mode_lines.extend([
            "Tools available: terminal only.",
            "- Inspect, create, edit, and test files with non-interactive shell commands.",
            "- For edits prefer reliable approaches: a small python script, cat with a heredoc, or sed.",
            "- Read a file (e.g. `cat -n path`) before editing it. Mind quoting in heredocs.",
            "- When data is too big to read directly (large logs, CSVs, datasets), write a small",
            "  script that processes it and prints only the summary or answer you need, then run it.",
        ])
    elif has_files:
        mode_lines.extend([
            "Tools available: structured file tools only. There is NO terminal in this mode.",
            "- Use list_files / read_file / search_files / write_file / edit_file / create_directory / delete_path.",
            "- To find code, use search_files and then read_file around the matching line numbers,",
            "  instead of reading entire files.",
            "- read_file before edit_file. write_file creates or fully rewrites a file.",
            "- You cannot run tests here. In your final answer, state the exact commands the user",
            "  should run to verify the work.",
            "",
            "Example of a correct edit cycle:",
            '  1) read_file {"path": "app.py"}',
            '  2) edit_file {"path": "app.py", "old_text": "    return 1", "new_text": "    return 2"}',
        ])
    else:
        mode_lines.extend([
            "Tools available: none.",
            "- You cannot inspect or modify the workspace. Reply with a concrete plan and the exact",
            "  file contents or patch text the user should apply manually.",
        ])

    if enabled & WEB_TOOL_NAMES:
        mode_lines.extend([
            "",
            "Web access:",
            "- Use web_search to find current information you are unsure about, then web_fetch a",
            "  result's url to read the page. Do not rely on memory for facts that may be outdated.",
            "- Prefer official docs and primary sources. Cross-check important claims across pages,",
            "  and cite the urls you used in your final answer.",
        ])

    root_line = f"Workspace root: {workspace_root}\n\n" if workspace_root else ""
    return BASE_SYSTEM_PROMPT + "\n\n" + root_line + "\n".join(mode_lines)

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_workdir",
            "description": "Return the absolute workspace directory used by this coding session.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files/directories under a relative workspace path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace.", "default": "."},
                    "recursive": {"type": "boolean", "description": "List recursively.", "default": False},
                    "max_entries": {"type": "integer", "description": "Maximum entries to return.", "default": 200},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file. Returns content with line numbers and total_lines. Reads from the top by default (head); set tail=true to read the last max_lines lines (useful for logs). Read a file before editing it; for large files page through it with start_line/max_lines or jump in with search_files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace."},
                    "start_line": {"type": "integer", "description": "1-based line to start at. Ignored when tail=true.", "default": 1},
                    "max_lines": {"type": "integer", "description": "Max lines to return.", "default": 240},
                    "tail": {"type": "boolean", "description": "Return the last max_lines lines instead of reading from start_line.", "default": False},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search the workspace for a text pattern and return matching lines with their file path and line number. Use this to locate code instead of reading whole files. Pair it with read_file to view the surrounding context of a match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text to find. A plain substring by default; set regex=true to treat it as a regular expression."},
                    "path": {"type": "string", "description": "Relative file or directory to search within.", "default": "."},
                    "regex": {"type": "boolean", "description": "Treat pattern as a regular expression.", "default": False},
                    "case_insensitive": {"type": "boolean", "description": "Case-insensitive match.", "default": False},
                    "max_results": {"type": "integer", "description": "Maximum matching lines to return.", "default": 100},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a UTF-8 text file inside the workspace. Parents are created automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace."},
                    "content": {"type": "string", "description": "Complete file content to write."},
                    "append": {"type": "boolean", "description": "Append instead of overwrite.", "default": False},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact text span in a file. old_text must match the file byte-for-byte (including whitespace and indentation) and is unique unless replace_all=true. Always read_file first and copy old_text from what you read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace."},
                    "old_text": {"type": "string", "description": "Exact text to replace."},
                    "new_text": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences.", "default": False},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a directory inside the workspace, including parents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative directory path."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_path",
            "description": "Delete a file or directory inside the workspace. Directories require recursive=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to delete."},
                    "recursive": {"type": "boolean", "description": "Required for directories.", "default": False},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Execute a bash command with cwd set to the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds.", "default": 60},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web (via DuckDuckGo) and return a ranked list of results, each with title, url, and a short snippet. Use this to find current information, documentation, or pages to read. Follow up with web_fetch on a result's url to read its full text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query. Keep it short and specific."},
                    "max_results": {"type": "integer", "description": "Maximum results to return.", "default": DEFAULT_WEB_MAX_RESULTS},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a web page by URL and return its readable text with HTML stripped. Use it to read a page found via web_search or a URL the user provided. Long pages are truncated; narrow with a more specific source if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL to fetch."},
                    "max_chars": {"type": "integer", "description": "Maximum characters of text to return.", "default": DEFAULT_WEB_MAX_CHARS},
                },
                "required": ["url"],
            },
        },
    },
]


class Workspace:
    def __init__(self, root: str, max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> None:
        self.root = Path(root).expanduser().resolve()
        self.max_output_chars = max_output_chars
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError(f"Workspace is not a directory: {self.root}")

    def resolve(self, path: str) -> Path:
        raw = Path(path or ".")
        candidate = raw.resolve() if raw.is_absolute() else (self.root / raw).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace: {path!r}") from exc
        return candidate

    def rel(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def truncate(self, text: str) -> Tuple[str, bool]:
        if len(text) <= self.max_output_chars:
            return text, False
        omitted = len(text) - self.max_output_chars
        return text[: self.max_output_chars] + f"\n...[truncated {omitted} chars]", True

    def truncate_middle(self, text: str) -> Tuple[str, bool]:
        """Keep the head AND the tail of oversized text (for terminal output, where
        the failure reason — a traceback, a pytest summary — is usually at the end)."""
        if len(text) <= self.max_output_chars:
            return text, False
        head_chars = self.max_output_chars // 2
        tail_chars = self.max_output_chars - head_chars
        omitted = len(text) - head_chars - tail_chars
        return (
            text[:head_chars]
            + f"\n...[truncated {omitted} chars in the middle; head and tail shown]...\n"
            + text[-tail_chars:],
            True,
        )

    def get_workdir(self) -> Dict[str, Any]:
        return {"ok": True, "workdir": str(self.root)}

    def load_project_instructions(self) -> Optional[Dict[str, Any]]:
        """Return {file, content, truncated} for the first project instruction file
        (e.g. AGENTS.md) found in the workspace root, or None if none exists."""
        for name in PROJECT_INSTRUCTION_FILES:
            candidate = self.root / name
            if not candidate.is_file():
                continue
            try:
                text = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            content, truncated = self.truncate(text)
            return {"file": name, "content": content, "truncated": truncated}
        return None

    def list_files(self, path: str = ".", recursive: bool = False, max_entries: int = 200) -> Dict[str, Any]:
        target = self.resolve(path)
        if not target.exists():
            return {"ok": False, "error": f"Path does not exist: {path}", "entries": []}
        max_entries = max(1, int(max_entries or 200))
        if target.is_file():
            return {"ok": True, "entries": [{"path": self.rel(target), "type": "file", "size": target.stat().st_size}]}
        iterator = target.rglob("*") if recursive else target.iterdir()
        entries = []
        truncated = False
        for item in sorted(iterator, key=lambda p: str(p).lower()):
            if len(entries) >= max_entries:
                truncated = True
                break
            try:
                entries.append({
                    "path": self.rel(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": None if item.is_dir() else item.stat().st_size,
                })
            except OSError as exc:
                entries.append({"path": str(item), "type": "error", "error": str(exc)})
        return {"ok": True, "path": self.rel(target), "recursive": recursive, "entries": entries, "truncated": truncated}

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 240, tail: bool = False) -> Dict[str, Any]:
        target = self.resolve(path)
        if not target.exists():
            return {"ok": False, "error": f"File does not exist: {path}"}
        if not target.is_file():
            return {"ok": False, "error": f"Not a file: {path}"}
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"ok": False, "error": f"File is not valid UTF-8 text: {path}"}
        max_lines = max(1, int(max_lines or 240))
        lines = text.splitlines(keepends=True)
        total = len(lines)
        if tail:
            start_idx = max(0, total - max_lines)
        else:
            start_line = max(1, int(start_line or 1))
            start_idx = min(start_line - 1, total)
        end_idx = min(start_idx + max_lines, total)
        content, trunc_chars = self.truncate("".join(lines[start_idx:end_idx]))
        more_after = end_idx < total
        more_before = start_idx > 0
        result = {
            "ok": True,
            "path": self.rel(target),
            "start_line": start_idx + 1,
            "end_line": end_idx,
            "total_lines": total,
            "content": content,
            "truncated": trunc_chars or more_after or more_before,
        }
        notes: List[str] = []
        if more_after:
            notes.append(f"continue downward with start_line={end_idx + 1}")
        if more_before:
            notes.append(f"earlier lines exist above; read with start_line=1 (or a smaller value) to see them")
        if notes:
            shown = f"Showing lines {start_idx + 1}-{end_idx} of {total}"
            if tail:
                shown += " (tail)"
            result["hint"] = (
                f"{shown}. This is partial; " + ", or ".join(notes)
                + ", or use search_files to jump to the relevant part instead of reading it all."
            )
        elif trunc_chars:
            result["hint"] = "Output was truncated by size. Read a narrower line range to see the rest."
        return result

    def search_files(self, pattern: str, path: str = ".", regex: bool = False,
                     case_insensitive: bool = False, max_results: int = 100) -> Dict[str, Any]:
        if not pattern:
            return {"ok": False, "error": "pattern must not be empty"}
        target = self.resolve(path)
        if not target.exists():
            return {"ok": False, "error": f"Path does not exist: {path}", "matches": []}
        max_results = max(1, int(max_results or 100))
        flags = re.IGNORECASE if case_insensitive else 0
        if regex:
            try:
                compiled = re.compile(pattern, flags)
            except re.error as exc:
                return {"ok": False, "error": f"Invalid regex: {exc}"}
            matcher = lambda line: compiled.search(line) is not None
        elif case_insensitive:
            needle = pattern.lower()
            matcher = lambda line: needle in line.lower()
        else:
            matcher = lambda line: pattern in line

        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", ".mypy_cache", ".pytest_cache"}
        max_file_bytes = 5 * 1024 * 1024
        files = [target] if target.is_file() else sorted(
            (p for p in target.rglob("*")
             if p.is_file() and not any(part in skip_dirs for part in p.relative_to(self.root).parts)),
            key=lambda p: str(p).lower(),
        )

        matches: List[Dict[str, Any]] = []
        files_scanned = 0
        truncated = False
        for fpath in files:
            if len(matches) >= max_results:
                truncated = True
                break
            try:
                if fpath.stat().st_size > max_file_bytes:
                    continue
                text = fpath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # skip binary / unreadable files
            files_scanned += 1
            for lineno, line in enumerate(text.splitlines(), start=1):
                if matcher(line):
                    matches.append({
                        "path": self.rel(fpath),
                        "line_number": lineno,
                        "line": line[:400],
                    })
                    if len(matches) >= max_results:
                        truncated = True
                        break
        return {
            "ok": True,
            "pattern": pattern,
            "path": self.rel(target),
            "files_scanned": files_scanned,
            "match_count": len(matches),
            "matches": matches,
            "truncated": truncated,
        }

    def write_file(self, path: str, content: str, append: bool = False) -> Dict[str, Any]:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8", newline="") as f:
            f.write(content)
        return {"ok": True, "path": self.rel(target), "bytes": len(content.encode("utf-8")), "append": append}

    def edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> Dict[str, Any]:
        target = self.resolve(path)
        if not target.exists():
            return {"ok": False, "error": f"File does not exist: {path}"}
        if not target.is_file():
            return {"ok": False, "error": f"Not a file: {path}"}
        text = target.read_text(encoding="utf-8")
        if not old_text:
            return {"ok": False, "error": "old_text must not be empty"}
        count_found = text.count(old_text)
        if count_found == 0:
            return {"ok": False, "error": "old_text not found exactly", "hint": "Re-read the file and use exact text."}
        if count_found > 1 and not replace_all:
            return {
                "ok": False,
                "error": f"old_text matches {count_found} locations; refusing an ambiguous edit",
                "hint": "Extend old_text with surrounding lines so it is unique, or set replace_all=true to replace every occurrence.",
            }
        count = count_found if replace_all else 1
        target.write_text(text.replace(old_text, new_text, count), encoding="utf-8", newline="")
        return {"ok": True, "path": self.rel(target), "occurrences_found": count_found, "occurrences_replaced": count}

    def create_directory(self, path: str) -> Dict[str, Any]:
        target = self.resolve(path)
        target.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "path": self.rel(target)}

    def delete_path(self, path: str, recursive: bool = False) -> Dict[str, Any]:
        target = self.resolve(path)
        if not target.exists():
            return {"ok": False, "error": f"Path does not exist: {path}"}
        if target == self.root:
            return {"ok": False, "error": "Refusing to delete workspace root"}
        if target.is_dir():
            if not recursive:
                return {"ok": False, "error": "Path is a directory; set recursive=true"}
            shutil.rmtree(target)
        else:
            target.unlink()
        return {"ok": True, "path": path, "deleted": True}

    def terminal(self, command: str, timeout: int = 60) -> Dict[str, Any]:
        started = time.time()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout or 60)),
            )
            output = completed.stdout or ""
            if completed.stderr:
                output += "\n--- stderr ---\n" + completed.stderr
            output, truncated = self.truncate_middle(output)
            return {
                "ok": completed.returncode == 0,
                "command": command,
                "exit_code": completed.returncode,
                "output": output,
                "truncated": truncated,
                "elapsed_sec": round(time.time() - started, 3),
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "command": command, "exit_code": -1, "error": f"Command timed out", "output": ""}
        except Exception as exc:
            return {"ok": False, "command": command, "exit_code": -1, "error": str(exc), "output": ""}


def _strip_html_to_text(html_content: str) -> str:
    """Strip HTML tags/scripts/styles and return collapsed readable text."""
    if not html_content:
        return ""
    text = html_content
    for tag in ("script", "style", "nav", "header", "footer", "noscript", "svg"):
        text = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)          # remaining tags
    text = html_lib.unescape(text)                # entities
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def _html_title(html_content: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html_content or "", flags=re.DOTALL | re.IGNORECASE)
    return html_lib.unescape(re.sub(r"\s+", " ", match.group(1)).strip()) if match else ""


def _host_is_private(host: str) -> bool:
    """True if host resolves to a loopback/private/link-local address (SSRF guard)."""
    if not host:
        return True
    host = host.strip("[]")  # strip IPv6 brackets
    if host.lower() == "localhost":
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return False  # cannot resolve; let the request fail normally
    for info in infos:
        addr = info[4][0].split("%")[0]  # drop zone id
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return True
    return False


# SearXNG simple-theme layout: <h3><a href="URL">title</a></h3> + <p class="content">snippet</p>.
_SEARXNG_RESULT_RE = re.compile(
    r'<h3[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    flags=re.DOTALL | re.IGNORECASE,
)
_SEARXNG_SNIPPET_RE = re.compile(
    r'<p[^>]*class="content"[^>]*>(.*?)</p>',
    flags=re.DOTALL | re.IGNORECASE,
)


def _throttle_search() -> None:
    """Block until at least WEB_SEARCH_MIN_INTERVAL has passed since the last search."""
    global _LAST_SEARCH_TS
    with _SEARCH_THROTTLE_LOCK:
        wait = WEB_SEARCH_MIN_INTERVAL - (time.monotonic() - _LAST_SEARCH_TS)
        if wait > 0:
            time.sleep(wait)
        _LAST_SEARCH_TS = time.monotonic()


def _searxng_search(base_url: str, query: str, max_results: int, timeout: float,
                    user_agent: str) -> Dict[str, Any]:
    """Query a SearXNG instance. Tries format=json first; many public instances
    disable it (HTTP 403), in which case we fall back to parsing the HTML page."""
    base = base_url.rstrip("/")
    instance_host = urlparse(base).netloc.lower()
    headers = {"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.9"}

    # 1) Native JSON API (needs 'json' enabled in the instance's settings.yml).
    try:
        resp = _WEB_SESSION.get(
            f"{base}/search",
            params={"q": query, "format": "json"},
            headers={**headers, "Accept": "application/json"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            results = []
            for item in (data.get("results") or [])[: max_results]:
                url = str(item.get("url") or "")
                if not url.startswith("http"):
                    continue
                results.append({
                    "title": str(item.get("title") or ""),
                    "url": url,
                    "snippet": str(item.get("content") or ""),
                })
            return {"ok": True, "results": results}
        json_error = f"HTTP {resp.status_code}"
    except ValueError:
        json_error = "non-JSON response"
    except Exception as exc:
        json_error = str(exc)

    # 2) HTML fallback (works on instances with the JSON format disabled).
    try:
        resp = _WEB_SESSION.get(
            f"{base}/search",
            params={"q": query},
            headers={**headers, "Accept": "text/html,application/xhtml+xml"},
            timeout=timeout,
        )
    except Exception as exc:
        return {"ok": False, "error": f"searxng failed (json: {json_error}; html: {exc})"}
    if resp.status_code != 200:
        return {"ok": False,
                "error": f"searxng failed (json: {json_error}; html: HTTP {resp.status_code})"}

    titles_urls = _SEARXNG_RESULT_RE.findall(resp.text)
    snippets = _SEARXNG_SNIPPET_RE.findall(resp.text)
    results: List[Dict[str, str]] = []
    seen = set()
    for idx, (href, title_html) in enumerate(titles_urls):
        url = html_lib.unescape(href)
        if not url.startswith("http") or urlparse(url).netloc.lower() == instance_host:
            continue
        if url in seen:
            continue
        seen.add(url)
        results.append({
            "title": _strip_html_to_text(title_html),
            "url": url,
            "snippet": _strip_html_to_text(snippets[idx]) if idx < len(snippets) else "",
        })
        if len(results) >= max_results:
            break
    return {"ok": True, "results": results}


def web_search(query: str, max_results: int = DEFAULT_WEB_MAX_RESULTS,
               timeout: float = DEFAULT_WEB_TIMEOUT, user_agent: str = WEB_USER_AGENT,
               searxng_url: str = "") -> Dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "query must not be empty", "results": []}
    max_results = max(1, min(int(max_results or DEFAULT_WEB_MAX_RESULTS), 20))

    last_error = "unknown error"
    if DDGS is None:
        last_error = (f"ddgs package not installed ({_DDGS_IMPORT_ERROR}); "
                      f"run: pip install ddgs")
    else:
        for attempt in range(WEB_SEARCH_MAX_ATTEMPTS):
            _throttle_search()
            try:
                raw = _get_ddgs_client(timeout).text(
                    query,
                    region="us-en",
                    safesearch="moderate",
                    max_results=max_results,
                    backend="auto",  # let ddgs rotate across its search backends
                )
            except RatelimitException as exc:
                last_error = f"rate limited: {exc}"
                time.sleep(1.5 * (2 ** attempt))  # 1.5s, 3s, 6s, ...
                continue
            except (TimeoutException, DDGSException) as exc:
                last_error = f"ddgs search failed: {exc}"
                time.sleep(1.0 * (attempt + 1))
                continue
            except Exception as exc:  # defensive: ddgs internals change between versions
                last_error = f"ddgs search failed: {exc}"
                time.sleep(1.0 * (attempt + 1))
                continue

            results: List[Dict[str, str]] = []
            seen = set()
            for item in raw or []:
                url = str(item.get("href") or item.get("url") or "")
                if not url.startswith("http") or url in seen:
                    continue
                seen.add(url)
                results.append({
                    "title": str(item.get("title") or ""),
                    "url": url,
                    "snippet": str(item.get("body") or ""),
                })
                if len(results) >= max_results:
                    break
            return {"ok": True, "query": query, "source": "ddgs",
                    "result_count": len(results), "results": results}

    # Final tier: SearXNG instance, if configured (--searxng-url / CODING_CLI_SEARXNG_URL).
    if searxng_url:
        _throttle_search()
        sx = _searxng_search(searxng_url, query, max_results, timeout, user_agent)
        if sx.get("ok"):
            results = sx["results"]
            return {"ok": True, "query": query, "source": "searxng",
                    "result_count": len(results), "results": results}
        last_error = f"{last_error}; then {sx.get('error', 'searxng failed')}"

    return {"ok": False, "error": f"search failed after {WEB_SEARCH_MAX_ATTEMPTS} attempts: {last_error}",
            "results": []}


def web_fetch(url: str, max_chars: int = DEFAULT_WEB_MAX_CHARS, timeout: float = DEFAULT_WEB_TIMEOUT,
              user_agent: str = WEB_USER_AGENT, allow_local: bool = False) -> Dict[str, Any]:
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"ok": False, "error": "url must be an absolute http(s) URL"}
    if not allow_local and _host_is_private(parsed.hostname or ""):
        return {"ok": False, "error": f"Refusing to fetch private/loopback host '{parsed.hostname}'. "
                                       f"Use --allow-local-fetch to override."}
    max_chars = max(500, int(max_chars or DEFAULT_WEB_MAX_CHARS))
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml,text/plain"},
            timeout=timeout,
            allow_redirects=True,
        )
    except Exception as exc:
        return {"ok": False, "error": f"fetch failed: {exc}", "url": url}
    if resp.status_code != 200:
        return {"ok": False, "error": f"fetch returned HTTP {resp.status_code}", "url": url,
                "final_url": resp.url}

    content_type = resp.headers.get("Content-Type", "").lower()
    if "html" in content_type:
        title = _html_title(resp.text)
        text = _strip_html_to_text(resp.text)
    elif "text/" in content_type or "json" in content_type or "xml" in content_type:
        title = ""
        text = resp.text
    else:
        return {"ok": False, "error": f"unsupported content type: {content_type or 'unknown'}",
                "url": url, "final_url": resp.url}

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n...[truncated]"
    return {
        "ok": True,
        "url": url,
        "final_url": resp.url,
        "content_type": content_type,
        "title": title,
        "truncated": truncated,
        "text": text,
    }


def tool_result(workspace: Workspace, name: str, args: Dict[str, Any], command_timeout: int,
                web_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    web = web_config or {}
    try:
        if name == "get_workdir":
            return workspace.get_workdir()
        if name == "list_files":
            return workspace.list_files(args.get("path", "."), bool(args.get("recursive", False)), int(args.get("max_entries", 200)))
        if name == "read_file":
            return workspace.read_file(str(args["path"]), int(args.get("start_line", 1)), int(args.get("max_lines", 240)), bool(args.get("tail", False)))
        if name == "search_files":
            return workspace.search_files(
                str(args["pattern"]),
                str(args.get("path", ".")),
                bool(args.get("regex", False)),
                bool(args.get("case_insensitive", False)),
                int(args.get("max_results", 100)),
            )
        if name == "write_file":
            return workspace.write_file(str(args["path"]), str(args.get("content", "")), bool(args.get("append", False)))
        if name == "edit_file":
            return workspace.edit_file(str(args["path"]), str(args.get("old_text", "")), str(args.get("new_text", "")), bool(args.get("replace_all", False)))
        if name == "create_directory":
            return workspace.create_directory(str(args["path"]))
        if name == "delete_path":
            return workspace.delete_path(str(args["path"]), bool(args.get("recursive", False)))
        if name == "terminal":
            return workspace.terminal(str(args.get("command", "")), int(args.get("timeout", command_timeout)))
        if name == "web_search":
            return web_search(
                str(args.get("query", "")),
                int(args.get("max_results", web.get("max_results", DEFAULT_WEB_MAX_RESULTS))),
                float(web.get("timeout", DEFAULT_WEB_TIMEOUT)),
                str(web.get("user_agent", WEB_USER_AGENT)),
                str(web.get("searxng_url", "") or ""),
            )
        if name == "web_fetch":
            return web_fetch(
                str(args.get("url", "")),
                int(args.get("max_chars", web.get("max_chars", DEFAULT_WEB_MAX_CHARS))),
                float(web.get("timeout", DEFAULT_WEB_TIMEOUT)),
                str(web.get("user_agent", WEB_USER_AGENT)),
                bool(web.get("allow_local", False)),
            )
        return {"ok": False, "error": f"Unknown tool: {name}"}
    except KeyError as exc:
        return {"ok": False, "error": f"Missing argument: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def normalize_tool_call(call: Dict[str, Any], idx: int) -> Tuple[str, str, Dict[str, Any]]:
    call_id = call.get("id") or f"call_{idx}_{int(time.time()*1000)}"
    fn = call.get("function") or {}
    name = fn.get("name") or call.get("name") or ""
    raw_args = fn.get("arguments", call.get("arguments", {}))
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args or "{}")
        except Exception:
            args = {"_raw_arguments": raw_args}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {"value": raw_args}
    return call_id, name, args


def parse_text_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    stripped = (text or "").strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[len("```json"):-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data.get("tool_calls"), list):
            return data["tool_calls"]
        if isinstance(data.get("name"), str):
            return [{
                "id": f"call_legacy_{int(time.time()*1000)}",
                "type": "function",
                "function": {"name": data["name"], "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)},
            }]
    return None


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def call_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    temperature: Optional[float],
    timeout: float,
    max_attempts: int = 3,
    backoff_base: float = 2.0,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if temperature is not None:
        payload["temperature"] = temperature

    url = base_url.rstrip("/") + "/chat/completions"
    last_error: Optional[BaseException] = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            if attempt < max_attempts:
                delay = backoff_base ** (attempt - 1)
                print(f"[call_chat] {type(exc).__name__} (attempt {attempt}/{max_attempts}), "
                      f"retrying in {delay:.0f}s...", file=sys.stderr)
                time.sleep(delay)
                continue
            raise RuntimeError(f"chat request failed after {max_attempts} attempts: {exc}") from exc

        if r.status_code in RETRYABLE_STATUS_CODES and attempt < max_attempts:
            delay = backoff_base ** (attempt - 1)
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
            print(f"[call_chat] HTTP {r.status_code} (attempt {attempt}/{max_attempts}), "
                  f"retrying in {delay:.0f}s...", file=sys.stderr)
            time.sleep(delay)
            continue

        try:
            r.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"{exc}\nResponse body:\n{r.text[:4000]}") from exc
        try:
            return r.json()
        except ValueError as exc:
            last_error = exc
            if attempt < max_attempts:
                delay = backoff_base ** (attempt - 1)
                print(f"[call_chat] invalid JSON in response (attempt {attempt}/{max_attempts}), "
                      f"retrying in {delay:.0f}s...", file=sys.stderr)
                time.sleep(delay)
                continue
            raise RuntimeError(f"chat response was not valid JSON after {max_attempts} attempts: {exc}") from exc

    raise RuntimeError(f"chat request failed: {last_error}")


class TurnLogger:
    """Append a JSONL transcript of every turn when a log path is provided.

    Each line is one JSON record with a timestamp and a "type" field, e.g.
    session_start, system_prompt, user, model_response, tool_call, tool_result,
    final, stopped, error. When path is falsy the logger is a no-op.
    """

    def __init__(self, path: Optional[str], session_meta: Optional[Dict[str, Any]] = None) -> None:
        self.path = path or None
        if not self.path:
            return
        try:
            parent = Path(self.path).expanduser().parent
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"[log error] cannot prepare log directory: {exc}", file=sys.stderr)
            self.path = None
            return
        self.log("session_start", **(session_meta or {}))

    def log(self, type_: str, **fields: Any) -> None:
        if not self.path:
            return
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "type": type_, **fields}
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            print(f"[log error] failed to write log: {exc}", file=sys.stderr)
            self.path = None  # stop trying after the first failure


def run_agent(args: argparse.Namespace) -> str:
    workspace = Workspace(args.workdir, max_output_chars=args.max_output_chars)
    enabled_tool_names = resolve_enabled_tool_names(args.tools)
    enabled_tools = filter_tools(enabled_tool_names)
    enabled_tool_set = set(enabled_tool_names)
    if args.verbose:
        print(f"Enabled tools: {', '.join(enabled_tool_names) if enabled_tool_names else '(none)'}", file=sys.stderr)

    logger = TurnLogger(getattr(args, "log_file", None), {
        "model": args.model,
        "workspace": str(workspace.root),
        "enabled_tools": enabled_tool_names,
        "max_iterations": args.max_iterations,
        "temperature": args.temperature,
        "task": args.task,
    })

    project = workspace.load_project_instructions()
    if project:
        task_content = (
            f"Project instructions from {project['file']} (follow these project conventions):\n"
            f"----- BEGIN {project['file']} -----\n"
            f"{project['content']}\n"
            f"----- END {project['file']} -----\n\n"
            f"Task:\n{args.task}"
        )
        if args.verbose:
            note = " (truncated)" if project.get("truncated") else ""
            print(f"Loaded project instructions from {project['file']}{note}.", file=sys.stderr)
    else:
        task_content = f"Task:\n{args.task}"

    web_config = {
        "timeout": getattr(args, "web_timeout", DEFAULT_WEB_TIMEOUT),
        "max_results": getattr(args, "web_max_results", DEFAULT_WEB_MAX_RESULTS),
        "max_chars": getattr(args, "web_max_chars", DEFAULT_WEB_MAX_CHARS),
        "allow_local": getattr(args, "allow_local_fetch", False),
        "user_agent": WEB_USER_AGENT,
        "searxng_url": getattr(args, "searxng_url", "") or "",
    }

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(enabled_tool_names, str(workspace.root))},
        {"role": "user", "content": task_content},
    ]
    logger.log("system_prompt", content=messages[0]["content"])
    if project:
        logger.log("project_instructions", file=project["file"],
                   truncated=project.get("truncated", False), content=project["content"])
    logger.log("user", content=messages[1]["content"])

    last_text = ""
    for iteration in range(1, args.max_iterations + 1):
        if args.verbose:
            print(f"\n--- model call {iteration}/{args.max_iterations} ---", file=sys.stderr)
        try:
            data = call_chat(args.base_url, args.api_key, args.model, messages, enabled_tools, args.temperature, args.api_timeout)
        except Exception as exc:
            logger.log("error", iteration=iteration, where="call_chat", error=str(exc))
            raise
        message = ((data.get("choices") or [{}])[0].get("message") or {})
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or parse_text_tool_calls(content) or []
        logger.log("model_response", iteration=iteration, content=content, tool_calls=tool_calls)
        if content:
            last_text = content
            if args.verbose:
                print(content[:1200], file=sys.stderr)
        if not tool_calls:
            messages.append({"role": "assistant", "content": content})
            logger.log("final", iteration=iteration, content=content)
            return content

        messages.append({"role": "assistant", "content": content if content else None, "tool_calls": tool_calls})
        marker_seen = False
        for idx, call in enumerate(tool_calls):
            call_id, name, tool_args = normalize_tool_call(call, idx)
            if args.verbose:
                print(f"[tool] {name} {json.dumps(tool_args, ensure_ascii=False)[:1000]}", file=sys.stderr)
            logger.log("tool_call", iteration=iteration, call_id=call_id, name=name, args=tool_args)
            if name not in enabled_tool_set:
                result = {
                    "ok": False,
                    "error": f"Tool '{name}' is not enabled in this run",
                    "enabled_tools": enabled_tool_names,
                }
            else:
                result = tool_result(workspace, name, tool_args, args.command_timeout, web_config)
            if args.verbose:
                print(f"[result] {json.dumps(result, ensure_ascii=False)[:2000]}", file=sys.stderr)
            logger.log("tool_result", iteration=iteration, call_id=call_id, name=name, result=result)
            # Only trust the marker when a terminal command actually printed it.
            # Checking the whole serialized result caused false positives whenever
            # read_file/search_files returned a file that merely contained the string.
            if name == "terminal" and COMPLETION_MARKER in str(result.get("output", "")):
                marker_seen = True
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })
        if marker_seen:
            messages.append({"role": "user", "content": "The completion marker was observed. Provide the final concise answer."})

    logger.log("stopped", reason="max_iterations", iterations=args.max_iterations, last_text=last_text)
    return last_text or f"Stopped after max_iterations={args.max_iterations} without final answer."


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny coding CLI with terminal and file tools.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--model", default=os.getenv("CODING_CLI_MODEL", "moa"))
    parser.add_argument(
        "--tools",
        default=os.getenv("CODING_CLI_TOOLS", "all"),
        help=(
            "Tool set to expose: all, terminal, files, none, or comma-separated names. "
            "Examples: --tools terminal ; --tools files ; --tools read_file,write_file,edit_file"
        ),
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "test-key"))
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--command-timeout", type=int, default=60)
    parser.add_argument("--api-timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-chars", type=int, default=DEFAULT_MAX_OUTPUT_CHARS)
    parser.add_argument("--web-timeout", type=float, default=DEFAULT_WEB_TIMEOUT,
                        help="Timeout (seconds) for web_search and web_fetch requests.")
    parser.add_argument("--web-max-results", type=int, default=DEFAULT_WEB_MAX_RESULTS,
                        help="Default number of results returned by web_search.")
    parser.add_argument("--web-max-chars", type=int, default=DEFAULT_WEB_MAX_CHARS,
                        help="Default max characters of page text returned by web_fetch.")
    parser.add_argument("--allow-local-fetch", action="store_true",
                        help="Allow web_fetch to reach private/loopback hosts (off by default; SSRF guard).")
    parser.add_argument("--searxng-url", default=os.getenv("CODING_CLI_SEARXNG_URL", ""),
                        help="Base URL of a SearXNG instance (e.g. http://localhost:8888 or a public "
                             "instance from searx.space). Used as a search fallback when DuckDuckGo "
                             "is rate limited. Tries the JSON API first, then parses HTML.")
    parser.add_argument(
        "--log-file",
        default=os.getenv("CODING_CLI_LOG_FILE"),
        help="If set, append a JSONL transcript of every turn (system prompt, task, model "
             "responses, tool calls, and tool results) to this file. Parent dirs are created.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        print(run_agent(args))
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
