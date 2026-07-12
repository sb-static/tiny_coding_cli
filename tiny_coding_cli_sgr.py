#!/usr/bin/env python3
"""
Tiny coding CLI — SGR (Schema-Guided Reasoning) edition.

This is a drop-in sibling of tiny_coding_cli_file_tools_optimized.py that replaces
native OpenAI tool calling with Schema-Guided Reasoning: the model is NOT given a
`tools` array. Instead, on every turn it must emit a SINGLE JSON object that first
reasons (reflection -> plan -> reasoning) and then selects EXACTLY ONE action from a
discriminated union of tool schemas. Generation is constrained with
`response_format: {type: "json_schema", ...}` (structured output / guided decoding),
so the output is always schema-valid and the reasoning is produced before the choice.

This is the approach popularized by SGR (https://abdullin.com/schema-guided-reasoning/)
and https://github.com/vamplabAI/sgr-vampi-code. It tends to work better than native
tool calling on small local models, which is what this CLI targets.

Everything else — the Workspace sandbox, file/terminal/web tools, SSRF guard, request
retries, and JSONL logging — is identical to the native-tool-calling version, so the
two can be benchmarked head to head.

Requires: requests, pydantic>=2

Example:
  python tiny_coding_cli_sgr.py \
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
import time
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Type, Union

import requests
from pydantic import BaseModel, ConfigDict, Field, create_model

DEFAULT_MAX_OUTPUT_CHARS = 24000
PROJECT_INSTRUCTION_FILES = ("AGENTS.md",)  # checked in workspace root, in order

DEFAULT_WEB_TIMEOUT = 15.0
DEFAULT_WEB_MAX_RESULTS = 6
DEFAULT_WEB_MAX_CHARS = 20000
WEB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# How many times in a row the model may emit output that does not match the schema
# before we give up (each failure is fed back to the model for self-correction).
MAX_CONSECUTIVE_SCHEMA_FAILURES = 3


# ---------------------------------------------------------------------------
# SGR tool schemas
#
# Each tool is a pydantic model whose `tool` field is a Literal that both names the
# action and acts as the discriminator for the union. Field names match exactly what
# the tool dispatcher (`tool_result`) expects, so execution is a thin adapter:
#   name = model.tool ; args = model.model_dump(exclude={"tool"})
#
# Optional args carry defaults so the model may omit them; required args (path,
# pattern, ...) have no default and become `required` in the JSON schema.
# ---------------------------------------------------------------------------

class _ToolBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetWorkdirTool(_ToolBase):
    """Return the absolute workspace directory for this session."""
    tool: Literal["get_workdir"]


class ListFilesTool(_ToolBase):
    """List files/directories under a relative workspace path."""
    tool: Literal["list_files"]
    path: str = Field(".", description="Relative path inside the workspace.")
    recursive: bool = Field(False, description="List recursively.")
    max_entries: int = Field(200, description="Maximum entries to return.")


class ReadFileTool(_ToolBase):
    """Read a UTF-8 text file. Returns content with line numbers and total_lines.
    Reads from the top by default; set tail=true to read the last max_lines lines.
    Always read a file before editing it; page large files with start_line/max_lines
    or jump in with search_files."""
    tool: Literal["read_file"]
    path: str = Field(..., description="Relative path inside the workspace.")
    start_line: int = Field(1, description="1-based line to start at. Ignored when tail=true.")
    max_lines: int = Field(240, description="Max lines to return.")
    tail: bool = Field(False, description="Return the last max_lines lines instead of reading from start_line.")


class SearchFilesTool(_ToolBase):
    """Search the workspace for a text pattern and return matching lines with their
    file path and line number. Use this to locate code instead of reading whole files,
    then read_file around the matches."""
    tool: Literal["search_files"]
    pattern: str = Field(..., description="Text to find. Plain substring by default; set regex=true for a regular expression.")
    path: str = Field(".", description="Relative file or directory to search within.")
    regex: bool = Field(False, description="Treat pattern as a regular expression.")
    case_insensitive: bool = Field(False, description="Case-insensitive match.")
    max_results: int = Field(100, description="Maximum matching lines to return.")


class WriteFileTool(_ToolBase):
    """Create or overwrite a UTF-8 text file inside the workspace. Parents are created
    automatically."""
    tool: Literal["write_file"]
    path: str = Field(..., description="Relative path inside the workspace.")
    content: str = Field(..., description="Complete file content to write.")
    append: bool = Field(False, description="Append instead of overwrite.")


class EditFileTool(_ToolBase):
    """Replace an exact text span in a file. old_text must match the file byte-for-byte
    (including whitespace/indentation) and be unique unless replace_all=true. Always
    read_file first and copy old_text from what you read."""
    tool: Literal["edit_file"]
    path: str = Field(..., description="Relative path inside the workspace.")
    old_text: str = Field(..., description="Exact text to replace.")
    new_text: str = Field(..., description="Replacement text.")
    replace_all: bool = Field(False, description="Replace all occurrences.")


class CreateDirectoryTool(_ToolBase):
    """Create a directory inside the workspace, including parents."""
    tool: Literal["create_directory"]
    path: str = Field(..., description="Relative directory path.")


class DeletePathTool(_ToolBase):
    """Delete a file or directory inside the workspace. Directories require recursive=true."""
    tool: Literal["delete_path"]
    path: str = Field(..., description="Relative path to delete.")
    recursive: bool = Field(False, description="Required for directories.")


class TerminalTool(_ToolBase):
    """Execute a non-interactive bash command with cwd set to the workspace."""
    tool: Literal["terminal"]
    command: str = Field(..., description="Shell command to execute.")
    timeout: int = Field(60, description="Timeout in seconds.")


class WebSearchTool(_ToolBase):
    """Search the web (via DuckDuckGo) and return ranked results with title, url, and a
    short snippet. Follow up with web_fetch on a result's url to read its full text."""
    tool: Literal["web_search"]
    query: str = Field(..., description="Search query. Keep it short and specific.")
    max_results: int = Field(DEFAULT_WEB_MAX_RESULTS, description="Maximum results to return.")


class WebFetchTool(_ToolBase):
    """Fetch a web page by URL and return its readable text with HTML stripped."""
    tool: Literal["web_fetch"]
    url: str = Field(..., description="Absolute http(s) URL to fetch.")
    max_chars: int = Field(DEFAULT_WEB_MAX_CHARS, description="Maximum characters of text to return.")


class ReportResultTool(_ToolBase):
    """Finish the task. Use this ONLY when the task is complete and (where possible)
    verified. `summary` is the final message shown to the user: a short description of
    what you did and how it was verified. In file-only mode, include the exact commands
    the user should run to verify."""
    tool: Literal["report_result"]
    summary: str = Field(..., description="Final answer shown to the user.")


# Ordered registry of selectable tools (report_result is added separately, always on).
TOOL_MODELS: "Dict[str, Type[_ToolBase]]" = {
    "get_workdir": GetWorkdirTool,
    "list_files": ListFilesTool,
    "read_file": ReadFileTool,
    "search_files": SearchFilesTool,
    "write_file": WriteFileTool,
    "edit_file": EditFileTool,
    "create_directory": CreateDirectoryTool,
    "delete_path": DeletePathTool,
    "terminal": TerminalTool,
    "web_search": WebSearchTool,
    "web_fetch": WebFetchTool,
}

ALL_TOOL_NAMES = list(TOOL_MODELS.keys())
FINISH_TOOL_NAME = "report_result"

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


# ---------------------------------------------------------------------------
# The NextStep schema: reasoning fields FIRST, then the chosen tool.
# Field order matters — the constrained decoder emits them in declaration order, so the
# model "thinks" (reflection -> plan -> reasoning) before it commits to an action.
# ---------------------------------------------------------------------------

class ReasoningBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reflection: str = Field(
        ...,
        description="1-3 sentences: what did the previous tool result tell you, and what is the current state of the task?",
    )
    plan: List[str] = Field(
        ...,
        description="The remaining concrete steps to finish the task, in order. Keep it short; update it every turn.",
    )
    reasoning: str = Field(
        ...,
        description="Why the single tool you are about to choose is the correct next action right now.",
    )


def build_next_step_model(enabled_names: List[str]) -> Type[BaseModel]:
    """Build a NextStep model whose `tool` field is a discriminated union of exactly the
    enabled tools plus the always-available report_result finisher."""
    members: List[Type[_ToolBase]] = [TOOL_MODELS[n] for n in enabled_names if n in TOOL_MODELS]
    members.append(ReportResultTool)

    if len(members) == 1:
        tool_annotation: Any = members[0]
    else:
        tool_annotation = Annotated[Union[tuple(members)], Field(discriminator="tool")]

    next_step = create_model(
        "NextStep",
        __base__=ReasoningBase,
        tool=(tool_annotation, Field(..., description="Exactly one action to take now.")),
    )
    next_step.__doc__ = (
        "A single reasoning-then-action step. Fill the reasoning fields first, then pick "
        "exactly one tool in `tool`."
    )
    return next_step


def strip_titles(schema: Any) -> Any:
    """Remove noisy auto-generated `title` keys from a JSON schema (in place-ish)."""
    if isinstance(schema, dict):
        schema.pop("title", None)
        for value in schema.values():
            strip_titles(value)
    elif isinstance(schema, list):
        for item in schema:
            strip_titles(item)
    return schema


def make_strict(schema: Any) -> Any:
    """Post-process a JSON schema for OpenAI-style strict structured outputs:
    every object gets additionalProperties=false and lists every property as required,
    and default values are dropped (strict mode forbids them). Most local guided-decoding
    backends do NOT need this; it is opt-in via --json-mode schema-strict."""
    if isinstance(schema, dict):
        schema.pop("default", None)
        if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
            schema["additionalProperties"] = False
            schema["required"] = list(schema["properties"].keys())
        for value in schema.values():
            make_strict(value)
    elif isinstance(schema, list):
        for item in schema:
            make_strict(item)
    return schema


def next_step_json_schema(model: Type[BaseModel], strict: bool = False) -> Dict[str, Any]:
    schema = model.model_json_schema()
    strip_titles(schema)
    if strict:
        make_strict(schema)
    return schema


def resolve_enabled_tool_names(spec: str) -> List[str]:
    """Resolve --tools into an ordered list of enabled tool names.

    Supported values: terminal | files | web | all | none | a,b,c
    (report_result is always available and is not listed here.)"""
    spec = (spec or "all").strip().lower()

    if spec in {"all", "default", "terminal+files", "files+terminal"}:
        return list(ALL_TOOL_NAMES)
    if spec in {"terminal", "shell"}:
        return [n for n in ALL_TOOL_NAMES if n in TERMINAL_TOOL_NAMES]
    if spec in {"files", "file", "file-tools", "file_tools"}:
        return [n for n in ALL_TOOL_NAMES if n in FILE_TOOL_NAMES]
    if spec in {"web", "internet", "search"}:
        return [n for n in ALL_TOOL_NAMES if n in WEB_TOOL_NAMES]
    if spec in {"none", "off", "no-tools", "no_tools"}:
        return []

    requested = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [n for n in requested if n not in set(ALL_TOOL_NAMES)]
    if unknown:
        raise ValueError(f"Unknown tool name(s) in --tools: {unknown}. Available: {ALL_TOOL_NAMES}")
    return [n for n in ALL_TOOL_NAMES if n in set(requested)]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """You are a coding agent that works by Schema-Guided Reasoning. You complete the user's \
task by working directly in a local workspace. You are autonomous: keep working until the task is fully done.

Output protocol (STRICT):
- On EVERY turn you reply with ONE JSON object and nothing else. No prose, no markdown, no code fences.
- The JSON must match the provided schema. First fill the reasoning fields (reflection, plan, reasoning), \
then choose EXACTLY ONE action in the `tool` field.
- You take one action per turn. After each action you receive its result as the next user message, then you \
reason again and take the next action.

How to work:
1. Explore first. Inspect the workspace and read relevant files before changing anything.
2. Plan briefly in the `plan` field and update it each turn.
3. Act in small steps. Make one focused change, then check the result on the next turn.
4. Verify. After meaningful edits, run a test or syntax check when a terminal is available.
5. If something fails, read the actual error, fix the real cause, and retry. Do not guess blindly.

Operating rules:
- The workspace root is given below. Treat all relative paths as relative to it. Stay inside it.
- If the workspace provides project instructions (e.g. an AGENTS.md), they are included with the task \
below. Follow them as project conventions, but the operating, safety, and finishing rules here still take \
precedence if they ever conflict.
- Make reasonable assumptions and proceed; do not stop to ask the user questions.
- Mind your context limits. Check a file's size with list_files before reading it. For large files, read a \
small part first, then narrow with line ranges or search_files instead of loading the whole thing. If a read \
or command output comes back truncated, it is incomplete: narrow your next read.
- Use only non-interactive commands. Never run editors (vim, nano), pagers (less, more), long-running \
servers, watchers, or anything that waits for input.

Finishing:
- You are done when the task is complete and verified. To finish, choose the `report_result` tool. Its \
`summary` is the ONLY thing shown to the user: a short summary of what you did and how it was verified. \
Do not choose report_result until the work is actually done.""".strip()


def build_system_prompt(enabled_names: List[str], schema: Dict[str, Any], workspace_root: str = "") -> str:
    enabled = set(enabled_names)
    has_terminal = bool(enabled & TERMINAL_TOOL_NAMES)
    has_files = bool(enabled & FILE_TOOL_NAMES)

    mode_lines: List[str] = []
    if has_terminal and has_files:
        mode_lines.extend([
            "Tools available: structured file tools + a terminal (plus report_result to finish).",
            "- Use read_file before edit_file. Use write_file to create new files or fully rewrite one.",
            "- To locate code, use search_files (returns clean path:line matches); use terminal grep/find",
            "  only when you need shell features like pipes. Read whole files only when you must.",
            "- Use the terminal for tests, builds, git, package installs, and inspection.",
            "- When data is too big to read directly (large logs, CSVs, datasets), write a small script",
            "  that processes it and prints only the summary or answer you need, then run it.",
            "- If edit_file reports old_text was not found, re-read the file and retry with a smaller,",
            "  exact snippet copied from what you just read (including indentation).",
        ])
    elif has_terminal:
        mode_lines.extend([
            "Tools available: terminal only (plus report_result to finish).",
            "- Inspect, create, edit, and test files with non-interactive shell commands.",
            "- For edits prefer reliable approaches: a small python script, cat with a heredoc, or sed.",
            "- Read a file (e.g. `cat -n path`) before editing it. Mind quoting in heredocs.",
            "- When data is too big to read directly, write a small script that prints only the summary.",
        ])
    elif has_files:
        mode_lines.extend([
            "Tools available: structured file tools only (plus report_result to finish). There is NO terminal.",
            "- Use list_files / read_file / search_files / write_file / edit_file / create_directory / delete_path.",
            "- To find code, use search_files and then read_file around the matching line numbers.",
            "- read_file before edit_file. write_file creates or fully rewrites a file.",
            "- You cannot run tests here. In report_result, state the exact commands the user should run to verify.",
        ])
    else:
        mode_lines.extend([
            "Tools available: none except report_result.",
            "- You cannot inspect or modify the workspace. In report_result, give a concrete plan and the exact",
            "  file contents or patch text the user should apply manually.",
        ])

    if enabled & WEB_TOOL_NAMES:
        mode_lines.extend([
            "",
            "Web access:",
            "- Use web_search to find current information you are unsure about, then web_fetch a result's url",
            "  to read the page. Prefer official docs and primary sources; cite urls in your final summary.",
        ])

    root_line = f"Workspace root: {workspace_root}\n\n" if workspace_root else ""
    schema_block = (
        "Respond with JSON conforming to this schema (the `tool` field is a tagged union; the `tool` value "
        "selects which action and which fields are valid):\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    return BASE_SYSTEM_PROMPT + "\n\n" + root_line + "\n".join(mode_lines) + "\n\n" + schema_block


# ---------------------------------------------------------------------------
# Workspace sandbox + tools  (unchanged from the native-tool-calling version)
# ---------------------------------------------------------------------------

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
        """Keep the head AND the tail of oversized text (for terminal output, where the
        failure reason — a traceback, a pytest summary — is usually at the end)."""
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
            notes.append("earlier lines exist above; read with start_line=1 (or a smaller value) to see them")
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
            return {"ok": False, "command": command, "exit_code": -1, "error": "Command timed out", "output": ""}
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


def _decode_ddg_href(href: str) -> str:
    """Resolve a DuckDuckGo HTML result href to the real destination URL."""
    from urllib.parse import unquote, urlparse  # local import keeps top clean
    href = html_lib.unescape(href)
    if "duckduckgo.com/l/" in href:
        m = re.search(r"[?&]uddg=([^&]+)", href)
        if m:
            return unquote(m.group(1))
    if href.startswith("//"):
        return "https:" + href
    return href


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


def web_search(query: str, max_results: int = DEFAULT_WEB_MAX_RESULTS,
               timeout: float = DEFAULT_WEB_TIMEOUT, user_agent: str = WEB_USER_AGENT) -> Dict[str, Any]:
    from urllib.parse import urlparse
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "query must not be empty", "results": []}
    max_results = max(1, min(int(max_results or DEFAULT_WEB_MAX_RESULTS), 20))
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "b": ""},
            headers={"User-Agent": user_agent, "Accept": "text/html"},
            timeout=timeout,
            allow_redirects=True,
        )
    except Exception as exc:
        return {"ok": False, "error": f"search request failed: {exc}", "results": []}
    if resp.status_code != 200:
        return {"ok": False, "error": f"search returned HTTP {resp.status_code}", "results": []}

    page = resp.text
    block_re = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.DOTALL | re.IGNORECASE,
    )
    snippet_re = re.compile(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        flags=re.DOTALL | re.IGNORECASE,
    )
    titles_urls = block_re.findall(page)
    snippets = snippet_re.findall(page)

    results: List[Dict[str, str]] = []
    seen = set()
    for idx, (href, title_html) in enumerate(titles_urls):
        url = _decode_ddg_href(href)
        if not url.startswith("http") or "duckduckgo.com" in urlparse(url).netloc:
            continue
        if url in seen:
            continue
        seen.add(url)
        title = _strip_html_to_text(title_html)
        snippet = _strip_html_to_text(snippets[idx]) if idx < len(snippets) else ""
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break

    return {"ok": True, "query": query, "result_count": len(results), "results": results}


def web_fetch(url: str, max_chars: int = DEFAULT_WEB_MAX_CHARS, timeout: float = DEFAULT_WEB_TIMEOUT,
              user_agent: str = WEB_USER_AGENT, allow_local: bool = False) -> Dict[str, Any]:
    from urllib.parse import urlparse
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


# ---------------------------------------------------------------------------
# JSON extraction — the model should return a bare JSON object, but we tolerate
# code fences and leading/trailing prose by scanning for the first balanced object.
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> Dict[str, Any]:
    """Return the first top-level JSON object found in `text`. Raises ValueError if none
    parses. Handles ```json fences and stray prose around the object."""
    if text is None:
        raise ValueError("empty response")
    stripped = text.strip()
    # Fast path: whole thing is JSON.
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Strip a single fenced block if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict):
                return obj
        except Exception:
            stripped = inner  # fall through to brace scan on the fenced content
    # Brace scan: find the first balanced {...}, ignoring braces inside strings.
    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break  # unbalanced/invalid; try next '{'
        start = stripped.find("{", start + 1)
    raise ValueError("no valid JSON object found in model response")


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def call_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    response_format: Optional[Dict[str, Any]],
    temperature: Optional[float],
    timeout: float,
    max_attempts: int = 3,
    backoff_base: float = 2.0,
) -> Dict[str, Any]:
    """POST to /chat/completions. Unlike the native-tools version this sends no `tools`
    array; instead it passes `response_format` to constrain the output to the SGR schema."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format
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


def build_response_format(json_mode: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map --json-mode to an OpenAI-compatible response_format payload.

    schema        -> json_schema, strict=false (best for vLLM/llama.cpp/LM Studio guided decoding)
    schema-strict -> json_schema, strict=true  (OpenAI structured outputs; schema forced closed)
    object        -> json_object (server guarantees valid JSON but not the schema; prompt carries it)
    none          -> no response_format at all (rely on prompt + tolerant parsing)
    """
    if json_mode == "none":
        return None
    if json_mode == "object":
        return {"type": "json_object"}
    strict = json_mode == "schema-strict"
    return {
        "type": "json_schema",
        "json_schema": {"name": "next_step", "schema": schema, "strict": strict},
    }


class TurnLogger:
    """Append a JSONL transcript of every turn when a log path is provided.

    Each line is one JSON record with a timestamp and a "type" field, e.g.
    session_start, system_prompt, user, model_response, reasoning, tool_call,
    tool_result, schema_error, final, stopped, error. When path is falsy it is a no-op."""

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
    enabled_tool_set = set(enabled_tool_names)

    next_step_model = build_next_step_model(enabled_tool_names)
    schema = next_step_json_schema(next_step_model, strict=(args.json_mode == "schema-strict"))
    response_format = build_response_format(args.json_mode, schema)

    if args.verbose:
        shown = ", ".join(enabled_tool_names) if enabled_tool_names else "(none)"
        print(f"Enabled tools: {shown} (+ {FINISH_TOOL_NAME})", file=sys.stderr)
        print(f"JSON mode: {args.json_mode}", file=sys.stderr)

    logger = TurnLogger(getattr(args, "log_file", None), {
        "mode": "sgr",
        "model": args.model,
        "workspace": str(workspace.root),
        "enabled_tools": enabled_tool_names,
        "json_mode": args.json_mode,
        "max_iterations": args.max_iterations,
        "temperature": args.temperature,
        "task": args.task,
    })
    logger.log("schema", schema=schema)

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
    }

    system_prompt = build_system_prompt(enabled_tool_names, schema, str(workspace.root))
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_content},
    ]
    logger.log("system_prompt", content=system_prompt)
    if project:
        logger.log("project_instructions", file=project["file"],
                   truncated=project.get("truncated", False), content=project["content"])
    logger.log("user", content=task_content)

    last_summary = ""
    consecutive_failures = 0

    for iteration in range(1, args.max_iterations + 1):
        if args.verbose:
            print(f"\n--- model call {iteration}/{args.max_iterations} ---", file=sys.stderr)
        try:
            data = call_chat(args.base_url, args.api_key, args.model, messages,
                             response_format, args.temperature, args.api_timeout)
        except Exception as exc:
            logger.log("error", iteration=iteration, where="call_chat", error=str(exc))
            raise
        message = ((data.get("choices") or [{}])[0].get("message") or {})
        content = message.get("content") or ""
        logger.log("model_response", iteration=iteration, content=content)

        # Parse + validate against the SGR schema.
        try:
            raw_obj = extract_json_object(content)
            step = next_step_model.model_validate(raw_obj)
        except Exception as exc:
            consecutive_failures += 1
            logger.log("schema_error", iteration=iteration, error=str(exc), attempt=consecutive_failures)
            if args.verbose:
                print(f"[schema error] {exc}", file=sys.stderr)
            if consecutive_failures >= MAX_CONSECUTIVE_SCHEMA_FAILURES:
                logger.log("stopped", reason="schema_failures", iterations=iteration)
                return (last_summary or
                        f"Stopped: model produced invalid JSON {consecutive_failures} times in a row. Last error: {exc}")
            # Feed the error back so the model can self-correct.
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    "Your previous message did not parse as a single JSON object matching the schema. "
                    f"Error: {exc}. Reply with ONLY the JSON object — no prose, no code fences."
                ),
            })
            continue

        consecutive_failures = 0
        tool_model = step.tool
        name = tool_model.tool
        tool_args = tool_model.model_dump(exclude={"tool"})

        # Persist the model's full reasoning+action turn in the transcript.
        reasoning_dump = step.model_dump(exclude={"tool"})
        logger.log("reasoning", iteration=iteration, **reasoning_dump)
        logger.log("tool_call", iteration=iteration, name=name, args=tool_args)
        if args.verbose:
            print(f"[reasoning] {reasoning_dump.get('reasoning', '')[:600]}", file=sys.stderr)
            print(f"[tool] {name} {json.dumps(tool_args, ensure_ascii=False)[:1000]}", file=sys.stderr)

        # Record the assistant's JSON turn verbatim (preserves reasoning in context).
        messages.append({"role": "assistant", "content": content})

        # Finishing is an explicit tool, not the absence of a tool call.
        if name == FINISH_TOOL_NAME:
            summary = tool_args.get("summary", "") or ""
            last_summary = summary
            logger.log("final", iteration=iteration, content=summary)
            return summary

        if name not in enabled_tool_set:
            result: Dict[str, Any] = {
                "ok": False,
                "error": f"Tool '{name}' is not enabled in this run",
                "enabled_tools": enabled_tool_names,
            }
        else:
            result = tool_result(workspace, name, tool_args, args.command_timeout, web_config)

        if args.verbose:
            print(f"[result] {json.dumps(result, ensure_ascii=False)[:2000]}", file=sys.stderr)
        logger.log("tool_result", iteration=iteration, name=name, result=result)

        # There is no native tool role/id in SGR, so the result is delivered as the next
        # user turn, wrapped so the model can tell it apart from the original task.
        messages.append({
            "role": "user",
            "content": (
                f"Result of {name}:\n"
                f"{json.dumps(result, ensure_ascii=False)}\n\n"
                "Continue: reason about this result, then choose the next tool "
                "(or report_result if the task is complete)."
            ),
        })

    logger.log("stopped", reason="max_iterations", iterations=args.max_iterations, last_summary=last_summary)
    return last_summary or f"Stopped after max_iterations={args.max_iterations} without calling report_result."


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny coding CLI (SGR / schema-guided reasoning; no native tool calls).")
    parser.add_argument("--task", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--model", default=os.getenv("CODING_CLI_MODEL", "moa"))
    parser.add_argument(
        "--tools",
        default=os.getenv("CODING_CLI_TOOLS", "all"),
        help=(
            "Tool set to expose: all, terminal, files, web, none, or comma-separated names. "
            "report_result is always available. "
            "Examples: --tools terminal ; --tools files ; --tools read_file,write_file,edit_file"
        ),
    )
    parser.add_argument(
        "--json-mode",
        default=os.getenv("CODING_CLI_JSON_MODE", "schema"),
        choices=["schema", "schema-strict", "object", "none"],
        help=(
            "How to constrain output. 'schema' (default): response_format json_schema, non-strict "
            "(best for local guided decoding). 'schema-strict': OpenAI strict structured outputs. "
            "'object': json_object only (schema carried in the prompt). 'none': no response_format."
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
    parser.add_argument(
        "--log-file",
        default=os.getenv("CODING_CLI_LOG_FILE"),
        help="If set, append a JSONL transcript of every turn (system prompt, task, model "
             "responses, reasoning, tool calls, and tool results) to this file. Parent dirs are created.",
    )
    parser.add_argument("--print-schema", action="store_true",
                        help="Print the generated NextStep JSON schema for the selected --tools and exit.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.print_schema:
        enabled = resolve_enabled_tool_names(args.tools)
        model = build_next_step_model(enabled)
        schema = next_step_json_schema(model, strict=(args.json_mode == "schema-strict"))
        print(json.dumps(schema, indent=2, ensure_ascii=False))
        return 0

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
