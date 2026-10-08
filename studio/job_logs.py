"""Ordered, sanitized IPC logs and cursor reads, independent of lifecycle records."""

from __future__ import annotations

import re

from core.security import REDACTED, sanitize_for_persistence
from core.security import sanitize_log_text as sanitize_core_text

_URL_USERINFO = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^\s/]*@")


def sanitize_log_text(text):
    """Extend Core sanitization to URL userinfo at runtime output boundaries."""
    return _URL_USERINFO.sub(lambda match: match.group(1) + REDACTED + "@", sanitize_core_text(text))


def sanitize_snapshot_value(value):
    """Sanitize record values without interpreting structural job-ID map keys."""
    sanitized = sanitize_for_persistence(value)

    def redact_urls(item):
        if isinstance(item, str):
            return sanitize_log_text(item)
        if isinstance(item, dict):
            return {key: redact_urls(child) for key, child in item.items()}
        if isinstance(item, list):
            return [redact_urls(child) for child in item]
        return item

    return redact_urls(sanitized)


class JobLogs:
    """Buffer child sequences and terminal tails; callers provide synchronization."""

    def __init__(self):
        self.entries = {}
        self._streams = {}

    def append(self, job_id, text):
        """Append a server log without touching the job record."""
        self.entries.setdefault(job_id, []).append(sanitize_log_text(text))

    def receive(self, job_id, payload):
        """Deduplicate and publish contiguous non-terminal messages while running."""
        if not isinstance(payload, dict):
            return
        sequence, text = payload.get("seq"), payload.get("text")
        if type(sequence) is not int or sequence < 0 or not isinstance(text, str):
            return
        stream = self._streams.setdefault(job_id, {"events": {}, "published": set(), "next": 0, "terminal": False})
        if sequence in stream["events"]:
            return
        stream["events"][sequence] = (sanitize_log_text(text), bool(payload.get("terminal")))
        while stream["next"] in stream["events"]:
            current = stream["next"]
            line, terminal = stream["events"][current]
            stream["terminal"] |= terminal
            if not stream["terminal"]:
                self.append(job_id, line)
                stream["published"].add(current)
            stream["next"] += 1

    def finish(self, job_id, result_logs=()):
        """Drain remaining sequence entries once, including terminal and stopped tails."""
        stream = self._streams.pop(job_id, {"events": {}, "published": set()})
        for sequence, text in enumerate(result_logs):
            stream["events"].setdefault(sequence, (sanitize_log_text(text), True))
        for sequence, (text, _) in sorted(stream["events"].items()):
            if sequence not in stream["published"]:
                self.append(job_id, text)

    def page(self, job_id, offset=0, limit=500):
        """Null next_offset means current tail; future logs may extend it."""
        if type(offset) is not int or type(limit) is not int or offset < 0 or limit < 1:
            raise ValueError("offset must be nonnegative and limit positive")
        lines = self.entries.get(job_id, [])
        start = min(offset, len(lines))
        end = min(start + limit, len(lines))
        return {
            "logs": list(lines[start:end]),
            "offset": start,
            "next_offset": end if end < len(lines) else None,
            "total": len(lines),
        }
