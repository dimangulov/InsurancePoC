"""
Supabase persistence layer for the Insurance PoC.

All writes are best-effort: if SUPABASE_URL / SUPABASE_KEY are not set,
or a write fails, the app logs the error and continues — the LangGraph
PostgresSaver checkpoint is the source of truth for resumability.
"""

import os
import logging
from datetime import date, timedelta
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client — lazy singleton so the app starts even without env vars
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        log.warning("SUPABASE_URL / SUPABASE_KEY not set — Supabase writes disabled.")
        return None

    try:
        from supabase import create_client
        _client = create_client(url, key)
        log.info("Supabase client initialised.")
        return _client
    except ImportError:
        log.warning("supabase package not installed — Supabase writes disabled.")
        return None
    except Exception as exc:
        log.warning("Supabase init failed: %s", exc)
        return None


def _write(table: str, payload: dict) -> bool:
    """Insert one row; returns True on success."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.table(table).insert(payload).execute()
        return True
    except Exception as exc:
        log.error("Supabase write to %s failed: %s", table, exc)
        return False


def _update(table: str, match: dict, payload: dict) -> bool:
    """Update rows matching `match` with `payload`; returns True on success."""
    client = _get_client()
    if client is None:
        return False
    try:
        q = client.table(table).update(payload)
        for col, val in match.items():
            q = q.eq(col, val)
        q.execute()
        return True
    except Exception as exc:
        log.error("Supabase update on %s failed: %s", table, exc)
        return False


# ---------------------------------------------------------------------------
# Public API — one function per table
# ---------------------------------------------------------------------------

def create_session(session_id: str) -> bool:
    return _write("sessions", {"id": session_id, "status": "in_progress"})


def close_session(session_id: str, status: str = "quoted") -> bool:
    return _update("sessions", {"id": session_id}, {"status": status})


def insert_client_profile(session_id: str, profile: dict) -> bool:
    return _write("client_profiles", {"session_id": session_id, **profile})


def insert_classification(session_id: str, product: str, reason: str) -> bool:
    return _write("classifications", {
        "session_id": session_id,
        "product":    product,
        "reason":     reason,
    })


def insert_underwriting_data(session_id: str, product: str, data: dict) -> bool:
    return _write("underwriting_data", {
        "session_id": session_id,
        "product":    product,
        "data":       data,
    })


def insert_quote(session_id: str, quote: dict) -> bool:
    valid_until = (date.today() + timedelta(days=quote.get("valid_days", 30))).isoformat()
    return _write("quotes", {
        "session_id":      session_id,
        "quote_ref":       quote["quote_id"],
        "product":         quote["product"],
        "coverage_limit":  quote.get("coverage_limit"),
        "annual_premium":  quote.get("annual_premium"),
        "monthly_premium": quote.get("monthly_premium"),
        "deductible":      quote.get("deductible"),
        "exclusions":      quote.get("exclusions", []),
        "notes":           quote.get("notes"),
        "valid_until":     valid_until,
    })


def insert_turn(
    session_id:  str,
    node:        str,
    turn_index:  int,
    role:        str,
    content:     str,
    tool_name:   str | None = None,
    tool_call_id: str | None = None,
    metadata:    dict[str, Any] | None = None,
) -> bool:
    return _write("conversation_turns", {
        "session_id":   session_id,
        "node":         node,
        "turn_index":   turn_index,
        "role":         role,
        "content":      content,
        "tool_name":    tool_name,
        "tool_call_id": tool_call_id,
        "metadata":     metadata,
    })
