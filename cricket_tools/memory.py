# cricket_tools/memory.py — Step 7.3 final verified
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any

_CACHE_DIR = Path(".cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_FILE = _CACHE_DIR / "session_memory.json"

# TTL (2 hours)
_TTL_SECONDS = 2 * 60 * 60

_CONTEXT_KEYS = [
    "player", "playerA", "playerB",
    "team", "venue", "city",
    "season", "start", "end",
    "metric", "n",
]

def _now() -> float:
    return time.time()

def _load_raw() -> Dict[str, Any]:
    if not _CACHE_FILE.exists():
        return {}
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_raw(data: Dict[str, Any]) -> None:
    try:
        _CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # never crash on write

def _expired(ts: float | None) -> bool:
    return (not ts) or (_now() - ts > _TTL_SECONDS)

def _empty_state() -> Dict[str, Any]:
    return {"ts": _now(), "context": {}}

def get_memory() -> Dict[str, Any]:
    raw = _load_raw()
    ts = raw.get("ts")
    if _expired(ts):
        return _empty_state()
    ctx = {k: v for k, v in (raw.get("context") or {}).items()
           if k in _CONTEXT_KEYS and v not in (None, "", [])}
    return {"ts": ts, "context": ctx}

def save_memory(context: Dict[str, Any]) -> None:
    pruned = {k: v for k, v in context.items()
              if k in _CONTEXT_KEYS and v not in (None, "", [])}
    _save_raw({"ts": _now(), "context": pruned})

def merge_with_memory(current: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine new query args with previous memory.
    Memory fills missing keys; user args override.
    """
    mem_state = get_memory()
    mem = mem_state.get("context", {}) if not _expired(mem_state.get("ts")) else {}
    merged = {**mem, **{k: v for k, v in current.items() if v not in (None, "", [])}}
    return {k: merged[k] for k in _CONTEXT_KEYS if merged.get(k) not in (None, "", [])}

def update_memory(resolved: Dict[str, Any]) -> None:
    """
    Update memory with any newly resolved entities.
    Keeps old ones unless explicitly overridden.
    """
    mem = get_memory().get("context", {})
    for k in _CONTEXT_KEYS:
        val = resolved.get(k)
        if val not in (None, "", []):
            mem[k] = val
    save_memory(mem)

def clear_memory() -> None:
    """Erase all memory (used for --clear-memory or tests)."""
    _save_raw({"ts": 0, "context": {}})

# --------------------------------------------------------
# Pending Resolution (Option-3)
# --------------------------------------------------------
def get_pending() -> Dict[str, Any] | None:
    raw = _load_raw()
    pend = raw.get("pending")
    return pend

def save_pending(pending: Dict[str, Any] | None):
    raw = _load_raw()
    raw["pending"] = pending
    _save_raw(raw)