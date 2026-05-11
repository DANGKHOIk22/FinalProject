"""
Loads tools_config.yaml once at import time.

Usage in each tool:
    from app.soccer_agent.toolbox._config_loader import tool_description, tool_input_description

    class MyTool(BaseTool):
        def __init__(self):
            super().__init__(description=tool_description("my_tool"))
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "tools_config.yaml"

# Loaded once at import time — never reloaded at runtime.
_config: dict[str, Any] = {}

try:
    with _CONFIG_PATH.open(encoding="utf-8") as _f:
        _raw = yaml.safe_load(_f)
        _config = (_raw or {}).get("tools", {})
    logger.info(f"[tools_config] Loaded {len(_config)} tool configs from {_CONFIG_PATH.name}")
except Exception as _exc:
    logger.error(f"[tools_config] Failed to load {_CONFIG_PATH}: {_exc}")


def tool_description(name: str) -> str:
    """Return the description string for *name* from tools_config.yaml."""
    cfg = _config.get(name, {})
    desc = cfg.get("description", "")
    if not desc:
        logger.warning(f"[tools_config] No description found for tool '{name}'")
    return desc.strip()


def tool_input_description(tool_name: str, input_name: str) -> str:
    """Return the description for a specific input field of a tool."""
    cfg = _config.get(tool_name, {})
    return cfg.get("inputs", {}).get(input_name, "").strip()
