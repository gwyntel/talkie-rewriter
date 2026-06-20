"""Configuration reader for the talkie-rewriter plugin.

Reads from ``plugins.entries.talkie-rewriter`` in ``~/.hermes/config.yaml``.
All settings are optional — the plugin uses sensible defaults.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.plugins.talkie-rewriter")

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = os.getenv("PLEXUS_BASE_URL", "https://plexus.nebulosa-bass.ts.net/v1")
DEFAULT_MODEL = "talkie-lm/talkie-1930-13b-it"
DEFAULT_SYSTEM_PROMPT = (
    "You are a response rewriter. Rewrite the following text in your own voice "
    "and style. Preserve all factual content, technical accuracy, and formatting. "
    "Do not add information that wasn't in the original. Output ONLY the rewritten text."
)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_P = 1.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_TIMEOUT = 60
DEFAULT_CONTEXT_MESSAGES = 4
DEFAULT_FLAG_TEMPLATE = (
    "(talkie reflects the culture and values of the texts it was trained on, "
    "not the views of its authors. It can produce outputs that are inaccurate or offensive.)"
)
DEFAULT_FAIL_OPEN = True

# Moderation defaults
DEFAULT_MOD_ENABLED = False
DEFAULT_MOD_MODEL = "Qwen/Qwen3Guard-Gen-4B"
DEFAULT_MOD_BLOCK_THRESHOLD = "high"  # "medium" or "high"
DEFAULT_MOD_ACTION = "block"  # "block" or "flag"


class TalkieRewriterConfig:
    """Parsed configuration for the talkie-rewriter plugin."""

    def __init__(self, raw: Optional[Dict[str, Any]] = None):
        raw = raw or {}

        # ── Rewriter LLM ──
        env_key = os.getenv("TALKIE_API_KEY", "")
        self.api_key: str = raw.get("api_key", env_key)
        # Resolve ${VAR} style refs
        if self.api_key and self.api_key.startswith("${") and self.api_key.endswith("}"):
            var_name = self.api_key[2:-1]
            self.api_key = os.getenv(var_name, "")

        self.base_url: str = raw.get("base_url", DEFAULT_BASE_URL)
        self.model: str = raw.get("model", DEFAULT_MODEL)
        self.system_prompt: str = raw.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.temperature: float = raw.get("temperature", DEFAULT_TEMPERATURE)
        self.max_tokens: int = raw.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.top_p: float = raw.get("top_p", DEFAULT_TOP_P)
        self.frequency_penalty: float = raw.get("frequency_penalty", DEFAULT_FREQUENCY_PENALTY)
        self.presence_penalty: float = raw.get("presence_penalty", DEFAULT_PRESENCE_PENALTY)
        self.timeout: int = raw.get("timeout", DEFAULT_TIMEOUT)

        # ── Context ──
        self.context_messages: int = raw.get("context_messages", DEFAULT_CONTEXT_MESSAGES)

        # ── Flag ──
        self.flag_template: str = raw.get("flag_template", DEFAULT_FLAG_TEMPLATE)
        self.fail_open: bool = raw.get("fail_open", DEFAULT_FAIL_OPEN)

        # ── Moderation ──
        mod_raw = raw.get("moderation", {}) or {}
        self.mod_enabled: bool = mod_raw.get("enabled", DEFAULT_MOD_ENABLED)
        self.mod_model: str = mod_raw.get("model", DEFAULT_MOD_MODEL)
        self.mod_api_key: str = mod_raw.get("api_key", self.api_key) or self.api_key
        if self.mod_api_key and self.mod_api_key.startswith("${") and self.mod_api_key.endswith("}"):
            var_name = self.mod_api_key[2:-1]
            self.mod_api_key = os.getenv(var_name, "")
        self.mod_base_url: str = mod_raw.get("base_url", self.base_url)
        self.mod_block_threshold: str = mod_raw.get("block_threshold", DEFAULT_MOD_BLOCK_THRESHOLD)
        self.mod_action: str = mod_raw.get("action", DEFAULT_MOD_ACTION)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a dict (for talkie_get_config tool and logging)."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "timeout": self.timeout,
            "context_messages": self.context_messages,
            "flag_template": self.flag_template,
            "fail_open": self.fail_open,
            "moderation": {
                "enabled": self.mod_enabled,
                "model": self.mod_model,
                "base_url": self.mod_base_url,
                "block_threshold": self.mod_block_threshold,
                "action": self.mod_action,
            },
        }

    @classmethod
    def from_hermes_config(cls) -> "TalkieRewriterConfig":
        """Load config from ``~/.hermes/config.yaml`` → plugins.entries.talkie-rewriter."""
        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            plugins_cfg = cfg.get("plugins", {})
            entries = plugins_cfg.get("entries", {}) if isinstance(plugins_cfg, dict) else {}
            talkie_cfg = entries.get("talkie-rewriter", {}) if isinstance(entries, dict) else {}
            return cls(talkie_cfg)
        except Exception as exc:
            logger.debug("talkie-rewriter config load fallback to defaults: %s", exc)
            return cls()


# ── Module-level singleton ──────────────────────────────────────────────────
_cached_config: Optional[TalkieRewriterConfig] = None


def get_config() -> TalkieRewriterConfig:
    """Return cached config, loading on first call."""
    global _cached_config
    if _cached_config is None:
        _cached_config = TalkieRewriterConfig.from_hermes_config()
    return _cached_config


def reload_config() -> TalkieRewriterConfig:
    """Force a config reload (useful after config changes)."""
    global _cached_config
    _cached_config = TalkieRewriterConfig.from_hermes_config()
    return _cached_config
