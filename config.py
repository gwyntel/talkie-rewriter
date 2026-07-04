"""Configuration reader for the talkie-rewriter plugin.

Reads from ``plugins.entries.talkie-rewriter`` in ``~/.hermes/config.yaml``.
All settings are optional — the plugin uses sensible defaults.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.plugins.talkie-rewriter")

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "https://plexus.nebulosa-bass.ts.net/v1"
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

_ENV_REF_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_dotenv_loaded = False
_dotenv_lock = threading.Lock()


def _load_dotenv_once() -> None:
    """Load Hermes' .env lazily before resolving plugin env references.

    Hermes startup normally loads ~/.hermes/.env, but plugin code can be
    imported/registered before that happens in some paths.  This function is
    called only from lazy config construction, never from register().
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return

    with _dotenv_lock:
        if _dotenv_loaded:
            return

        try:
            from hermes_cli.env_loader import load_hermes_dotenv

            load_hermes_dotenv()
        except Exception:
            try:
                from dotenv import load_dotenv

                env_path = os.path.join(os.path.expanduser("~/.hermes"), ".env")
                if os.path.exists(env_path):
                    load_dotenv(env_path, override=False)
            except Exception as exc:
                logger.debug("talkie-rewriter: .env lazy-load skipped: %s", exc)

        _dotenv_loaded = True


def _resolve_env_ref(value: Any, default: Any = None) -> Any:
    """Resolve exact ``${VAR}`` config values against ``os.environ``."""
    if value is None:
        return default
    if isinstance(value, str):
        match = _ENV_REF_PATTERN.match(value.strip())
        if match:
            return os.getenv(match.group(1), "")
    return value


def _get_raw(raw: Dict[str, Any], key: str, default: Any) -> Any:
    """Get a raw config value and resolve exact env-var references."""
    return _resolve_env_ref(raw.get(key, default), default)


class TalkieRewriterConfig:
    """Parsed configuration for the talkie-rewriter plugin."""

    def __init__(self, raw: Optional[Dict[str, Any]] = None):
        _load_dotenv_once()
        raw = raw or {}

        # ── Rewriter LLM ──
        env_key = os.getenv("TALKIE_API_KEY", "")
        self.api_key: str = _get_raw(raw, "api_key", env_key)
        self.base_url: str = _get_raw(
            raw,
            "base_url",
            os.getenv("PLEXUS_BASE_URL", DEFAULT_BASE_URL),
        )
        self.model: str = _get_raw(raw, "model", DEFAULT_MODEL)
        self.system_prompt: str = _get_raw(raw, "system_prompt", DEFAULT_SYSTEM_PROMPT)
        self.temperature: float = raw.get("temperature", DEFAULT_TEMPERATURE)
        self.max_tokens: int = raw.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.top_p: float = raw.get("top_p", DEFAULT_TOP_P)
        self.frequency_penalty: float = raw.get("frequency_penalty", DEFAULT_FREQUENCY_PENALTY)
        self.presence_penalty: float = raw.get("presence_penalty", DEFAULT_PRESENCE_PENALTY)
        self.timeout: int = raw.get("timeout", DEFAULT_TIMEOUT)

        # ── Context ──
        self.context_messages: int = raw.get("context_messages", DEFAULT_CONTEXT_MESSAGES)

        # ── Flag ──
        self.flag_template: str = _get_raw(raw, "flag_template", DEFAULT_FLAG_TEMPLATE)
        self.fail_open: bool = raw.get("fail_open", DEFAULT_FAIL_OPEN)

        # ── Moderation ──
        mod_raw = raw.get("moderation", {}) or {}
        self.mod_enabled: bool = mod_raw.get("enabled", DEFAULT_MOD_ENABLED)
        self.mod_model: str = _get_raw(mod_raw, "model", DEFAULT_MOD_MODEL)
        self.mod_api_key: str = _get_raw(mod_raw, "api_key", self.api_key) or self.api_key
        self.mod_base_url: str = _get_raw(mod_raw, "base_url", self.base_url)
        self.mod_block_threshold: str = _get_raw(mod_raw, "block_threshold", DEFAULT_MOD_BLOCK_THRESHOLD)
        self.mod_action: str = _get_raw(mod_raw, "action", DEFAULT_MOD_ACTION)

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
