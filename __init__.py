"""GwynTel Talkie Rewriter — Hermes Agent plugin for LLM response style rewriting.

Rewrites every LLM response through a style-flavored finetune model
(Talkie-1930-13B) to flavor the text's style and voice. Optional Qwen3Guard
moderation layer. Settable system prompt, generation params, and context
depth — all configurable at runtime via registered tools.

Hooks:
  - pre_llm_call — stashes conversation history for the rewriter
  - transform_llm_output — runs guard (optional) + rewrite, returns new text

Tools:
  - talkie_set_system_prompt
  - talkie_set_param
  - talkie_toggle_moderation
  - talkie_set_context_depth
  - talkie_get_config
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

from .config import TalkieRewriterConfig, get_config, reload_config
from .rewriter import rewrite_response
from .moderator import moderate_response

logger = logging.getLogger("hermes.plugins.talkie-rewriter")

# ────────────────────────────────────────────────────────────────────────────
# Session-scoped runtime state (overlays on top of config defaults)
# ────────────────────────────────────────────────────────────────────────────

# {session_id: {"last_messages": [...], "mod_enabled": bool}}
_session_state: Dict[str, Dict[str, Any]] = {}
_state_lock = threading.Lock()


def _get_state(session_id: str) -> Dict[str, Any]:
    """Get or create session state."""
    with _state_lock:
        if session_id not in _session_state:
            _session_state[session_id] = {
                "last_messages": [],
                "mod_enabled": None,  # None = use config default
                "system_prompt_override": None,  # None = use config default
                "param_overrides": {},  # {param_name: value}
                "context_depth_override": None,  # None = use config default
            }
        return _session_state[session_id]


def _stash_messages(session_id: str, messages: List[Dict[str, str]]) -> None:
    """Stash the last N messages from conversation history."""
    with _state_lock:
        state = _get_state(session_id)
        state["last_messages"] = messages


def _get_stashed_messages(session_id: str) -> List[Dict[str, str]]:
    """Retrieve stashed messages for a session."""
    with _state_lock:
        state = _get_state(session_id)
        return state.get("last_messages", [])


# ────────────────────────────────────────────────────────────────────────────
# Effective config resolver (merges config + runtime overrides)
# ────────────────────────────────────────────────────────────────────────────


_base_config_cache = None

def _get_effective_config(session_id: str) -> TalkieRewriterConfig:
    """Return config with runtime overrides applied for this session."""
    global _base_config_cache
    if _base_config_cache is None:
        _base_config_cache = get_config()
    config = _base_config_cache
    state = _get_state(session_id)

    # Create a shallow copy to overlay overrides
    effective = TalkieRewriterConfig()
    effective.__dict__.update(config.__dict__)

    # Apply runtime overrides
    if state.get("system_prompt_override"):
        effective.system_prompt = state["system_prompt_override"]

    if state.get("mod_enabled") is not None:
        effective.mod_enabled = state["mod_enabled"]

    if state.get("context_depth_override") is not None:
        effective.context_messages = state["context_depth_override"]

    for param, value in state.get("param_overrides", {}).items():
        if hasattr(effective, param):
            setattr(effective, param, value)

    return effective


# ────────────────────────────────────────────────────────────────────────────
# Plugin entry point
# ────────────────────────────────────────────────────────────────────────────


def register(ctx) -> None:
    """Called by the Hermes plugin loader. Registers hooks + tools."""

    # Config is loaded LAZILY on first hook invocation, NOT at register()
    # time. Calling get_config() / load_config() here interferes with the
    # agent's auth initialization and causes the model API call to hang.

    # ════════════════════════════════════════════════════════════════════════
    # HOOK: pre_llm_call — stash conversation history
    # ════════════════════════════════════════════════════════════════════════

    def on_pre_llm_call(**kwargs):
        """Stash the last N user+assistant messages for the rewriter to use."""
        session_id = kwargs.get("session_id", "")
        conversation_history = kwargs.get("conversation_history", [])
        user_message = kwargs.get("user_message", "")

        if not session_id:
            return None

        # Use hardcoded default — config loading deferred to transform hook
        n = 4

        # Extract recent user+assistant messages from conversation history
        recent = []
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                recent.append({"role": role, "content": content})

        # Also add the current user message
        if user_message:
            recent.append({"role": "user", "content": user_message})

        # Keep only the last N
        if len(recent) > n:
            recent = recent[-n:]

        _stash_messages(session_id, recent)

        return None  # pre_llm_call can only inject context, not block

    # ════════════════════════════════════════════════════════════════════════
    # HOOK: transform_llm_output — guard + rewrite
    # ════════════════════════════════════════════════════════════════════════

    def on_transform_llm_output(**kwargs) -> Optional[str]:
        """Optionally moderate, then rewrite the LLM output through the Talkie model."""
        response_text = kwargs.get("response_text", "")
        session_id = kwargs.get("session_id", "")

        if not response_text or not response_text.strip():
            return None

        # Get effective config (with runtime overrides)
        config = _get_effective_config(session_id)

        # Get stashed conversation messages for context
        context_messages = _get_stashed_messages(session_id)

        # Get the last user message (for moderation input)
        last_user_msg = ""
        for msg in reversed(context_messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        # ── Step 1: Moderation (if enabled) ──
        if config.mod_enabled:
            mod_result = moderate_response(
                user_prompt=last_user_msg,
                assistant_response=response_text,
                config=config,
            )

            if mod_result.get("blocked"):
                logger.info(
                    "talkie-rewriter: response blocked by moderation "
                    "(safety=%s, categories=%s)",
                    mod_result.get("safety_label"),
                    mod_result.get("categories"),
                )
                return (
                    "⚠️ This response was blocked by the content moderation guard. "
                    "Please try rephrasing your request."
                )

            if mod_result.get("error") and not mod_result.get("safe"):
                # Moderation had an error but we still check — fail-open
                logger.warning(
                    "talkie-rewriter: moderation error (fail-open): %s",
                    mod_result.get("error"),
                )

        # ── Step 2: Rewrite through Talkie model ──
        # system_prompt_override is only passed if the runtime override changed it
        # from the base config. Since config already has overrides applied, we
        # compare against a fresh base load.
        rewritten = rewrite_response(
            original_output=response_text,
            config=config,
            system_prompt_override=config.system_prompt,
            context_messages=context_messages,
        )

        if rewritten is None:
            # Rewrite failed — fail-open or not
            if config.fail_open:
                logger.info("talkie-rewriter: rewrite failed, passing through original")
                return None  # pass through original
            else:
                return response_text  # still return original but no None passthrough

        if rewritten.strip() == response_text.strip():
            # Model returned identical text — no flag needed
            return None

        # ── Step 3: Prepend flag ──
        flag = config.flag_template
        return f"{flag}\n\n{rewritten}"

    # ════════════════════════════════════════════════════════════════════════
    # TOOL: talkie_set_system_prompt
    # ════════════════════════════════════════════════════════════════════════

    def _talkie_set_system_prompt_handler(args, **kwargs) -> str:
        session_id = kwargs.get("session_id", "")
        text = args.get("text", "").strip()
        if not text:
            return json.dumps({"success": False, "error": "No text provided"})
        state = _get_state(session_id)
        state["system_prompt_override"] = text
        logger.info("talkie-rewriter: system prompt updated (session %s)", session_id)
        return json.dumps({"success": True, "message": "System prompt updated", "length": len(text)})

    ctx.register_tool(
        name="talkie_set_system_prompt",
        toolset="talkie-rewriter",
        schema={
            "name": "talkie_set_system_prompt",
            "description": (
                "Set or update the system prompt for the Talkie rewriter LLM. "
                "This controls the style and behavior instructions given to the "
                "rewriting model. Changes apply to the current session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The new system prompt text.",
                    },
                },
                "required": ["text"],
            },
        },
        handler=_talkie_set_system_prompt_handler,
        check_fn=lambda: True,
        requires_env=[],
        is_async=False,
        description="Set Talkie rewriter system prompt",
        emoji="✏️",
    )

    # ════════════════════════════════════════════════════════════════════════
    # TOOL: talkie_set_param
    # ════════════════════════════════════════════════════════════════════════

    _VALID_PARAMS = {
        "temperature": float,
        "max_tokens": int,
        "top_p": float,
        "frequency_penalty": float,
        "presence_penalty": float,
        "timeout": int,
    }

    def _talkie_set_param_handler(args, **kwargs) -> str:
        session_id = kwargs.get("session_id", "")
        param = args.get("param", "").strip()
        value = args.get("value")

        if not param:
            return json.dumps({"success": False, "error": "No param name provided"})

        if param not in _VALID_PARAMS:
            return json.dumps({
                "success": False,
                "error": f"Unknown param '{param}'. Valid: {list(_VALID_PARAMS.keys())}",
            })

        try:
            expected_type = _VALID_PARAMS[param]
            coerced = expected_type(float(value)) if expected_type == int else expected_type(value)
        except (TypeError, ValueError):
            return json.dumps({
                "success": False,
                "error": f"Invalid value for '{param}': expected {expected_type.__name__}",
            })

        state = _get_state(session_id)
        state["param_overrides"][param] = coerced

        logger.info("talkie-rewriter: param '%s' set to %s (session %s)", param, coerced, session_id)
        return json.dumps({"success": True, "param": param, "value": coerced})

    ctx.register_tool(
        name="talkie_set_param",
        toolset="talkie-rewriter",
        schema={
            "name": "talkie_set_param",
            "description": (
                "Set a generation parameter for the Talkie rewriter LLM. "
                "Valid params: temperature, max_tokens, top_p, "
                "frequency_penalty, presence_penalty, timeout. "
                "Changes apply to the current session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "param": {
                        "type": "string",
                        "description": "Parameter name (temperature, max_tokens, top_p, etc.)",
                        "enum": list(_VALID_PARAMS.keys()),
                    },
                    "value": {
                        "type": "number",
                        "description": "The new value for the parameter.",
                    },
                },
                "required": ["param", "value"],
            },
        },
        handler=_talkie_set_param_handler,
        check_fn=lambda: True,
        requires_env=[],
        is_async=False,
        description="Set Talkie rewriter generation parameter",
        emoji="🎛️",
    )

    # ════════════════════════════════════════════════════════════════════════
    # TOOL: talkie_toggle_moderation
    # ════════════════════════════════════════════════════════════════════════

    def _talkie_toggle_moderation_handler(args, **kwargs) -> str:
        session_id = kwargs.get("session_id", "")
        enabled = args.get("enabled", True)

        state = _get_state(session_id)
        state["mod_enabled"] = bool(enabled)

        logger.info(
            "talkie-rewriter: moderation %s (session %s)",
            "enabled" if enabled else "disabled",
            session_id,
        )
        return json.dumps({
            "success": True,
            "moderation_enabled": bool(enabled),
        })

    ctx.register_tool(
        name="talkie_toggle_moderation",
        toolset="talkie-rewriter",
        schema={
            "name": "talkie_toggle_moderation",
            "description": (
                "Enable or disable the Qwen3Guard content moderation layer. "
                "When enabled, LLM responses are checked for safety before rewriting. "
                "Changes apply to the current session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "True to enable moderation, False to disable.",
                    },
                },
                "required": ["enabled"],
            },
        },
        handler=_talkie_toggle_moderation_handler,
        check_fn=lambda: True,
        requires_env=[],
        is_async=False,
        description="Toggle Qwen3Guard moderation on/off",
        emoji="🛡️",
    )

    # ════════════════════════════════════════════════════════════════════════
    # TOOL: talkie_set_context_depth
    # ════════════════════════════════════════════════════════════════════════

    def _talkie_set_context_depth_handler(args, **kwargs) -> str:
        session_id = kwargs.get("session_id", "")
        messages = args.get("messages", 4)

        try:
            n = int(messages)
            if n < 0:
                raise ValueError("Must be non-negative")
        except (TypeError, ValueError):
            return json.dumps({"success": False, "error": "messages must be a non-negative integer"})

        state = _get_state(session_id)
        state["context_depth_override"] = n

        logger.info("talkie-rewriter: context depth set to %d (session %s)", n, session_id)
        return json.dumps({"success": True, "context_messages": n})

    ctx.register_tool(
        name="talkie_set_context_depth",
        toolset="talkie-rewriter",
        schema={
            "name": "talkie_set_context_depth",
            "description": (
                "Set how many recent user+assistant messages to pass as context "
                "to the Talkie rewriter LLM. Higher values give more conversation "
                "context but use more tokens. Set to 0 to disable context passing. "
                "Changes apply to the current session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "integer",
                        "description": "Number of recent messages to pass (0-20 recommended).",
                    },
                },
                "required": ["messages"],
            },
        },
        handler=_talkie_set_context_depth_handler,
        check_fn=lambda: True,
        requires_env=[],
        is_async=False,
        description="Set context message depth for rewriter",
        emoji="📏",
    )

    # ════════════════════════════════════════════════════════════════════════
    # TOOL: talkie_get_config
    # ════════════════════════════════════════════════════════════════════════

    def _talkie_get_config_handler(args, **kwargs) -> str:
        session_id = kwargs.get("session_id", "")
        config = _get_effective_config(session_id)
        cfg_dict = config.to_dict()
        # Don't expose the API key
        cfg_dict.pop("api_key", None)
        mod = cfg_dict.get("moderation", {})
        mod.pop("api_key", None) if isinstance(mod, dict) else None
        return json.dumps({"success": True, "config": cfg_dict}, indent=2, ensure_ascii=False)

    ctx.register_tool(
        name="talkie_get_config",
        toolset="talkie-rewriter",
        schema={
            "name": "talkie_get_config",
            "description": (
                "Read the current Talkie rewriter configuration, including "
                "system prompt, generation params, context depth, and moderation "
                "status. Shows runtime overrides applied for this session."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        handler=_talkie_get_config_handler,
        check_fn=lambda: True,
        requires_env=[],
        is_async=False,
        description="Get current Talkie rewriter config",
        emoji="📋",
    )

    # ════════════════════════════════════════════════════════════════════════
    # Register hooks
    # ════════════════════════════════════════════════════════════════════════

    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("transform_llm_output", on_transform_llm_output)

    logger.info("talkie-rewriter plugin registered (hooks: pre_llm_call, transform_llm_output)")
    logger.info("talkie-rewriter tools registered: talkie_set_system_prompt, "
                "talkie_set_param, talkie_toggle_moderation, "
                "talkie_set_context_depth, talkie_get_config")
