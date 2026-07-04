"""Rewriter engine for the talkie-rewriter plugin.

Calls the Talkie finetune model (via direct OpenAI-compatible client) to
rewrite LLM responses with the finetune's style and voice.

Uses a separate API key + base_url from the host's model — never touches
ctx.llm. Fail-open: returns the original text if the rewrite fails.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import TalkieRewriterConfig, get_config

logger = logging.getLogger("hermes.plugins.talkie-rewriter")


def _build_messages(
    original_output: str,
    system_prompt: str,
    context_messages: List[Dict[str, str]],
    retry_hint: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build chat messages for the rewriter LLM call.

    The Talkie finetune model does not support the 'system' role, so the
    system prompt is folded into the first user message.

    Structure:
      [last N user+assistant messages for context]
      [user: system_prompt + rewrite instruction + original output]
      [if retry: additional user message with stronger direction]
    """
    # Prepend the system prompt to the rewrite instruction
    rewrite_instruction = (
        f"{system_prompt}\n\n"
        "Rewrite the following response in your own voice and style. "
        "Preserve all factual content, technical accuracy, and formatting. "
        "Do not add information that wasn't in the original. "
        "Output ONLY the rewritten text with no preamble or explanation.\n\n"
        f"{original_output}"
    )

    messages: List[Dict[str, str]] = []

    # Add context messages (already truncated to N by caller)
    for msg in context_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # The rewrite instruction (with system prompt folded in)
    messages.append({"role": "user", "content": rewrite_instruction})

    # On retry, add a stronger direction message
    if retry_hint:
        messages.append({"role": "user", "content": retry_hint})

    return messages


def rewrite_response(
    original_output: str,
    config: Optional[TalkieRewriterConfig] = None,
    system_prompt_override: Optional[str] = None,
    context_messages: Optional[List[Dict[str, str]]] = None,
    retry_hint: Optional[str] = None,
) -> Optional[str]:
    """Call the Talkie model to rewrite a response.

    Returns the rewritten text on success, or None on failure (caller should
    fail-open and pass through the original).

    Args:
        retry_hint: If provided (on retry attempts), appended to the rewrite
            instruction to steer the model away from the failure mode.
    """
    config = config or get_config()
    system_prompt = system_prompt_override or config.system_prompt
    context = context_messages or []

    if not config.api_key:
        logger.warning("talkie-rewriter: TALKIE_API_KEY not configured — skipping rewrite")
        return None

    messages = _build_messages(original_output, system_prompt, context, retry_hint)

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("talkie-rewriter: openai package not installed — cannot rewrite")
        return None

    try:
        client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        )

        # Build kwargs dynamically — some backends reject unknown params
        # (e.g. Talkie finetune endpoint returns 422 for frequency_penalty
        # and presence_penalty). Only include params we know are set.
        create_kwargs: dict[str, Any] = dict(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        # Only pass penalty params if they're non-zero — many finetune
        # endpoints reject them as extra_forbidden.
        if config.frequency_penalty:
            create_kwargs["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty:
            create_kwargs["presence_penalty"] = config.presence_penalty

        response = client.chat.completions.create(**create_kwargs)

        content = response.choices[0].message.content
        if content and content.strip():
            return content.strip()

        logger.warning("talkie-rewriter: model returned empty response")
        return None

    except Exception as exc:
        logger.warning("talkie-rewriter: rewrite failed: %s", exc)
        return None
