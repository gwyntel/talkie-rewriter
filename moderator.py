"""Moderation engine for the talkie-rewriter plugin.

Uses Qwen3Guard-Gen-4B (via OpenAI-compatible API) to check if LLM output
is safe before rewriting. The guard model takes user+assistant messages
and returns a structured safety assessment.

Output format from Qwen3Guard-Gen-4B:
    Safety: Safe|Unsafe|Controversial
    Categories: Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|
                PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive
                Topics|Copyright Violation|Jailbreak|None
    Refusal: Yes|No  (only for response moderation)

Fail-open: if the guard model is unreachable, moderation passes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import TalkieRewriterConfig, get_config

logger = logging.getLogger("hermes.plugins.talkie-rewriter")

# ── Parsing patterns ────────────────────────────────────────────────────────

_SAFETY_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
_CATEGORY_PATTERN = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|"
    r"PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)"
)
_REFUSAL_PATTERN = re.compile(r"Refusal:\s*(Yes|No)", re.IGNORECASE)

# Risk level ordering for threshold comparison
_RISK_ORDER = {"safe": 0, "controversial": 1, "unsafe": 2}
_THRESHOLD_MAP = {"low": 1, "medium": 1, "high": 2}


def _parse_guard_response(content: str) -> Tuple[str, List[str], Optional[str]]:
    """Parse the guard model's text response.

    Returns (safety_label, categories, refusal) where:
      - safety_label: "Safe", "Unsafe", or "Controversial"
      - categories: list of matched category strings
      - refusal: "Yes", "No", or None (not always present)
    """
    safety_match = _SAFETY_PATTERN.search(content)
    safety_label = safety_match.group(1) if safety_match else "Safe"

    categories = _CATEGORY_PATTERN.findall(content)

    refusal_match = _REFUSAL_PATTERN.search(content)
    refusal = refusal_match.group(1) if refusal_match else None

    return safety_label, categories, refusal


def _exceeds_threshold(safety_label: str, threshold: str) -> bool:
    """Check if the safety label exceeds the configured block threshold."""
    label_rank = _RISK_ORDER.get(safety_label.lower(), 0)
    threshold_rank = _THRESHOLD_MAP.get(threshold.lower(), 2)
    return label_rank >= threshold_rank


def moderate_response(
    user_prompt: str,
    assistant_response: str,
    config: Optional[TalkieRewriterConfig] = None,
) -> Dict[str, Any]:
    """Run Qwen3Guard moderation on a user prompt + assistant response pair.

    Returns a dict with:
      - safe (bool): whether the response passed moderation
      - blocked (bool): whether the response should be blocked
      - safety_label (str): "Safe", "Unsafe", or "Controversial"
      - categories (list[str]): matched risk categories
      - refusal (str|None): whether the model refused
      - error (str|None): error message if moderation failed
    """
    config = config or get_config()

    result = {
        "safe": True,
        "blocked": False,
        "safety_label": "Safe",
        "categories": [],
        "refusal": None,
        "error": None,
    }

    if not config.mod_enabled:
        return result

    if not config.mod_api_key:
        logger.warning("talkie-rewriter: moderation enabled but no API key — skipping")
        result["error"] = "No moderation API key configured"
        return result

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("talkie-rewriter: openai package not installed — cannot moderate")
        result["error"] = "openai package not installed"
        return result

    # Qwen3Guard uses its own chat template — just pass user + assistant messages
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]

    try:
        client = OpenAI(
            base_url=config.mod_base_url,
            api_key=config.mod_api_key,
            timeout=config.timeout,
        )

        response = client.chat.completions.create(
            model=config.mod_model,
            messages=messages,
            max_tokens=128,
            temperature=0.0,
        )

        content = response.choices[0].message.content or ""

    except Exception as exc:
        logger.warning("talkie-rewriter: moderation call failed: %s", exc)
        # Fail-open: can't moderate, allow response
        result["error"] = str(exc)
        return result

    # Parse the guard response
    safety_label, categories, refusal = _parse_guard_response(content)

    result["safety_label"] = safety_label
    result["categories"] = categories
    result["refusal"] = refusal

    # Check against threshold
    if _exceeds_threshold(safety_label, config.mod_block_threshold):
        result["safe"] = False
        if config.mod_action == "block":
            result["blocked"] = True

    logger.info(
        "talkie-rewriter: moderation result: safety=%s categories=%s blocked=%s",
        safety_label, categories, result["blocked"],
    )

    return result
