"""Verification engine for the talkie-rewriter plugin.

After the Talkie model rewrites a response, this module calls a second LLM
to check whether the rewrite actually addresses the user's message or has
mistakenly started replying to / commenting on the original LLM output.

Uses a separate (faster, cheaper) model via Plexus — not the Talkie finetune,
which is a prose model unsuited for structured verification.

Fail-open: if the verification call fails, the rewrite is accepted as-is.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import TalkieRewriterConfig, get_config

logger = logging.getLogger("hermes.plugins.talkie-rewriter")

# ── Result parsing ──────────────────────────────────────────────────────────

# Matches "PASS - reason" or "FAIL - reason" (case-insensitive)
_VERDICT_PATTERN = re.compile(r"^\s*(PASS|FAIL)\s*[-—:]\s*(.+)", re.IGNORECASE | re.DOTALL)


def _format_context(messages: List[Dict[str, str]], max_chars: int = 2000) -> str:
    """Format context messages into a compact string for the verifier prompt."""
    lines = []
    total = 0
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        line = f"{label}: {content}"
        if total + len(line) > max_chars:
            remaining = max_chars - total
            if remaining > 50:
                line = line[:remaining] + "..."
            else:
                break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) if lines else "(no prior context)"


def _extract_last_user_message(messages: List[Dict[str, str]]) -> str:
    """Pull the most recent user message from context."""
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            return msg["content"]
    return ""


def _parse_verdict(content: str) -> Tuple[bool, str]:
    """Parse the verifier's response into (passed, reason).

    Expected format: 'PASS - reason' or 'FAIL - reason'
    Falls back to (True, 'unparseable') on unexpected output.
    """
    match = _VERDICT_PATTERN.search(content.strip())
    if match:
        verdict = match.group(1).upper()
        reason = match.group(2).strip()
        return (verdict == "PASS", reason)

    # Fallback: look for standalone PASS/FAIL
    upper = content.strip().upper()
    if upper.startswith("PASS"):
        return (True, content.strip())
    if upper.startswith("FAIL"):
        return (False, content.strip())

    logger.debug("talkie-rewriter: unparseable verifier output: %s", content[:200])
    return (True, "unparseable verifier output — accepting")


def verify_rewrite(
    original_output: str,
    rewritten_output: str,
    context_messages: List[Dict[str, str]],
    config: Optional[TalkieRewriterConfig] = None,
) -> Tuple[bool, str]:
    """Check if the rewritten output addresses the user, not the original LLM output.

    Returns ``(passed, reason)`` where:
      - passed=True: the rewrite is acceptable (addresses the user)
      - passed=False: the rewrite appears to reply to the original LLM output
      - reason: one-sentence explanation from the verifier
    """
    config = config or get_config()

    # Can't verify without the user's message
    last_user_msg = _extract_last_user_message(context_messages)
    if not last_user_msg:
        logger.debug("talkie-rewriter: no user message in context — skipping verification")
        return (True, "no user message to verify against")

    if not config.verification_api_key:
        logger.warning("talkie-rewriter: verification enabled but no API key — skipping")
        return (True, "no verification API key configured")

    # Truncate original output for token budget
    original_snippet = original_output[:800]
    if len(original_output) > 800:
        original_snippet += "..."

    # Build the verification prompt
    context_str = _format_context(context_messages, max_chars=1500)

    verify_prompt = (
        "You are a quality checker. Examine whether the rewritten text responds to the "
        "user's message, or if it has mistakenly started replying to or commenting on "
        "the original AI output instead.\n\n"
        f"Recent conversation:\n{context_str}\n\n"
        f"User's last message:\n{last_user_msg}\n\n"
        f"Original AI output (source material that was supposed to be rewritten):\n{original_snippet}\n\n"
        f"Rewritten output:\n{rewritten_output}\n\n"
        "Does the rewritten output directly answer or address what the user asked? "
        "Or does it instead read like a reply to, commentary on, or reaction to the "
        "original AI output?\n\n"
        "Respond with EXACTLY one line in this format:\n"
        "PASS - <one sentence reason>\n"
        "or\n"
        "FAIL - <one sentence reason>"
    )

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("talkie-rewriter: openai package not installed — cannot verify")
        return (True, "openai package not installed")

    try:
        client = OpenAI(
            base_url=config.verification_base_url,
            api_key=config.verification_api_key,
            timeout=config.verification_timeout,
        )

        response = client.chat.completions.create(
            model=config.verification_model,
            messages=[{"role": "user", "content": verify_prompt}],
            temperature=0.0,
            max_tokens=128,
        )

        content = response.choices[0].message.content or ""

    except Exception as exc:
        logger.warning("talkie-rewriter: verification call failed (fail-open): %s", exc)
        return (True, f"verification call failed: {exc}")

    passed, reason = _parse_verdict(content)
    logger.info(
        "talkie-rewriter: verification %s — %s",
        "PASS" if passed else "FAIL",
        reason,
    )
    return (passed, reason)
