# Talkie Rewriter

Hermes Agent plugin that rewrites LLM responses through a style-flavored finetune model. Designed for weakly pretrained LLMs finetuned on novel datasets to flavor the style of the text.

**Author:** GwynTel
**License:** MIT

## What It Does

Every LLM response passes through a rewriting model (e.g. Talkie-1930-13B) that rewrites the text in its own trained style while preserving factual content. Optional Qwen3Guard-Gen-4B moderation runs first to check for unsafe content.

```
user message → pre_llm_call (stash history) → main LLM → transform_llm_output → user
                                                         ↓
                                               ┌────────────────────────┐
                                               │ 1. [optional] Guard    │
                                               │    Qwen3Guard-Gen-4B  │
                                               │ 2. Rewriter LLM       │
                                               │    talkie-1930-13b    │
                                               │    system prompt +    │
                                               │    last N messages +   │
                                               │    original output     │
                                               │ 3. Return rewritten    │
                                               └────────────────────────┘
```

## Installation

```bash
hermes plugins install gwyntel/talkie-rewriter
```

Or manual install: clone into `~/.hermes/plugins/talkie-rewriter/`.

## Configuration

Add to `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - talkie-rewriter
  entries:
    talkie-rewriter:
      # Rewriter LLM
      api_key: ${TALKIE_API_KEY}            # separate key, NOT the host key
      base_url: "https://plexus.nebulosa-bass.ts.net/v1"
      model: "talkie-lm/talkie-1930-13b-it"
      system_prompt: ""                     # settable system prompt for the rewriting LLM
      temperature: 0.7
      max_tokens: 4096
      top_p: 1.0
      frequency_penalty: 0.0
      presence_penalty: 0.0
      timeout: 60
      
      # Context passing
      context_messages: 4                   # last N user+assistant messages to pass
      
      # Moderation (optional)
      moderation:
        enabled: false                      # toggle via tool too
        model: "Qwen/Qwen3Guard-Gen-4B"     # or your gateway alias
        block_threshold: "high"             # "medium" or "high"
        action: "block"                     # "block" or "flag"
      
      # Output
      flag_template: "(talkie reflects the culture and values of the texts it was trained on, not the views of its authors. It can produce outputs that are inaccurate or offensive.)"
      fail_open: true                       # pass original through if rewriter fails
```

Add to `~/.hermes/.env`:

```
TALKIE_API_KEY=sk-your-talkie-api-key
```

## Registered Tools

The plugin registers 5 tools in the `talkie-rewriter` toolset:

- **`talkie_set_system_prompt`** — Set/update the rewriter's system prompt at runtime
- **`talkie_set_param`** — Set generation parameters (temperature, max_tokens, top_p, etc.)
- **`talkie_toggle_moderation`** — Enable/disable the Qwen3Guard moderation guard
- **`talkie_set_context_depth`** — Change how many recent messages to pass as context
- **`talkie_get_config`** — Read current runtime config

These let the agent running in the harness control its own rewriter.

## Architecture

### Hooks

- **`pre_llm_call`** — Stashes the last N user+assistant messages from `conversation_history` into session-scoped state
- **`transform_llm_output`** — Runs guard check (if enabled), then calls the Talkie model to rewrite. Returns rewritten text with flag prefix

### Why Direct HTTP, Not `ctx.llm`

The Talkie model uses its own API key on a separate provider endpoint. `ctx.llm.complete()` uses the host's active model+auth and trust gates block `provider=`/`model=` overrides by default. Direct `openai` client calls with explicit `base_url` + `api_key` bypass this cleanly.

### Fail-Open

If the Talkie model is unreachable, times out, or errors, the original response passes through unchanged. Never blocks output.

## Qwen3Guard-Gen-4B Moderation

The guard model takes user+assistant message pairs and returns:

```
Safety: Safe|Unsafe|Controversial
Categories: Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None
Refusal: Yes|No
```

- **`block` action** — unsafe responses are replaced with a moderation message
- **`flag` action** — unsafe responses are still rewritten but prefixed with a warning
- **`block_threshold`** — `"medium"` blocks controversial+unsafe, `"high"` blocks only unsafe

## File Structure

```
talkie-rewriter/
├── plugin.yaml          # manifest
├── __init__.py          # register(ctx) — hooks + tools + session state
├── config.py            # config reader
├── rewriter.py          # direct OpenAI client call to Talkie model
├── moderator.py         # Qwen3Guard moderation + response parsing
└── README.md
```
