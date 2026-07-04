# Task: Investigate and Fix the talkie-rewriter Hermes Plugin

## Overview

You are working on the `talkie-rewriter` plugin for Hermes Agent (by NousResearch). This plugin intercepts every LLM response from the host agent and rewrites it through a style-flavored finetune model (Talkie-1930-13B) to give responses a pre-1931 English prose style. There is also an optional Qwen3Guard moderation layer.

The plugin has been reverted to commit `eece63a` (the last known-good state: "fix: fold system prompt into user message — Talkie finetune doesn't support system role"). Your job is to:

1. **Understand** the Hermes Agent hook system by examining the Hermes source code
2. **Review** the talkie-rewriter plugin code for correctness, edge cases, and integration issues
3. **Fix** any bugs or issues you find
4. **Deploy** the plugin to the talkie-agent machine (SSH alias: `talkie-agent`, user: `hermes`)
5. **Test end-to-end** by running `hermes -z '<test prompt>'` on talkie-agent

## Codebase Locations

### Plugin Code (this repo)
- **Local repo:** `~/.hermes/plugins/talkie-rewriter/`
- **GitHub:** `https://github.com/gwyntel/talkie-rewriter.git`
- **Files:**
  - `__init__.py` — Main plugin file: hooks, tools, session state, registration
  - `config.py` — Config reader (reads from `plugins.entries.talkie-rewriter` in `~/.hermes/config.yaml`)
  - `rewriter.py` — Rewrite engine: calls Talkie model via OpenAI-compatible client
  - `moderator.py` — Optional Qwen3Guard-Gen-4B moderation
  - `plugin.yaml` — Plugin manifest

### Hermes Agent Source (read-only reference)
- **Location:** `/home/hermes/.hermes/hermes-agent/hermes_cli/`
- **Key files to examine:**
  - `plugins.py` — Contains `VALID_HOOKS` set, `PluginContext` class (where `register_hook()` and `register_tool()` are defined), `PluginManager`, hook emit logic
  - `hooks.py` — Shell hooks system, hook test harness, kwarg shape reference
  - `config.py` — Config loading, how `plugins.entries` is structured

### Talkie-Agent Machine (deployment target + test environment)
- **SSH:** `ssh talkie-agent` (user: `hermes`)
- **Hermes version:** v0.17.0 (2026.6.19)
- **Hermes binary:** `/usr/local/bin/hermes`
- **Hermes source:** NOT at the same path as local — check `python3 -c "import hermes_cli; print(hermes_cli.__file__)"` on talkie-agent
- **Config:** `~/.hermes/config.yaml`
  - Model: `glm.short.fast` via Plexus Gateway (`https://plexus.nebulosa-bass.ts.net/v1`)
  - Plugin config under `plugins.entries.talkie-rewriter`:
    - `api_key: ${TALKIE_API_KEY}` (separate restricted Plexus key)
    - `base_url: https://plexus.nebulosa-bass.ts.net/v1`
    - `model: talkie-lm/talkie-1930-13b-it`
    - `temperature: 0.7`, `max_tokens: 4096`, `context_messages: 4`
    - `moderation.enabled: false`
  - `hooks: {}` (no shell hooks configured)
  - `plugins.enabled: [talkie-rewriter]`

## Hermes Hook System — Deep Reference

### VALID_HOOKS (from `hermes_cli/plugins.py` line 128)

```python
VALID_HOOKS: Set[str] = {
    "pre_tool_call",
    "post_tool_call",
    "transform_terminal_output",
    "transform_tool_result",
    "transform_llm_output",       # ← talkie-rewriter uses this
    "pre_llm_call",               # ← talkie-rewriter uses this
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "subagent_start",
    "subagent_stop",
    "pre_gateway_dispatch",
    "pre_approval_request",
    "post_approval_response",
    "kanban_task_claimed",
    "kanban_task_completed",
    "kanban_task_failed",
}
```

### Hook Types

1. **Plugin Hooks** — Registered via `ctx.register_hook(event_name, callback)` in a plugin's `register(ctx)` function. Run in BOTH CLI and Gateway. Callbacks receive `**kwargs`.

2. **Shell Hooks** — Declared in `hooks:` block of `~/.hermes/config.yaml`. Spawn subprocesses. Run in CLI + Gateway. Use PLUGIN hook event names (from VALID_HOOKS).

3. **Gateway Hooks** — Declared via `HOOK.yaml` + `handler.py` in `~/.hermes/hooks/<name>/`. Gateway-only. Use gateway event names (`gateway:startup`, `agent:start`, `agent:end`). Do NOT run in CLI.

### Hook kwargs (relevant to this plugin)

**`pre_llm_call` kwargs:**
```python
{
    "session_id": "test-session",
    "user_message": "What is the weather?",
    "conversation_history": [],      # list of {"role": "user"/"assistant", "content": str}
    "is_first_turn": True,
    "model": "glm.short.fast",
    "platform": "cli",               # or "discord", "telegram", etc.
}
```
- Return value: `{"context": str}` to prepend context to the user message, or `None` to do nothing.

**`transform_llm_output` kwargs:**
```python
{
    "response_text": "The LLM's response text",
    "session_id": "test-session",
}
```
- Return value: A non-empty `str` to REPLACE the response text, or `None`/empty to leave unchanged.
- **First non-None string wins** — if multiple plugins register this hook, the first one to return a string gets its value used.

### PluginContext (from `hermes_cli/plugins.py` line 315)

```python
class PluginContext:
    def register_tool(self, name, toolset, schema, handler, check_fn=None,
                      requires_env=None, is_async=False, description="", emoji=""):
        """Register a custom tool accessible via the agent's tool layer."""
    
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a lifecycle hook callback."""
        # Unknown hook names produce a WARNING but are still stored.
        # This means typos silently fail — the hook is stored but never fires
        # because no emit() call will match an unknown event name.
```

### Critical Pitfalls (Known Issues with Plugin Hooks)

1. **API key resolution at register() time:** Hermes loads plugins BEFORE dotenv. `os.getenv("TALKIE_API_KEY")` returns empty during `register()`. The config reader must lazily load `.env` via `python-dotenv` before reading env vars, OR defer all config loading to hook callback time (first use), NOT at `register()` time.

2. **`${VAR}` resolution:** `load_config()` returns raw `${VAR}` refs unresolved for plugin entries. The config reader must manually resolve these: check if value starts with `${` and ends with `}`, extract var name, call `os.getenv(var_name)`.

3. **Shell hooks use PLUGIN event names:** If using shell hooks (config.yaml `hooks:` block), the event names must come from `VALID_HOOKS` (e.g., `pre_llm_call`), NOT gateway event names (e.g., `agent:start`). Gateway names silently fail — no error, hook just doesn't match.

4. **`pre_llm_call` can only inject context, not block.** It returns `{"context": str}` or `None`.

5. **`transform_llm_output` replaces the ENTIRE response.** If it returns a string, that string IS the new response. The original is discarded unless the plugin preserves it.

6. **Talkie finetune model does NOT support the `system` role.** All system prompt instructions must be folded into the user message. (This was the last fix at commit `eece63a`.)

## Plugin Architecture (Current State at `eece63a`)

### Hooks registered:
- `pre_llm_call` (`on_pre_llm_call`): Stashes the last N messages from conversation history into session state for the rewriter to use.
- `transform_llm_output` (`on_transform_llm_output`): Optionally moderates, then rewrites the LLM response through the Talkie model. Returns rewritten text with a flag preamble.

### Tools registered:
- `talkie_set_system_prompt` — Set/update the rewriter's system prompt (per-session)
- `talkie_set_param` — Set generation params (temperature, max_tokens, top_p, etc.)
- `talkie_toggle_moderation` — Enable/disable Qwen3Guard moderation
- `talkie_set_context_depth` — Set how many recent messages to pass as context
- `talkie_get_config` — Read current effective config

### Config flow:
1. `config.py` → `TalkieRewriterConfig.from_hermes_config()` calls `hermes_cli.config.load_config()` and reads `plugins.entries.talkie-rewriter`
2. Falls back to env vars: `TALKIE_API_KEY` for the API key, `PLEXUS_BASE_URL` for the base URL
3. `${VAR}` refs are resolved manually in the `TalkieRewriterConfig.__init__`
4. Config is cached at module level (`_cached_config` singleton)

### Rewrite flow:
1. `pre_llm_call` fires → stashes conversation history + current user message
2. `transform_llm_output` fires → gets stashed messages, calls Talkie model via `openai.OpenAI` client
3. If rewrite succeeds: returns `flag_template + "\n\n" + rewritten_text`
4. If rewrite fails and `fail_open=True`: returns `None` (passes through original)

## Your Tasks

### Phase 1: Investigate Hermes Hook System
- Read `/home/hermes/.hermes/hermes-agent/hermes_cli/plugins.py` — focus on:
  - `VALID_HOOKS` definition (line ~128)
  - `PluginContext` class (line ~315) — how `register_hook()` and `register_tool()` work
  - `PluginManager` — how hooks are emitted (`emit()` or similar)
  - How `transform_llm_output` return values are consumed
- Read `hermes_cli/hooks.py` — shell hook registration, the test harness, kwarg shapes
- Read `hermes_cli/config.py` — search for how `plugins.entries` is parsed and passed to plugins
- **Goal:** Understand exactly what kwargs are passed to each hook, what return values are expected, and the exact lifecycle order.

### Phase 2: Review the Plugin
- Examine each file in `~/.hermes/plugins/talkie-rewriter/`:
  - `__init__.py` — Check hook callbacks receive the correct kwargs, handle edge cases
  - `config.py` — Check config loading, env var resolution, `${VAR}` handling
  - `rewriter.py` — Check the OpenAI client call, message structure, error handling
  - `moderator.py` — Check moderation flow (currently disabled by default)
  - `plugin.yaml` — Check manifest correctness
- **Known issues to look for:**
  1. Does `register()` call any function that touches `load_config()` or `os.getenv()`? (This was a bug in later commits — the fix should ensure register() is config-free)
  2. Are kwargs correctly destructured? (`session_id`, `conversation_history`, `user_message`, `response_text`)
  3. Is the `${TALKIE_API_KEY}` env var actually resolved at hook fire time (not register time)?
  4. Does the `openai` package get imported lazily (inside the function, not at module level)?
  5. Are there any race conditions in the session state (thread safety of `_session_state`)?
  6. Does the `context_messages` truncation work correctly (keeping the LAST N, not the first N)?
  7. Is the flag template prepended correctly?

### Phase 3: Fix Issues
- Fix any bugs found in Phase 2
- Commit with clear message(s)
- Push to `origin` (the repo is `https://github.com/gwyntel/talkie-rewriter.git`)

### Phase 4: Deploy to talkie-agent
- SSH to `talkie-agent` (user: `hermes`)
- The plugin directory is at `~/.hermes/plugins/talkie-rewriter/` on that machine
- Pull the latest code from git (or push from local and pull on the agent)
- Verify `TALKIE_API_KEY` is set in `~/.hermes/.env` on talkie-agent
- Verify the plugin is listed in `plugins.enabled` in `~/.hermes/config.yaml`

### Phase 5: Test End-to-End
- Run: `hermes -z 'Hello, what is 2+2?'` on talkie-agent
- The response should be rewritten through Talkie-1930-13B
- Look for the flag template preamble: `(talkie reflects the culture and values...)`
- Check logs if the rewrite doesn't work (look for `talkie-rewriter:` log lines)
- If it fails to rewrite (fail-open), you should still get the original response
- Try a few more prompts to verify consistency
- Run `hermes -z 'Write a short poem about autumn'` — creative prompts should show more style change

## Important Notes

- **Plexus Gateway URL:** `https://plexus.nebulosa-bass.ts.net/v1` — this is the OpenAI-compatible endpoint for BOTH the host model and the Talkie rewriter model
- **Talkie model name:** `talkie-lm/talkie-1930-13b-it` — this is a finetune deployed via Plexus
- **TALKIE_API_KEY** is a SEPARATE restricted API key from the main Plexus key — it only has access to the Talkie model endpoint
- **The Talkie model does NOT support the `system` role** — all instructions must be in the user message
- **Fail-open is the default** — if the rewrite fails for any reason, the original response passes through
- **The `openai` Python package must be installed** — it's listed in `pip_dependencies` in `plugin.yaml`
- **Hermes loads plugins before dotenv** — any env var reads at `register()` time will get empty values. Config must be loaded lazily at hook fire time.
- **The Hermes source on this machine is at `/home/hermes/.hermes/hermes-agent/hermes_cli/`** — use this as reference for hook system internals
- **On talkie-agent, the Hermes source may be in a different location** — use `python3 -c "import hermes_cli; print(hermes_cli.__file__)"` to find it, or check `/usr/local/lib/hermes-agent/`

## Expected Outcome

1. Any bugs in the plugin are fixed and committed
2. The fix is pushed to GitHub
3. The plugin is deployed to talkie-agent
4. `hermes -z '<prompt>'` on talkie-agent produces a Talkie-rewritten response (flag preamble + rewritten text)
5. A summary of what was found, what was fixed, and test results

## SSH Access

- `ssh talkie-agent` works directly (SSH key configured)
- User is `hermes`
- Hermes binary: `/usr/local/bin/hermes`
- To test: `ssh talkie-agent "hermes -z 'Hello, what is 2+2?'"`
