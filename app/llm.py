"""All LLM calls go through here (timeouts, retries, JSON parsing).
Supports OpenAI and Anthropic, picked with LLM_PROVIDER in config."""
import json
import logging
import time
from functools import lru_cache

from app.config import (ANTHROPIC_API_KEY, FAST_EFFORT, LLM_PROVIDER, LLM_RETRIES, LLM_TIMEOUT,
                        OPENAI_API_KEY)

log = logging.getLogger("llm")

# reasoning tokens count toward the output limit on OpenAI, so leave room for them
REASONING_HEADROOM = 4000


@lru_cache(maxsize=1)
def _client():
    # both SDKs handle retries with backoff
    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT, max_retries=LLM_RETRIES)
    import anthropic
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=LLM_TIMEOUT, max_retries=LLM_RETRIES)


def _openai(model, system, user, max_tokens, effort, json_mode):
    kwargs = {"model": model, "instructions": system, "input": user, "max_output_tokens": max_tokens}
    if effort:
        kwargs["reasoning"] = {"effort": effort}
        if effort != "none":
            kwargs["max_output_tokens"] += REASONING_HEADROOM
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}
        kwargs["input"] = user + "\n\nReply with JSON."   # openai wants "json" in the input itself
    resp = _client().responses.create(**kwargs)
    return resp.output_text.strip(), resp.usage.input_tokens, resp.usage.output_tokens


def _anthropic(model, system, user, max_tokens, effort):
    kwargs = {"model": model, "max_tokens": max_tokens, "system": system,
              "messages": [{"role": "user", "content": user}]}
    if effort:
        kwargs["output_config"] = {"effort": effort}
    msg = _client().messages.create(**kwargs)
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    return text, msg.usage.input_tokens, msg.usage.output_tokens


def complete(model, system, user, max_tokens=1024, effort=None, json_mode=False):
    """effort: OpenAI none/low/medium/high, Anthropic only on models that support it (not Haiku)."""
    t = time.time()
    if LLM_PROVIDER == "openai":
        text, tokens_in, tokens_out = _openai(model, system, user, max_tokens, effort, json_mode)
    else:
        text, tokens_in, tokens_out = _anthropic(model, system, user, max_tokens, effort)
    log.info("%s: %d in / %d out tokens, %.1fs", model, tokens_in, tokens_out, time.time() - t)
    return text


def complete_json(model, system, user, max_tokens=500, default=None, effort=FAST_EFFORT):
    """Like complete() but parses a JSON object. Returns `default` if the reply isn't valid JSON."""
    text = complete(model, system, user, max_tokens=max_tokens, effort=effort, json_mode=True)
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except ValueError:
        log.warning("%s returned non-JSON: %r", model, text[:200])
        return default


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    from app.config import GEN_EFFORT, GEN_MODEL, ROUTER_MODEL
    print(complete(ROUTER_MODEL, "Reply in five words or fewer.", "Say hello to a new joiner.", effort=FAST_EFFORT))
    print(complete_json(ROUTER_MODEL, "Reply with JSON only.", 'Return {"ok": true, "language": "<language of: Si je?>"}'))
    print(complete(GEN_MODEL, "Reply in one sentence.", "What is RAG?", effort=GEN_EFFORT))
