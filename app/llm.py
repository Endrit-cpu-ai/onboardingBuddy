"""One place for every LLM call: timeouts, automatic retries, and safe JSON parsing.
If we ever swap vendors (e.g. GPT via Azure), this is the only file that changes."""
import json
import logging
import time
from functools import lru_cache

import anthropic

from app.config import ANTHROPIC_API_KEY, LLM_RETRIES, LLM_TIMEOUT

log = logging.getLogger("llm")


@lru_cache(maxsize=1)
def _client():
    # The SDK retries rate limits, overloads and network errors itself, with backoff.
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=LLM_TIMEOUT, max_retries=LLM_RETRIES)


def complete(model, system, user, max_tokens=1024, effort=None):
    """Send one message, return the reply text.
    effort: "low" / "medium" / "high" for models that support it (Sonnet 5); leave None for Haiku."""
    kwargs = {"model": model, "max_tokens": max_tokens, "system": system,
              "messages": [{"role": "user", "content": user}]}
    if effort:
        kwargs["output_config"] = {"effort": effort}
    t = time.time()
    msg = _client().messages.create(**kwargs)
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    log.info("%s: %d in / %d out tokens, %.1fs", model, msg.usage.input_tokens, msg.usage.output_tokens, time.time() - t)
    return text


def complete_json(model, system, user, max_tokens=500, default=None):
    """Like complete(), but expects a JSON object back. Returns `default` instead of crashing if the reply isn't valid JSON."""
    text = complete(model, system, user, max_tokens=max_tokens)
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except ValueError:
        log.warning("%s returned non-JSON: %r", model, text[:200])
        return default


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    from app.config import ROUTER_MODEL
    print(complete(ROUTER_MODEL, "Reply in five words or fewer.", "Say hello to a new joiner."))
    print(complete_json(ROUTER_MODEL, "Reply with JSON only.", 'Return {"ok": true, "language": "<language of: Si je?>"}'))