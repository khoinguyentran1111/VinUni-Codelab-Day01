"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Completed solution. Run: pytest tests/ -v
"""

import os
import time
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Load .env (looks in the project root and in the current working directory)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"

# The Anthropic API needs a full model id, not the short pricing key.
_ANTHROPIC_API_IDS = {
    "claude-3-5-haiku": "claude-3-5-haiku-latest",
    "claude-3-5-sonnet": "claude-3-5-sonnet-latest",
}

# Model id actually sent to the Gemini API (gemini-2.5-flash is no longer
# available to new keys). Override it with GEMINI_API_MODEL in .env.
GEMINI_API_MODEL = os.getenv("GEMINI_API_MODEL", "gemini-3.8-flash")


def _calc_cost(model: str, usage: dict) -> float:
    """Cost = (input_tokens * input_rate + output_tokens * output_rate) / 1,000,000."""
    rates = PRICING_1M_TOKENS[model]
    return (
        usage["input_tokens"] * rates["input"]
        + usage["output_tokens"] * rates["output"]
    ) / 1_000_000


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Returns:
        (response_text, latency_seconds, {'input_tokens': int, 'output_tokens': int})
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY") or "mock-key"
    client = OpenAI(api_key=api_key)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time

    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API and return the response text, latency, and
    token usage stats.

    Supports dual-import fallback (new google-genai and legacy
    google-generativeai) to ensure zero-friction execution.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "mock-key"
    # Keep "gemini-2.5-flash" as the logical/pricing name, but call the live model id.
    api_model = GEMINI_API_MODEL if model == GEMINI_MODEL else model

    try:
        # Option A: New Google GenAI SDK (preferred standard)
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

        start_time = time.time()
        response = retry_with_backoff(
            lambda: client.models.generate_content(
                model=api_model,
                contents=prompt,
                config=config,
            ),
            max_retries=3,
            base_delay=1.0,
        )
        latency = time.time() - start_time
    except ImportError:
        # Option B: Legacy Google GenerativeAI SDK
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model_inst = genai.GenerativeModel(api_model)
        config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

        start_time = time.time()
        response = retry_with_backoff(
            lambda: model_inst.generate_content(prompt, generation_config=config),
            max_retries=3,
            base_delay=1.0,
        )
        latency = time.time() - start_time

    text = response.text or ""
    meta = response.usage_metadata
    usage = {
        # Some responses return None for these fields, so default to 0.
        "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(meta, "candidates_token_count", 0) or 0,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API and return the response text, latency,
    and token usage stats.
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY") or "mock-key"
    client = anthropic.Anthropic(api_key=api_key)

    start_time = time.time()
    response = client.messages.create(
        model=_ANTHROPIC_API_IDS.get(model, model),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time

    text = "".join(
        block.text for block in response.content if getattr(block, "type", "text") == "text"
    )
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call gpt-4o, gpt-4o-mini and gemini-2.5-flash with the same prompt and
    return a structured comparison dictionary keyed by model name.

    Each value: {response, latency, input_tokens, output_tokens, cost}
    """
    calls = [
        (OPENAI_MODEL, lambda: call_openai(prompt, model=OPENAI_MODEL)),
        (OPENAI_MINI_MODEL, lambda: call_openai(prompt, model=OPENAI_MINI_MODEL)),
        (GEMINI_MODEL, lambda: call_gemini(prompt, model=GEMINI_MODEL)),
    ]

    results: dict = {}
    for name, fn in calls:
        text, latency, usage = fn()
        results[name] = {
            "response": text,
            "latency": latency,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "cost": _calc_cost(name, usage),
        }
    return results


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini.

    Maintains the last 3 turns of conversation history for context.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    dummy_mode = False
    if not api_key:
        print("\033[93m[System Warning] GEMINI_API_KEY environment variable not set. Running in dummy mode.\033[0m")
        print(f"\033[93m[Hint] Looked for .env in: {Path(__file__).resolve().parent.parent} and {Path.cwd()}\033[0m")
        api_key = "mock-key"
        dummy_mode = True

    print("\n\033[94m==================================================")
    print("🤖 Vin Smart Future — Intelligent Chat Assistant")
    print(f"Powered by Google Gemini ({GEMINI_API_MODEL})")
    print("Type 'quit' or 'exit' to end the session.")
    print("==================================================\033[0m\n")

    # Pick SDK once (new SDK preferred, legacy as fallback)
    sdk = None
    client = None
    model_inst = None
    types = None
    if not dummy_mode:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            sdk = "new"
        except ImportError:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model_inst = genai.GenerativeModel(GEMINI_API_MODEL)
            sdk = "legacy"

    history: list[tuple[str, str]] = []  # (user_message, assistant_reply)
    max_turns = 3

    def build_contents(user_msg: str):
        turns = history[-max_turns:]
        if sdk == "new":
            contents = []
            for u, a in turns:
                contents.append(types.Content(role="user", parts=[types.Part(text=u)]))
                contents.append(types.Content(role="model", parts=[types.Part(text=a)]))
            contents.append(types.Content(role="user", parts=[types.Part(text=user_msg)]))
            return contents
        contents = []
        for u, a in turns:
            contents.append({"role": "user", "parts": [u]})
            contents.append({"role": "model", "parts": [a]})
        contents.append({"role": "user", "parts": [user_msg]})
        return contents

    def stream_reply(user_msg: str):
        if dummy_mode:
            for word in f"[dummy] Bạn vừa nói: {user_msg}".split(" "):
                yield word + " "
                time.sleep(0.03)
            return
        contents = build_contents(user_msg)
        if sdk == "new":
            # Larger limit: newer Gemini models spend tokens on "thinking".
            config = types.GenerateContentConfig(
                temperature=0.7, top_p=0.9, max_output_tokens=4096
            )
            for chunk in client.models.generate_content_stream(
                model=GEMINI_API_MODEL, contents=contents, config=config
            ):
                if chunk.text:
                    yield chunk.text
        else:
            for chunk in model_inst.generate_content(contents, stream=True):
                if chunk.text:
                    yield chunk.text

    while True:
        try:
            user_input = input("\033[92mYou: \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        print("\033[96mBot: \033[0m", end="", flush=True)
        reply_parts: list[str] = []
        last_error = None
        # Retry up to 3 times (e.g. 503 overload) as long as nothing was printed yet.
        for attempt in range(3):
            try:
                for piece in stream_reply(user_input):
                    print(piece, end="", flush=True)
                    reply_parts.append(piece)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if reply_parts:  # already printed partial text: don't retry
                    break
                time.sleep(1.5 * (attempt + 1))

        if last_error is not None:
            print(f"\n\033[91m[Error] {last_error}\033[0m\n")
            continue
        if not reply_parts:
            print("\033[93m[Không nhận được nội dung từ model.]\033[0m\n")
            continue
        print("\n")

        history.append((user_input, "".join(reply_parts)))
        history = history[-max_turns:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).
    Re-raises the last exception if all attempts fail.
    """
    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exception  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results: list[dict] = []
    for prompt in prompts:
        result = dict(compare_models(prompt))
        result["prompt"] = prompt
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Columns:
    | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """

    def clean(text: str, limit: int | None = None) -> str:
        text = str(text).replace("\n", " ").replace("|", "\\|").strip()
        if limit is not None and len(text) > limit:
            text = text[:limit] + "..."
        return text

    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]
    for item in results:
        prompt = clean(item.get("prompt", ""), 50)
        for model_name, stats in item.items():
            if model_name == "prompt" or not isinstance(stats, dict):
                continue
            lines.append(
                f"| {prompt} | {model_name} | {clean(stats.get('response', ''), 50)} "
                f"| {stats.get('latency', 0):.2f}s "
                f"| {stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)} "
                f"| ${stats.get('cost', 0):.6f} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")