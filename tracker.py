"""Two Provider Cost Tracker.

Sends the SAME prompt to Google Gemini and Groq, then prints each reply plus a
comparison table (tokens, seconds, USD cost) and saves everything to results.json.

Usage (from the repo root):
    uv run tracker.py "Explain Artificial Intelligence in simple words."

Design rules (from the assignment):
  * Token counts are ALWAYS read from each API's usage data - never estimated.
  * Cost = tokens * price_per_million / 1_000_000, separately for input and output.
  * Secrets live only in .env (git-ignored). Model names come from .env too.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
import groq

RESULTS_FILE = Path(__file__).resolve().parent / "results.json"

PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "gemini": {
        # model: (input_price_per_m, output_price_per_m)
        "gemini-3.5-flash-lite": (0.30, 2.50),
        "gemini-3.1-flash-lite": (0.25, 1.50),
        "gemini-3-flash-preview": (0.50, 3.00),
    },
    "groq": {
        "openai/gpt-oss-20b": (0.075, 0.30),
        "openai/gpt-oss-120b": (0.15, 0.60),
    },
}

REQUEST_TIMEOUT_SECONDS = 60


class TrackerError(Exception):
    """A problem we can explain to the user in one clear sentence."""


@dataclass
class ProviderResult:
    """Everything we measure for one provider call (plus extra display info)."""

    provider: str
    model: str
    reply: str
    input_tokens: int
    output_tokens: int
    seconds: float
    note: str = ""  # shown in the terminal only, e.g. thinking-token breakdown


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def get_env(name: str) -> str | None:
    """Return a stripped env var, treating empty values and the .env.example
    placeholders (your_..._here style) as 'not set'."""
    value = (os.getenv(name) or "").strip()
    if not value or value.startswith("your_"):
        return None
    return value


def check_configuration() -> list[str]:
    """Return a list of human-readable problems with the environment."""
    problems = []
    for var in ("GEMINI_API_KEY", "GEMINI_MODEL", "GROQ_API_KEY", "GROQ_MODEL"):
        if get_env(var) is None:
            problems.append(f"{var} is missing or still a placeholder")
    return problems


def get_prices(provider: str, model: str) -> tuple[float, float]:
    """Look up (input_price_per_m, output_price_per_m) for a model.

    Order: .env override -> PRICING table. If neither has it we stop with an
    error instead of inventing a price.
    """
    prefix = provider.upper()
    override_in = get_env(f"{prefix}_INPUT_PRICE_PER_M")
    override_out = get_env(f"{prefix}_OUTPUT_PRICE_PER_M")
    if override_in is not None and override_out is not None:
        try:
            return float(override_in), float(override_out)
        except ValueError:
            raise TrackerError(
                f"{prefix}_INPUT_PRICE_PER_M / {prefix}_OUTPUT_PRICE_PER_M "
                "must be plain numbers like 0.30"
            )
    try:
        return PRICING[provider][model]
    except KeyError:
        raise TrackerError(
            f"No known price for {provider} model '{model}'. Add it to PRICING "
            f"in tracker.py, or set {prefix}_INPUT_PRICE_PER_M and "
            f"{prefix}_OUTPUT_PRICE_PER_M in .env (see the provider's pricing page)."
        )


# ---------------------------------------------------------------------------
# Provider calls - one function each, returning reply + real token counts
# ---------------------------------------------------------------------------
def call_gemini(prompt: str, api_key: str, model: str) -> ProviderResult:
    """Send the prompt to Google Gemini via the google-genai SDK."""
    client = genai.Client(api_key=api_key)
    start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=prompt)
    except genai_errors.APIError as exc:
        raise TrackerError(explain_gemini_error(exc, model)) from exc
    except Exception as exc:  # network down, DNS failure, timeout, ...
        raise TrackerError(
            f"Could not reach Gemini ({type(exc).__name__}: {exc}). "
            "Check your internet connection and try again."
        ) from exc
    seconds = time.perf_counter() - start

    reply = (response.text or "").strip()
    if not reply:
        raise TrackerError(
            "Gemini returned an empty reply (the response may have been blocked "
            "by a safety filter or cut off). Try a different prompt."
        )

    usage = response.usage_metadata
    if (
        usage is None
        or usage.prompt_token_count is None
        or usage.candidates_token_count is None
    ):
        raise TrackerError(
            "Gemini did not return usage metadata, so the real token counts are "
            "unavailable. Refusing to estimate them."
        )

    # Gemini bills "thinking" tokens as output tokens, but reports them in a
    # separate field. Real output = visible answer tokens + thinking tokens.
    thoughts = usage.thoughts_token_count or 0
    output_tokens = usage.candidates_token_count + thoughts
    note = (
        f"output = {usage.candidates_token_count} answer + {thoughts} thinking tokens"
        if thoughts
        else ""
    )
    return ProviderResult(
        "gemini", model, reply, usage.prompt_token_count, output_tokens, seconds, note
    )


def explain_gemini_error(exc: genai_errors.APIError, model: str) -> str:
    """Turn a Gemini API error into a beginner-friendly message."""
    code = getattr(exc, "code", None)
    text = str(exc)
    if code in (401, 403) or "API key not valid" in text or "API_KEY_INVALID" in text:
        return "Gemini rejected your API key (invalid or not allowed). Check GEMINI_API_KEY in .env."
    if code == 429:
        return "Gemini rate limit or free-tier quota reached. Wait a minute and try again."
    if code == 404:
        return f"Gemini model '{model}' was not found. Check GEMINI_MODEL in .env."
    if code is not None and code >= 500:
        return f"Gemini server error ({code}). Try again in a moment."
    return f"Gemini API error ({code}): {text}"


def call_groq(prompt: str, api_key: str, model: str) -> ProviderResult:
    """Send the prompt to Groq via the official groq SDK."""
    client = groq.Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
    except groq.AuthenticationError as exc:
        raise TrackerError(
            "Groq rejected your API key (invalid). Check GROQ_API_KEY in .env."
        ) from exc
    except groq.RateLimitError as exc:
        raise TrackerError(
            "Groq rate limit reached. Wait a minute and try again."
        ) from exc
    except groq.NotFoundError as exc:
        raise TrackerError(
            f"Groq model '{model}' was not found or is not available to your "
            "account. Check GROQ_MODEL in .env."
        ) from exc
    except groq.APIConnectionError as exc:  # also covers timeouts
        raise TrackerError(
            "Could not reach Groq. Check your internet connection and try again."
        ) from exc
    except groq.APIStatusError as exc:
        raise TrackerError(f"Groq API error ({exc.status_code}): {exc.message}") from exc
    seconds = time.perf_counter() - start

    reply = ""
    if response.choices:
        reply = (response.choices[0].message.content or "").strip()
    if not reply:
        raise TrackerError("Groq returned an empty reply. Try a different prompt.")

    usage = response.usage
    if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
        raise TrackerError(
            "Groq did not return usage information, so the real token counts are "
            "unavailable. Refusing to estimate them."
        )
    # For reasoning models, completion_tokens already includes reasoning tokens.
    return ProviderResult(
        "groq", model, reply, usage.prompt_tokens, usage.completion_tokens, seconds
    )


# ---------------------------------------------------------------------------
# Cost, output and saving
# ---------------------------------------------------------------------------
def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_m: float,
    output_price_per_m: float,
) -> float:
    """cost_usd = input_cost + output_cost, prices are USD per 1M tokens."""
    input_cost = input_tokens * input_price_per_m / 1_000_000
    output_cost = output_tokens * output_price_per_m / 1_000_000
    return input_cost + output_cost


def build_record(result: ProviderResult) -> dict:
    """Build the exact dict that goes into results.json for one provider."""
    in_price, out_price = get_prices(result.provider, result.model)
    return {
        "provider": result.provider,
        "model": result.model,
        "reply": result.reply,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "input_price_per_m": in_price,
        "output_price_per_m": out_price,
        "cost_usd": calculate_cost(
            result.input_tokens, result.output_tokens, in_price, out_price
        ),
    }


def print_table(results: list[ProviderResult], records: list[dict]) -> None:
    """Print the side-by-side comparison table."""
    headers = ["Provider", "Model", "Input tokens", "Output tokens", "Seconds", "Cost (USD)"]
    rows = [
        [
            rec["provider"],
            rec["model"],
            str(rec["input_tokens"]),
            str(rec["output_tokens"]),
            f"{res.seconds:.2f}",
            f"${rec['cost_usd']:.8f}",
        ]
        for res, rec in zip(results, records)
    ]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    def fmt(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    print("\n" + line)
    print(fmt(headers))
    print(line)
    for row in rows:
        print(fmt(row))
    print(line)
    for res in results:
        if res.note:
            print(f"Note ({res.provider}): {res.note}")


def save_results(records: list[dict], path: Path = RESULTS_FILE) -> None:
    """Write the list of per-provider records to results.json."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one prompt to Gemini and Groq and compare tokens, time and cost."
    )
    parser.add_argument("prompt", nargs="+", help="the prompt to send (quotes recommended)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        print("Error: the prompt is empty. Example:\n"
              '  uv run tracker.py "Explain Artificial Intelligence in simple words."',
              file=sys.stderr)
        return 2

    load_dotenv()  # reads .env from the current directory
    problems = check_configuration()
    if problems:
        print("Error: configuration problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("Copy .env.example to .env and fill in your real values.", file=sys.stderr)
        return 2

    gemini_model = get_env("GEMINI_MODEL")
    groq_model = get_env("GROQ_MODEL")
    try:  # fail fast on missing prices BEFORE spending any API calls
        get_prices("gemini", gemini_model)
        get_prices("groq", groq_model)
    except TrackerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    calls = [
        ("Gemini", call_gemini, get_env("GEMINI_API_KEY"), gemini_model),
        ("Groq", call_groq, get_env("GROQ_API_KEY"), groq_model),
    ]
    results: list[ProviderResult] = []
    failed = False
    for label, func, key, model in calls:
        try:
            results.append(func(prompt, key, model))
        except TrackerError as exc:
            failed = True
            print(f"Error [{label}]: {exc}", file=sys.stderr)

    for res in results:
        print(f"\n===== {res.provider.upper()} ({res.model}) =====\n{res.reply}")

    if failed:
        print("\nOne or more providers failed, so results.json was NOT updated.",
              file=sys.stderr)
        return 1

    records = [build_record(r) for r in results]
    print_table(results, records)
    save_results(records)
    print(f"\nSaved {RESULTS_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
