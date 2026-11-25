# Two Provider Cost Tracker

A small Python command-line tool that sends **the same prompt** to two LLM providers
(**Google Gemini** and **Groq**) and shows their replies side by side with the
**real token counts, response time and cost in USD** of each call.

## Features

- One prompt in, two providers called, identical text sent to both
- Token counts come straight from each API's usage data (never estimated)
- Each call is timed separately with `time.perf_counter()`
- Cost calculated per provider from prices per 1M tokens (configurable)
- Comparison table printed in the terminal
- `results.json` written automatically after every successful run
- Clear error messages: missing/invalid key, rate limit, no internet, empty prompt,
  missing usage data, empty reply

## Technologies

Python 3.12+, [uv](https://docs.astral.sh/uv/), `google-genai` (Gemini),
`groq` (Groq), `python-dotenv`, JSON, Git/GitHub.

## Project structure

```
two-provider-cost-tracker/
├── tracker.py        # the whole program
├── pyproject.toml    # dependencies (managed by uv)
├── uv.lock           # exact dependency versions
├── .env.example      # template with FAKE values (committed)
├── .env              # your real keys (NEVER committed, git-ignored)
├── .gitignore
├── README.md
└── results.json      # created/overwritten on every successful run
```

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- A free Gemini API key and a free Groq API key (see below)

## Installation

```bash
git clone <your-repo-url>
cd two-provider-cost-tracker
uv sync          # installs the dependencies from pyproject.toml / uv.lock
```

(Starting from scratch instead? `uv init`, then
`uv add google-genai groq python-dotenv`.)

## API key configuration

1. **Gemini key** – go to [Google AI Studio](https://aistudio.google.com/apikey),
   sign in, click **Create API key**.
2. **Groq key** – go to the [GroqCloud console](https://console.groq.com/keys),
   sign up, click **Create API Key** (the key is shown only once).
3. Copy the template and paste your keys:

```bash
cp .env.example .env      # Windows PowerShell: Copy-Item .env.example .env
```

```
GEMINI_API_KEY=paste_your_real_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
GROQ_API_KEY=paste_your_real_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

`.env` is listed in `.gitignore`, so it is never uploaded. **Never paste keys into
the README, screenshots, or Google Classroom.**

## Run it

From the repository root:

```bash
uv run tracker.py "Explain Artificial Intelligence in simple words."
```

## Sample terminal output

> **Illustrative only.** These numbers are made up to show the layout. They are
> **not** real API results. Your own run will print different replies, tokens and times.

```
===== GEMINI (gemini-3.5-flash-lite) =====
<the full Gemini reply is printed here>

===== GROQ (openai/gpt-oss-20b) =====
<the full Groq reply is printed here>

+----------+-----------------------+--------------+---------------+---------+--------------+
| Provider | Model                 | Input tokens | Output tokens | Seconds | Cost (USD)   |
+----------+-----------------------+--------------+---------------+---------+--------------+
| gemini   | gemini-3.5-flash-lite | 12           | 180           | 1.84    | $0.00045360  |
| groq     | openai/gpt-oss-20b    | 78           | 310           | 0.62    | $0.00009885  |
+----------+-----------------------+--------------+---------------+---------+--------------+

Saved results.json
```

`results.json` holds one object per provider with the fields: `provider`, `model`,
`reply`, `input_tokens`, `output_tokens`, `input_price_per_m`, `output_price_per_m`,
`cost_usd`.

## How the cost is calculated

Pricing pages list **USD per 1 million tokens**, so we divide by 1,000,000:

```
input_cost  = input_tokens  * input_price_per_m  / 1_000_000
output_cost = output_tokens * output_price_per_m / 1_000_000
cost_usd    = input_cost + output_cost
```

Prices live in the `PRICING` table at the top of `tracker.py` (or can be overridden
in `.env`). Prices used, checked **25 November 2026**:

| Provider | Model | Input $/1M | Output $/1M | Source |
|----------|-------|-----------:|------------:|--------|
| Gemini | `gemini-3.5-flash-lite` | 0.30 | 2.50 | <https://ai.google.dev/gemini-api/docs/pricing> (Standard tier) |
| Groq | `openai/gpt-oss-20b` | 0.075 | 0.30 | <https://console.groq.com/docs/models> |

Notes you may be asked about:

- **Free tiers:** both providers let you call these models for free within rate
  limits, so you will not actually be billed. The tracker shows what the call
  *would* cost at paid prices.
- **Thinking tokens:** these models can "think" before answering. Gemini reports
  thinking tokens separately from the visible answer but bills them as output, so
  `output_tokens` = answer tokens + thinking tokens (the terminal prints the
  breakdown). Groq's `completion_tokens` already includes reasoning tokens.
- **Why not `llama-3.1-8b-instant` on Groq?** Groq moved it to Enterprise
  "Contact Sales" pricing in 2026, so it has no public price and is not on the
  free plan. Many older tutorials still list it.
- Prices change. Re-check the pricing pages before relying on any number.

## What I Learned

*(Replace this with your own words before submitting.)*
The same sentence produces different token counts on different providers because
each model has its own tokenizer, so token counts must be read from the API and
never guessed. Output tokens cost several times more than input tokens, and
"thinking" tokens are billed as output, which can make a short answer more
expensive than it looks. Pricing and model availability change often, so prices
belong in configuration with a documented source and date.
