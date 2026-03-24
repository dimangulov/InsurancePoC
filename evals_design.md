# Testing Non-Deterministic AI — Eval Design

## The Problem

You cannot unit-test a conversational AI the same way you test deterministic code.

- The same prompt produces different outputs on every run
- "Correct" is often subjective (tone, clarity, empathy)
- The pipeline is multi-step — a bug in the receptionist may only surface in the quote
- Manual testing at scale is impossible: a prompt change could silently break 80% of conversations

The solution is **LLM-assisted evals**: use a second LLM to simulate users and a third to judge the output.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Eval Runner                       │
│                                                      │
│  Scenario + Persona                                  │
│       │                                              │
│       ▼                                              │
│  SimulatedUser (Haiku)  ◄──► Agent Pipeline         │
│       │                       (Sonnet + LangGraph)  │
│       │                              │               │
│       │                       Final State            │
│       │                              │               │
│       └──────────────►  LLM Judge (Haiku)            │
│                               │                      │
│                         EvalScores                   │
└─────────────────────────────────────────────────────┘
```

Three distinct roles, three distinct LLMs:

| Role | Model | Why |
|---|---|---|
| **Agent** | Claude Sonnet 4.6 | Production model — tested as-is |
| **Simulated User** | Claude Haiku 4.5 | Cheap, fast, consistent enough for personas |
| **LLM Judge** | Claude Haiku 4.5 | Cheap qualitative scoring |

---

## Personas

A persona defines *how* a customer communicates, independent of what their business actually is. Separating persona from scenario data lets you cross-test any business against any communication style.

| Persona | What it stress-tests |
|---|---|
| `cooperative` | Happy path — baseline for correctness |
| `grumpy` | Partial answers, pushback — does the agent stay professional and re-ask effectively? |
| `confused` | Vague answers, tangents — does the agent clarify without being condescending? |
| `non_native` | Imprecise language, worded numbers — does the agent parse intent correctly? |

---

## Scenarios

Each scenario is a ground-truth record with:

- **Business profile** — the facts the simulated user will reveal when asked
- **Expected product** — what the classifier *should* route to
- **Persona** — which communication style to use

```python
{
    "id": "consultant_pl",
    "persona": "grumpy",
    "expected_product": "professional_liability",
    "business": {
        "business_name": "Apex Strategy Group",
        "industry": "management consulting — we advise Fortune 500 companies",
        "annual_revenue": 850000,
        ...
    }
}
```

The simulated user only reveals information when asked — it does not volunteer details. This tests whether the agent asks the right questions.

---

## Scoring

Scoring is split into two layers:

### Layer 1 — Rule-based (deterministic)

| Check | How |
|---|---|
| `product_correct` | `quote["product"] == scenario["expected_product"]` |
| `all_fields_collected` | All required `ClientProfile` fields are non-empty |
| `quote_reasonable` | `0 < annual_premium < revenue * 0.15` |

These run instantly with no LLM call and catch the most critical failures.

### Layer 2 — LLM Judge (qualitative)

The judge (Haiku) receives the final quote JSON and the scenario persona, then scores:

| Check | What it catches |
|---|---|
| `agent_professional` | Rude tone, confusing language, going off-script |
| `handled_persona` | Impatient with a confused customer, weak with a grumpy one |

The judge uses `tool_use` with a strict schema to return structured scores — no free-form output to parse.

### `passed` definition

A run `passed` if and only if all three rule-based checks pass:
- product routed correctly
- all required fields collected
- premium is in a sane range

Qualitative scores (tone, persona handling) are reported separately — they inform prompt tuning but do not gate pass/fail.

---

## What Gets Tested

```
Receptionist → Classifier → Specialist → Underwriter
     ↑               ↑            ↑            ↑
 field        product       gap          quote
 collection   routing       detection    sanity
```

| Failure mode | Detected by |
|---|---|
| Receptionist fails to extract a field from a vague answer | `all_fields_collected` |
| Classifier routes to the wrong product | `product_correct` |
| Gap detection misses a missing field | `all_fields_collected` on underwriter output |
| Underwriter generates an absurd premium | `quote_reasonable` |
| Agent becomes rude with a grumpy customer | `agent_professional` (judge) |
| Agent confuses a confused customer further | `handled_persona` (judge) |

---

## How the Injection Works

`main.py` exposes a module-level hook:

```python
_user_input_fn: Callable | None = None
```

In production: `None` → `input("You: ")` is called normally.

In evals: replaced with `SimulatedUser.respond` before the graph runs, then restored in `finally`:

```python
agent._user_input_fn = sim_user.respond
try:
    graph.invoke(...)
finally:
    agent._user_input_fn = None   # always restore, even on exception
```

The graph uses `MemorySaver` in evals instead of `PostgresSaver` — no database connection required to run the test suite.

---

## Running Evals

```bash
# All scenarios (4 runs total)
python evals.py

# Single scenario
python evals.py --scenario bakery_gl

# Repeat each scenario 5 times — surfaces non-determinism
python evals.py --runs 5

# Full regression: 20 runs across all scenarios
python evals.py --runs 5
```

Results are printed to the console and saved to `eval_results.json`.

### Sample output

```
--- consultant_pl run #1 (grumpy) ---
[PASS] consultant_pl run#1 (grumpy)  product:✓  fields:✓  premium:✓  tone:✓  persona:✓  — Agent handled pushback well.

--- trucking_ca run #1 (confused) ---
[FAIL] trucking_ca run#1 (confused)  product:✓  fields:✗  premium:✓  tone:✓  persona:✓  — state_or_province was empty in final profile.

============================================================
RESULTS: 3/4 passed
  product routing:    4/4
  field completeness: 3/4
  premium sanity:     4/4
  agent tone:         4/4
  persona handling:   4/4
============================================================
```

---

## When to Run

| Event | Action |
|---|---|
| Any prompt change | Full eval suite (`--runs 3`) |
| New product added | Add matching scenario + update `PRODUCT_REQUIRED_FIELDS` |
| New persona complaint from real users | Add a persona, add a scenario |
| Pre-release | `--runs 10` across all scenarios |

---

## Extending

**Add a scenario:** append to `SCENARIOS` in `evals.py` with a new business profile and expected product.

**Add a persona:** append to `PERSONAS` with a natural-language description of the communication style.

**Add a scoring dimension:** extend `EvalScores` with a new `bool` field and add the check to `score_run()`.

**Parallelise:** wrap `run_single()` calls with `concurrent.futures.ThreadPoolExecutor` — each run is independent.
