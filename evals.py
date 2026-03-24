"""
Eval suite — Virtual Insurance Agency

Replaces the human user with a SimulatedUser (Haiku-backed LLM) and scores
the final quote against ground-truth scenarios using an LLM judge.

Usage:
    python evals.py                  # run all scenarios
    python evals.py --scenario bakery_gl
    python evals.py --runs 5         # repeat each scenario N times
"""

import argparse
import json
import uuid
from dataclasses import dataclass, field

from anthropic import Anthropic
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

import main as agent

JUDGE_MODEL = "claude-haiku-4-5-20251001"
SIM_MODEL   = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------

PERSONAS: dict[str, str] = {
    "cooperative": (
        "You are a cooperative small business owner who answers questions clearly and "
        "completely. You are friendly and provide all requested information without hesitation."
    ),
    "grumpy": (
        "You are a grumpy, impatient business owner. You give short, sometimes incomplete "
        "answers. You occasionally push back ('Why do you need that?') but eventually provide "
        "the information when pressed. You want to get this done quickly."
    ),
    "confused": (
        "You are an elderly business owner who is not very tech-savvy. You often misunderstand "
        "questions, give vague answers, and sometimes go off on tangents about unrelated things. "
        "You need things explained more than once before you understand."
    ),
    "non_native": (
        "Your English is limited. You mix up words, give partial answers, and sometimes "
        "misunderstand idiomatic expressions. You are polite but struggle to express "
        "exact numbers — you say things like 'maybe three hundred, little more?'"
    ),
}

# ---------------------------------------------------------------------------
# Test scenarios — ground truth
# ---------------------------------------------------------------------------

SCENARIOS: list[dict] = [
    {
        "id": "bakery_gl",
        "persona": "cooperative",
        "expected_product": "general_liability",
        "business": {
            "business_name": "Sweet Dreams Bakery",
            "owner_name": "Maria Santos",
            "email": "maria@sweetdreams.com",
            "phone": "555-0101",
            "city": "Austin",
            "state_or_province": "TX",
            "country": "US",
            "postal_code": "78701",
            "industry": "bakery — we make and sell pastries and cakes",
            "employees": 8,
            "annual_revenue": 320000,
            "years_in_business": 5,
        },
    },
    {
        "id": "consultant_pl",
        "persona": "grumpy",
        "expected_product": "professional_liability",
        "business": {
            "business_name": "Apex Strategy Group",
            "owner_name": "James Chen",
            "email": "james@apexstrategy.com",
            "phone": "555-0202",
            "city": "Chicago",
            "state_or_province": "IL",
            "country": "US",
            "postal_code": "60601",
            "industry": "management consulting — we advise Fortune 500 companies",
            "employees": 3,
            "annual_revenue": 850000,
            "years_in_business": 7,
        },
    },
    {
        "id": "trucking_ca",
        "persona": "confused",
        "expected_product": "commercial_auto",
        "business": {
            "business_name": "Fast Freight LLC",
            "owner_name": "Bob Kowalski",
            "email": "bob@fastfreight.com",
            "phone": "555-0303",
            "city": "Detroit",
            "state_or_province": "MI",
            "country": "US",
            "postal_code": "48201",
            "industry": "we drive trucks, deliver things for other companies",
            "employees": 12,
            "annual_revenue": 1200000,
            "years_in_business": 10,
        },
    },
    {
        "id": "construction_wc",
        "persona": "non_native",
        "expected_product": "workers_comp",
        "business": {
            "business_name": "Bright Build Co",
            "owner_name": "Andrei Popescu",
            "email": "andrei@brightbuild.com",
            "phone": "555-0404",
            "city": "Houston",
            "state_or_province": "TX",
            "country": "US",
            "postal_code": "77001",
            "industry": "construction — we build houses and renovate buildings",
            "employees": 22,
            "annual_revenue": 2100000,
            "years_in_business": 4,
        },
    },
]

# ---------------------------------------------------------------------------
# Simulated user
# ---------------------------------------------------------------------------

class SimulatedUser:
    """Haiku-backed customer that responds based on a persona and business scenario."""

    def __init__(self, persona_key: str, scenario: dict):
        self._client = Anthropic()
        biz = json.dumps(scenario["business"], indent=2)
        self._system = (
            f"{PERSONAS[persona_key]}\n\n"
            "Your business details (use these when asked — do NOT volunteer info unprompted):\n"
            f"{biz}\n\n"
            "Describe your industry naturally, never use technical codes. "
            "Express revenue in words if your persona calls for it."
        )
        self._history: list[dict] = []

    def respond(self, agent_message: str) -> str:
        self._history.append({"role": "user", "content": agent_message})
        response = self._client.messages.create(
            model=SIM_MODEL,
            max_tokens=200,
            system=self._system,
            messages=self._history,
        )
        reply = response.content[0].text.strip()
        self._history.append({"role": "assistant", "content": reply})
        return reply

# ---------------------------------------------------------------------------
# Scoring — rule-based + LLM judge
# ---------------------------------------------------------------------------

class EvalScores(BaseModel):
    product_correct:      bool  = Field(description="Routed to the expected insurance product?")
    all_fields_collected: bool  = Field(description="Client profile has no empty required fields?")
    quote_reasonable:     bool  = Field(description="Premium is plausible for the business size?")
    agent_professional:   bool  = Field(description="Agent maintained professional, helpful tone?")
    handled_persona:      bool  = Field(description="Agent handled the persona appropriately (patient, clear)?")
    notes:                str   = Field(description="One-sentence summary of any issues found.")


def score_run(scenario: dict, final_state: dict) -> EvalScores:
    quote   = final_state.get("quote", {})
    profile = quote.get("client", {})

    # Rule-based checks
    required_fields    = ["business_name", "owner_name", "email", "city"]
    all_fields         = all(profile.get(f) for f in required_fields)
    # Underwriter may return a display name ("General Liability Insurance") rather
    # than the snake_case key — normalise both sides before comparing.
    actual_product  = quote.get("product", "").lower().replace(" ", "_").replace("insurance", "").strip("_")
    expected        = scenario["expected_product"].lower()
    product_correct = expected in actual_product or actual_product in expected
    premium            = quote.get("annual_premium", 0)
    revenue            = scenario["business"].get("annual_revenue", 1)
    quote_reasonable   = 0 < premium < (revenue * 0.15)   # premium < 15% of revenue

    # LLM judge for qualitative dimensions
    client  = Anthropic()
    prompt  = (
        f"Evaluate this insurance agent run.\n\n"
        f"Scenario persona: {scenario['persona']}\n"
        f"Expected product: {scenario['expected_product']}\n\n"
        f"Final quote:\n{json.dumps(quote, indent=2)}\n\n"
        "Score agent_professional and handled_persona. "
        "agent_professional: did the agent stay on topic and avoid rude or confusing language? "
        "handled_persona: was the agent appropriately patient with a confused customer, "
        "or appropriately firm with a grumpy one?"
    )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        tools=[{
            "name":        "submit_scores",
            "description": "Submit qualitative evaluation scores",
            "input_schema": {
                "type": "object",
                "properties": {
                    "agent_professional": {"type": "boolean"},
                    "handled_persona":    {"type": "boolean"},
                    "notes":              {"type": "string"},
                },
                "required": ["agent_professional", "handled_persona", "notes"],
            },
        }],
        tool_choice={"type": "tool", "name": "submit_scores"},
        messages=[{"role": "user", "content": prompt}],
    )
    args = response.content[0].input

    return EvalScores(
        product_correct      = product_correct,
        all_fields_collected = all_fields,
        quote_reasonable     = quote_reasonable,
        agent_professional   = args.get("agent_professional", True),
        handled_persona      = args.get("handled_persona", True),
        notes                = args.get("notes", ""),
    )

# ---------------------------------------------------------------------------
# Single eval run
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    scenario_id: str
    persona:     str
    run:         int
    scores:      EvalScores
    quote:       dict = field(default_factory=dict)
    error:       str  = ""

    @property
    def passed(self) -> bool:
        s = self.scores
        return s.product_correct and s.all_fields_collected and s.quote_reasonable

    def summary_line(self) -> str:
        s = self.scores
        checks = {
            "product":    s.product_correct,
            "fields":     s.all_fields_collected,
            "premium":    s.quote_reasonable,
            "tone":       s.agent_professional,
            "persona":    s.handled_persona,
        }
        icons = {k: "OK" if v else "FAIL" for k, v in checks.items()}
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.scenario_id} run#{self.run} ({self.persona})  "
            + "  ".join(f"{k}:{v}" for k, v in icons.items())
            + f"  — {self.scores.notes}"
        )


def run_single(scenario: dict, run_index: int) -> EvalResult:
    sim_user = SimulatedUser(scenario["persona"], scenario)

    # Inject simulated user — cleared after the run
    agent._user_input_fn = sim_user.respond
    try:
        graph        = agent.build_graph(MemorySaver())
        session_id   = str(uuid.uuid4())
        final_state  = graph.invoke(
            agent.State(session_id=session_id),
            config={"configurable": {"thread_id": session_id}},
        )
        scores = score_run(scenario, final_state)
        return EvalResult(
            scenario_id = scenario["id"],
            persona     = scenario["persona"],
            run         = run_index,
            scores      = scores,
            quote       = final_state.get("quote", {}),
        )
    except Exception as exc:
        return EvalResult(
            scenario_id = scenario["id"],
            persona     = scenario["persona"],
            run         = run_index,
            scores      = EvalScores(
                product_correct=False, all_fields_collected=False,
                quote_reasonable=False, agent_professional=False,
                handled_persona=False, notes=str(exc),
            ),
            error       = str(exc),
        )
    finally:
        agent._user_input_fn = None   # always restore for next run

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run insurance agent evals")
    parser.add_argument("--scenario", default=None, help="Run a single scenario by id")
    parser.add_argument("--runs",     type=int, default=1, help="Repeat each scenario N times")
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s["id"] == args.scenario]
        if not scenarios:
            print(f"Unknown scenario '{args.scenario}'. Available: {[s['id'] for s in SCENARIOS]}")
            return

    results: list[EvalResult] = []
    total = len(scenarios) * args.runs
    print(f"\nRunning {total} eval(s) across {len(scenarios)} scenario(s)...\n")

    for scenario in scenarios:
        for run_i in range(1, args.runs + 1):
            print(f"--- {scenario['id']} run #{run_i} ({scenario['persona']}) ---")
            result = run_single(scenario, run_i)
            results.append(result)
            print(result.summary_line())
            print()

    # Aggregate report
    passed = sum(1 for r in results if r.passed)
    print("=" * 60)
    print(f"RESULTS: {passed}/{len(results)} passed")
    print(f"  product routing:    {sum(r.scores.product_correct      for r in results)}/{len(results)}")
    print(f"  field completeness: {sum(r.scores.all_fields_collected  for r in results)}/{len(results)}")
    print(f"  premium sanity:     {sum(r.scores.quote_reasonable      for r in results)}/{len(results)}")
    print(f"  agent tone:         {sum(r.scores.agent_professional    for r in results)}/{len(results)}")
    print(f"  persona handling:   {sum(r.scores.handled_persona       for r in results)}/{len(results)}")
    print("=" * 60)

    # Save full results
    out = [
        {
            "scenario_id": r.scenario_id,
            "persona":     r.persona,
            "run":         r.run,
            "passed":      r.passed,
            "scores":      r.scores.model_dump(),
            "error":       r.error,
        }
        for r in results
    ]
    with open("eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results saved to eval_results.json")


if __name__ == "__main__":
    main()
