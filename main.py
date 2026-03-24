"""
Virtual Insurance Agency — LangGraph version (typed LLM edition)

Same 4-stage pipeline. All LLM I/O goes through Pydantic models:
  - No JSON format instructions in prompts
  - Completion signalled by tool_use, not {"done": true, "data": ...}
  - _converse and _call_once both accept a model_cls and return a typed instance

Graph (linear):
  [receptionist] → [classifier] → [specialist] → [underwriter] → END
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import TypeVar, Type, Callable
from pydantic import create_model
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, field_validator, model_validator, ValidationError
import db
from pricing import PremiumCalculator
from auditor  import audit_quote

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Injectable input function — override for evals/testing (see evals.py)
# ---------------------------------------------------------------------------
# Default: real terminal input. Evals replace this with a SimulatedUser.respond.
_user_input_fn: Callable | None = None

# LangSmith tracing — enabled automatically when these env vars are set:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=ls__...
#   LANGCHAIN_PROJECT=InsurancePoC   (optional, groups runs in the UI)

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 15
llm = ChatAnthropic(model=MODEL, max_tokens=1024)

# ---------------------------------------------------------------------------
# Pre-defined lookup tables used by validators
# ---------------------------------------------------------------------------

INDUSTRY_TYPES: list[str] = [
    "accounting_and_bookkeeping",
    "agriculture_and_farming",
    "auto_repair_and_maintenance",
    "bakery_and_food_production",
    "beauty_and_personal_care",
    "cleaning_services",
    "clothing_and_apparel",
    "construction_general",
    "consulting_business",
    "consulting_it",
    "consulting_management",
    "dental_practice",
    "e_commerce",
    "education_and_tutoring",
    "electrical_contracting",
    "entertainment_and_events",
    "financial_services",
    "fitness_and_wellness",
    "flooring_and_tile",
    "food_truck_and_catering",
    "freight_and_trucking",
    "funeral_services",
    "graphic_design_and_marketing",
    "health_and_medical",
    "home_inspection",
    "hospitality_and_hotels",
    "hvac_services",
    "insurance_agency",
    "it_services_and_msp",
    "janitorial_and_commercial_cleaning",
    "jewelry_and_accessories",
    "landscape_and_lawn_care",
    "legal_services",
    "manufacturing_light",
    "media_and_publishing",
    "moving_and_storage",
    "nonprofit",
    "pest_control",
    "pet_services",
    "photography_and_videography",
    "plumbing_services",
    "printing_and_signage",
    "property_management",
    "real_estate",
    "restaurant_and_bar",
    "retail_general",
    "roofing_and_siding",
    "security_services",
    "staffing_and_hr",
    "veterinary_services",
]

# ---------------------------------------------------------------------------
# Pydantic models — all LLM I/O is typed
# ---------------------------------------------------------------------------

class ClientProfile(BaseModel):
    business_name: str = ""
    owner_name: str = ""
    email: str = ""
    phone: str = ""
    city: str = ""
    state_or_province: str = ""
    country: str = ""
    postal_code: str = ""       # any format (US ZIP, UK postcode, etc.)
    industry_type: str = ""     # validated: must be a key in INDUSTRY_TYPES
    annual_revenue: int = 0     # validated: positive integer (LLM converts "eighty thousand" → 80000)
    employees: int = 0
    years_in_business: int = 0

    @field_validator("industry_type")
    @classmethod
    def validate_industry_type(cls, v: str) -> str:
        if v and v not in INDUSTRY_TYPES:
            close = [t for t in INDUSTRY_TYPES if v.lower().replace(" ", "_") in t or t in v.lower()]
            hint = f" Did you mean one of: {close[:3]}?" if close else ""
            raise ValueError(
                f"'{v}' is not a recognised industry type.{hint} "
                f"Must be one of the {len(INDUSTRY_TYPES)} accepted values."
            )
        return v

    @field_validator("annual_revenue")
    @classmethod
    def validate_annual_revenue(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f"{v} is not a valid annual revenue. Must be a positive integer (e.g. 80000)."
            )
        return v


class ClassificationResult(BaseModel):
    product: str   # one of the four product keys
    reason: str    # one sentence


class GeneralLiabilityData(BaseModel):
    clients_on_premises: bool
    handles_property: bool
    prior_claims: int
    state: str
    coverage_limit: str


class ProfessionalLiabilityData(BaseModel):
    services: str
    largest_contract: str
    written_contracts: bool
    prior_claims: int
    coverage_limit: str


class CommercialAutoData(BaseModel):
    vehicle_count: int
    vehicle_types: str
    primary_use: str
    young_drivers: bool
    incidents: int


class WorkersCompData(BaseModel):
    job_classes: str
    states: str
    prior_injuries: int
    uses_subs: bool
    payroll: str


class InsuranceQuote(BaseModel):
    quote_id:        str
    product:         str
    coverage_limit:  str
    annual_premium:  int
    monthly_premium: int
    deductible:      int
    exclusions:      list[str]
    notes:           str
    valid_days:      int = 30

    @field_validator("annual_premium")
    @classmethod
    def premium_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"annual_premium must be > 0, got {v}")
        return v

    @field_validator("monthly_premium")
    @classmethod
    def monthly_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"monthly_premium must be > 0, got {v}")
        return v

    @field_validator("deductible")
    @classmethod
    def deductible_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"deductible must be >= 0, got {v}")
        return v

    @model_validator(mode="after")
    def monthly_consistent_with_annual(self) -> "InsuranceQuote":
        import math
        expected = math.ceil(self.annual_premium / 12)
        if abs(self.monthly_premium - expected) > 1:
            raise ValueError(
                f"monthly_premium {self.monthly_premium} inconsistent with "
                f"annual_premium {self.annual_premium} (expected ~{expected})"
            )
        return self


class LLMQuoteContent(BaseModel):
    """What the underwriter LLM is permitted to decide — no financial fields."""
    quote_id:       str
    coverage_limit: str
    exclusions:     list[str]
    notes:          str


# Map product key → specialist model class
UNDERWRITING_MODELS: dict[str, type[BaseModel]] = {
    "general_liability":      GeneralLiabilityData,
    "professional_liability": ProfessionalLiabilityData,
    "commercial_auto":        CommercialAutoData,
    "workers_comp":           WorkersCompData,
}

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# System prompts — no JSON format instructions needed
# ---------------------------------------------------------------------------

RECEPTIONIST_PROMPT = f"""You are a professional insurance agency receptionist.
Collect the following from the client through natural conversation:
  1. Business name
  2. Owner / contact name
  3. Email address
  4. Phone number
  5. City, state/province, country, and postal code
  6. Industry type — you MUST map what they say to one of the accepted values listed below.
     Accepted values: {", ".join(INDUSTRY_TYPES)}
     Example: "bakery" → bakery_and_food_production
  7. Number of employees
  8. Annual revenue — convert written-out amounts to integers (e.g. "eighty thousand" → 80000)
  9. Years in business

Ask 1-2 questions at a time. Be friendly but concise.
When you have ALL nine items, call the submit_data tool.
"""

CLASSIFIER_PROMPT = """You are an insurance product classifier.
Given a business profile, choose the single best insurance product.

Products:
  general_liability      — physical premises, customers visit, property risk
  professional_liability — advice / consulting / services, errors & omissions
  commercial_auto        — vehicles used for business operations
  workers_comp           — employees doing physical or hazardous work

Call the submit_result tool with your classification.
"""

SPECIALIST_PROMPTS = {
    "general_liability": """You are a General Liability underwriting specialist.
Collect these details in natural conversation:
  1. Do clients visit your premises?
  2. Do you handle third-party property?
  3. Prior claims in the last 3 years?
  4. Primary business state?
  5. Desired coverage limit — $1M, $2M, or $5M?

Ask 1-2 questions at a time. When you have all five, call the submit_data tool.
""",
    "professional_liability": """You are a Professional Liability (E&O) underwriting specialist.
Collect in natural conversation:
  1. Specific professional services provided?
  2. Largest single contract value?
  3. Do you use written contracts with all clients?
  4. Prior E&O claims in last 3 years?
  5. Desired coverage limit — $500K, $1M, or $2M?

Ask 1-2 questions at a time. When you have all five, call the submit_data tool.
""",
    "commercial_auto": """You are a Commercial Auto underwriting specialist.
Collect in natural conversation:
  1. Number of business vehicles?
  2. Vehicle types (sedans, trucks, vans, etc.)?
  3. Primary use (delivery, sales, transport)?
  4. Any drivers under 25?
  5. Accidents or violations in the last 3 years?

Ask 1-2 questions at a time. When you have all five, call the submit_data tool.
""",
    "workers_comp": """You are a Workers Compensation underwriting specialist.
Collect in natural conversation:
  1. Job classifications (office, field, construction, etc.)?
  2. States where employees work?
  3. Workplace injuries in last 3 years?
  4. Do you use subcontractors?
  5. Total annual payroll?

Ask 1-2 questions at a time. When you have all five, call the submit_data tool.
""",
}

UNDERWRITER_PROMPT = """You are a senior insurance underwriter.
Given a client profile and underwriting details, produce the descriptive parts of a quote.
You decide: quote_id, coverage_limit, exclusions, and notes.
Do NOT set premiums, deductibles, or any financial figures — those are computed separately.

quote_id format: use the product prefix (GL, PL, CA, WC) + client abbreviation + year + sequence.
  Example: "GL-SWEETDREAMS-2024-001"
coverage_limit: use the value the client chose during the specialist conversation.
exclusions: list 3–6 standard exclusions appropriate for this product and business type.
notes: 1–2 sentences about underwriting considerations for this account.

Call the submit_result tool with the completed content.
"""

# ---------------------------------------------------------------------------
# One-shot chains — prompt template + structured output, no helper needed
# ---------------------------------------------------------------------------

CLASSIFIER_CHAIN = (
    ChatPromptTemplate.from_messages([("system", CLASSIFIER_PROMPT), ("human", "{profile}")])
    | llm.with_structured_output(ClassificationResult)
)

UNDERWRITER_CHAIN = (
    ChatPromptTemplate.from_messages([("system", UNDERWRITER_PROMPT), ("human", "{details}")])
    | llm.with_structured_output(LLMQuoteContent)
)

# ---------------------------------------------------------------------------
# State — single source of truth flowing through all nodes
# ---------------------------------------------------------------------------

# Fields each product requires from ClientProfile before the specialist can begin.
# Classifier detects any that are empty and passes them as gaps to the specialist.
PRODUCT_REQUIRED_FIELDS: dict[str, list[str]] = {
    "general_liability":      ["state_or_province", "city"],
    "professional_liability": ["state_or_province", "annual_revenue"],
    "commercial_auto":        ["state_or_province", "employees"],
    "workers_comp":           ["state_or_province", "employees", "annual_revenue"],
}


class State(BaseModel):
    session_id: str = ""
    client_data: ClientProfile = ClientProfile()
    product: str = ""
    gaps: list[str] = []          # missing ClientProfile fields the specialist must fill first
    underwriting_data: dict = {}
    quote: dict = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _converse(system_prompt: str, label: str, model_cls: Type[T], session_id: str, node: str) -> tuple[T, list]:
    """
    Conversational loop shared by receptionist and specialist nodes.
    Runs until Claude calls the bound tool (model_cls).
    Records every turn to Supabase for audit.
    Returns (typed instance, full message list) so callers can extract extra values if needed.
    """
    bound_llm = llm.bind_tools([model_cls])
    print(f"\n{'-'*54}\n  {label}\n{'-'*54}\n")
    messages = [SystemMessage(content=system_prompt), HumanMessage(content="Please begin.")]
    turn = 0

    # Record the system prompt once
    db.insert_turn(session_id, node, turn, "system", system_prompt)
    turn += 1

    for _ in range(MAX_TURNS):
        response = bound_llm.invoke(messages)

        # Always append the AIMessage first — ToolMessage must follow its AIMessage
        messages.append(response)

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            try:
                instance = model_cls(**tool_call["args"])
                db.insert_turn(
                    session_id, node, turn, "tool",
                    content=json.dumps(tool_call["args"]),
                    tool_name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
                print(f"[{label}]: Thank you, I have all the information I need.\n")
                return instance, messages
            except ValidationError as exc:
                error_msg = (
                    "Validation failed — do NOT submit again yet.\n"
                    "Re-ask the user only for the fields listed below, then retry:\n\n"
                    f"{exc}"
                )
                messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
                db.insert_turn(session_id, node, turn, "tool", content=error_msg,
                               tool_name=tool_call["name"], tool_call_id=tool_call["id"])
                turn += 1
                continue

        # Conversational reply — record assistant turn, then prompt user
        assistant_text = response.content if isinstance(response.content, str) else json.dumps(response.content)
        db.insert_turn(session_id, node, turn, "assistant", assistant_text)
        turn += 1

        print(f"[{label}]: {response.content}\n")
        if _user_input_fn is not None:
            user_input = _user_input_fn(str(response.content)) or "Please continue."
        else:
            user_input = input("You: ").strip() or "Please continue."
        db.insert_turn(session_id, node, turn, "user", user_input)
        turn += 1
        messages.append(HumanMessage(content=user_input))

    raise RuntimeError(f"{label}: max turns ({MAX_TURNS}) reached without completing intake — aborting pipeline.")


def _extract_gap_values(messages: list, gaps: list[str]) -> dict:
    """
    One-shot extraction: given the completed specialist conversation and a list
    of field names that were missing from ClientProfile, return a dict of
    {field: value} to merge back into client_data.
    """
    GapModel = create_model("GapModel", **{f: (str, "") for f in gaps})
    chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "Extract the following fields from the conversation transcript below. "
             "Return an empty string for any field that was not mentioned.\n"
             f"Fields to extract: {', '.join(gaps)}"),
            ("human", "{transcript}"),
        ])
        | llm.with_structured_output(GapModel)
    )
    transcript = "\n".join(
        f"{m.type.upper()}: {m.content}"
        for m in messages
        if hasattr(m, "content") and isinstance(m.content, str) and m.content
    )
    result = chain.invoke({"transcript": transcript})
    return {k: v for k, v in result.model_dump().items() if v}


# ---------------------------------------------------------------------------
# Nodes — each reads from state, returns only what it changes
# ---------------------------------------------------------------------------

def receptionist_node(state: State) -> dict:
    profile, _ = _converse(RECEPTIONIST_PROMPT, "Receptionist", ClientProfile,
                           state.session_id, "receptionist")
    print(f"[System]: Profile collected → {profile.model_dump()}\n")
    db.insert_client_profile(state.session_id, profile.model_dump())
    return {"client_data": profile}


def classifier_node(state: State) -> dict:
    result = CLASSIFIER_CHAIN.invoke({"profile": state.client_data.model_dump_json()})
    product = result.product
    if product not in SPECIALIST_PROMPTS:
        print(f"[System]: WARNING — unknown product '{product}', defaulting to general_liability.")
        product = "general_liability"
    print(f"[System]: Routing to {product.replace('_', ' ').title()} specialist.")
    print(f"[Reason]: {result.reason}\n")

    profile = state.client_data.model_dump()
    gaps    = [f for f in PRODUCT_REQUIRED_FIELDS.get(product, []) if not profile.get(f)]
    if gaps:
        print(f"[System]: Gap detected — specialist will collect missing fields: {gaps}\n")

    db.insert_classification(state.session_id, product, result.reason)
    return {"product": product, "gaps": gaps}


def specialist_node(state: State) -> dict:
    prompt = SPECIALIST_PROMPTS.get(state.product, SPECIALIST_PROMPTS["general_liability"])
    if state.gaps:
        gap_list = ", ".join(state.gaps)
        prompt = (
            f"Before your normal questions, you must collect these missing fields "
            f"from the client profile: {gap_list}. "
            f"Ask for them naturally as part of your opening, then continue as normal.\n\n"
        ) + prompt
    label = f"{state.product.replace('_', ' ').title()} Specialist"
    model_cls = UNDERWRITING_MODELS.get(state.product, GeneralLiabilityData)
    data, messages = _converse(prompt, label, model_cls, state.session_id, "specialist")
    print(f"[System]: Underwriting data collected → {data.model_dump_json()}\n")
    db.insert_underwriting_data(state.session_id, state.product, data.model_dump())

    # Back-fill any gap values into client_data so the underwriter has complete info
    updates = {}
    if state.gaps:
        gap_values = _extract_gap_values(messages, state.gaps)
        if gap_values:
            print(f"[System]: Back-filling client profile gaps: {gap_values}\n")
            updates["client_data"] = ClientProfile(
                **{**state.client_data.model_dump(), **gap_values}
            )

    return {"underwriting_data": data.model_dump(), **updates}


def underwriter_node(state: State) -> dict:
    print("[System]: Generating quote...\n")

    # Step 1 — LLM generates only non-financial content
    llm_content: LLMQuoteContent = UNDERWRITER_CHAIN.invoke({"details": json.dumps({
        "client":               state.client_data.model_dump(),
        "product":              state.product,
        "underwriting_details": state.underwriting_data,
    })})

    # Step 2 — Audit for injection attacks and policy violations
    # audit_quote() raises RuntimeError if action=="block"
    audit_result = audit_quote(llm_content.model_dump(), state.product)
    if audit_result.flags:
        print(f"[Auditor]: {audit_result.action.upper()} — {audit_result.flags}")

    # Step 3 — Deterministic premium calculation (LLM never touches these)
    premium = PremiumCalculator.calculate(
        state.product,
        state.client_data.model_dump(),
        state.underwriting_data,
    )

    # Step 4 — Assemble and validate the full quote
    quote = InsuranceQuote(
        quote_id        = llm_content.quote_id,
        product         = state.product,
        coverage_limit  = audit_result.normalized_coverage_limit or llm_content.coverage_limit,
        annual_premium  = premium.annual_premium,
        monthly_premium = premium.monthly_premium,
        deductible      = premium.deductible,
        exclusions      = audit_result.sanitized_exclusions,
        notes           = audit_result.sanitized_notes,
    )

    quote_dict = quote.model_dump()
    quote_dict["client"] = state.client_data.model_dump()

    print("=" * 54)
    print("  INSURANCE QUOTE")
    print("=" * 54)
    print(json.dumps(quote_dict, indent=2))
    print("=" * 54)

    qid = quote.quote_id.replace("/", "-")
    out_path = Path(__file__).parent / f"quote_{qid}.json"
    with open(out_path, "w") as f:
        json.dump(quote_dict, f, indent=2)
    print(f"\nSaved to: {out_path}")

    db.insert_quote(state.session_id, quote_dict)
    db.close_session(state.session_id, "quoted")
    return {"quote": quote_dict}

# ---------------------------------------------------------------------------
# Entry point — graph built here so checkpointer opens at run time, not import
# ---------------------------------------------------------------------------

def build_graph(checkpointer):
    """Compile the LangGraph pipeline. Accepts any checkpointer (Postgres or Memory)."""
    builder = StateGraph(State)
    builder.add_node("receptionist", receptionist_node)
    builder.add_node("classifier",   classifier_node)
    builder.add_node("specialist",   specialist_node)
    builder.add_node("underwriter",  underwriter_node)
    builder.set_entry_point("receptionist")
    builder.add_edge("receptionist", "classifier")
    builder.add_edge("classifier",   "specialist")
    builder.add_edge("specialist",   "underwriter")
    builder.add_edge("underwriter",  END)
    return builder.compile(checkpointer=checkpointer)


def main():
    pg_url = os.environ.get("SUPABASE_DB_URL", "")
    if not pg_url:
        raise RuntimeError(
            "SUPABASE_DB_URL not set.\n"
            "Set it as a DSN keyword string to avoid URI parsing issues with dotted usernames:\n"
            "  host=aws-0-<region>.pooler.supabase.com port=5432 dbname=postgres "
            "user=postgres.<ref> password=<password> sslmode=require"
        )

    session_id = str(uuid.uuid4())
    db.create_session(session_id)

    with PostgresSaver.from_conn_string(pg_url) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)

        print("\n" + "=" * 54)
        print("  Virtual Insurance Agency — AI Quote System (LangGraph)")
        print("=" * 54)
        print(f"Session: {session_id}")
        print("Answer the agent's questions to receive a quote.\n")

        graph.invoke(
            State(session_id=session_id),
            config={"configurable": {"thread_id": session_id}},
        )


if __name__ == "__main__":
    main()
