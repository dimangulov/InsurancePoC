# CLAUDE.md — Project Instructions

## Database schema sync rule

Whenever a Pydantic model in `main.py` has fields added, removed, or renamed,
you MUST update **both** of the following before finishing:

1. **`schema.sql`** — update the corresponding `CREATE TABLE` column definitions
2. **`migrate_NNN_<description>.sql`** — create a new numbered migration file with
   `ALTER TABLE` statements for anyone who already ran the previous schema

Migration file naming: `migrate_001_description.sql`, `migrate_002_description.sql`, etc.

### Models → tables mapping

| Pydantic model | Table |
|---|---|
| `ClientProfile` | `client_profiles` |
| `ClassificationResult` | `classifications` |
| `GeneralLiabilityData` / `ProfessionalLiabilityData` / `CommercialAutoData` / `WorkersCompData` | `underwriting_data.data` (JSONB) |
| `InsuranceQuote` | `quotes` |
| `State` (LangGraph) | LangGraph checkpoint tables (managed by `checkpointer.setup()`) |

Note: `underwriting_data.data` is JSONB — no migration needed when product-specific
models change, but update the relevant Pydantic model and its tests.

## Security layer sync rule

Whenever a new insurance product is added or an existing one changes its data model:

1. **`pricing.py`** — add/update the per-product calculator function and wire it into `PremiumCalculator._CALCULATORS`
2. **`auditor.py`** — update `ALLOWED_COVERAGE_LIMITS` with valid coverage tiers for the new product
3. **`evals.py`** — add at least one scenario for the new product in `SCENARIOS`

## Eval sync rule

Whenever a prompt in `main.py` changes (system prompts, specialist prompts, underwriter prompt):

1. Run `python evals.py --runs 3` to verify no regression
2. If a new failure mode is discovered, add a scenario or persona to `evals.py` before closing the PR
