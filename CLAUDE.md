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
