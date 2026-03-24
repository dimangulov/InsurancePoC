# Database Design Decisions

## Context

Insurance applications have strict audit requirements: every quote must be reproducible from its raw inputs, and every message exchanged between the user and any AI agent must be recoverable. This document records the schema design decisions for the Supabase (PostgreSQL) persistence layer.

---

## Table Map

| Table | Storage Strategy | Why |
|---|---|---|
| `sessions` | Relational | Anchor for all FK relationships; status lifecycle |
| `client_profiles` | Flat columns | Fixed 9-field shape; queried by email, indexed for CRM |
| `classifications` | Flat columns | Tiny record; `product` and `reason` are always scalars |
| `underwriting_data` | **JSONB** | 4 non-overlapping product shapes — columns would be sparse |
| `quotes` | Flat columns | Premiums/deductibles must be queryable for actuary analytics |
| `conversation_turns` | Relational rows | One row per message — never a JSONB array |

---

## Decision 1 — `conversation_turns` as rows, not a JSONB array

**Rejected:** `sessions.history JSONB`

**Chosen:** `conversation_turns` with one row per message, indexed on `(session_id, node)`.

**Why:**
- Enables `WHERE node = 'receptionist' AND role = 'user'` — exact replay of what the client told each agent
- Enables full-text search across turns: `WHERE content ILIKE '%flood%'`
- Enables aggregations: average turns per node before data collection completes
- JSONB arrays can't be partially indexed or efficiently queried by element

**Trade-off:** More rows, but conversation history is write-once and read rarely — the insert cost is negligible.

---

## Decision 2 — `underwriting_data.data` as JSONB

**Rejected:** Four separate tables (`general_liability_data`, `professional_liability_data`, etc.)

**Chosen:** Single `underwriting_data` table with a `product` discriminator column and a `JSONB data` column.

**Why:**
- `GeneralLiabilityData`, `ProfessionalLiabilityData`, `CommercialAutoData`, `WorkersCompData` are non-overlapping shapes with no shared columns worth normalising
- Four tables would mean four JOIN paths for every query that starts from `sessions`
- Adding a new product in the future is a Pydantic model change, not a migration
- A GIN index on `data` covers any future need to query inside the blob

**Boundary rule:** Only `underwriting_data` gets JSONB. Everything with a known, stable shape gets flat columns.

---

## Decision 3 — `client_profiles` as flat columns

**Rejected:** `sessions.client_data JSONB`

**Chosen:** Flat columns for all 9 fields from `ClientProfile`.

**Why:**
- Shape is fixed and validated by Pydantic before write
- `email` needs a B-tree index for CRM lookups
- `state` and `annual_revenue` are filter targets for risk segmentation queries
- Flat columns make NULL-checks and constraints enforceable at the DB layer

---

## Decision 4 — `quotes` fully normalised

**Rejected:** Storing `InsuranceQuote` as a JSON blob on sessions

**Chosen:** Flat columns for all quote fields including `exclusions TEXT[]` (native Postgres array).

**Why:**
- `annual_premium`, `monthly_premium`, `deductible` are the primary targets for actuary reporting — they need to be `SUM`/`AVG`/`GROUP BY` queryable
- `exclusions` as `TEXT[]` allows `WHERE 'flood' = ANY(exclusions)` without JSON parsing
- Quote records are legally binding documents — explicit column constraints are safer than schema-less blobs

---

## Decision 5 — LangGraph SqliteSaver kept alongside Supabase

The app uses two persistence mechanisms in parallel:

| Layer | Purpose |
|---|---|
| LangGraph `SqliteSaver` (`sessions.db`) | Graph state resumption — if the process crashes mid-conversation, LangGraph can resume from the last checkpoint |
| Supabase | Structured audit log — queryable, long-term, cross-session analytics |

The SQLite checkpoint is an implementation detail of the runtime; the Supabase tables are the system of record for business and compliance purposes.

---

## Audit Query

Replay a full session, node by node, message by message:

```sql
SELECT
    node,
    turn_index,
    role,
    content,
    tool_name,
    recorded_at
FROM   conversation_turns
WHERE  session_id = '<uuid>'
ORDER  BY node, turn_index;
```

---

## Setup

1. Run `schema.sql` against your Supabase project (SQL editor or `psql`)
2. Set environment variables:
   ```bash
   export SUPABASE_URL="https://<project>.supabase.co"
   export SUPABASE_KEY="<anon-or-service-role-key>"
   ```
3. `pip install supabase`

If the env vars are absent the app runs normally — Supabase writes are skipped with a warning and the LangGraph SQLite checkpoint continues to function.
