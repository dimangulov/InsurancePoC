-- Insurance PoC — Supabase / PostgreSQL schema
-- Run once against your Supabase project via the SQL editor or psql.

-- Enable UUID generation (already on in Supabase by default)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Sessions — one row per quote run
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at  TIMESTAMPTZ NOT NULL    DEFAULT now(),
    status      TEXT        NOT NULL    DEFAULT 'in_progress',   -- in_progress | quoted | abandoned
    user_id     UUID                                              -- nullable until auth is wired
    -- CONSTRAINT sessions_status_check CHECK (status IN ('in_progress','quoted','abandoned'))
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Client profiles — receptionist output (9 known fields → proper columns)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS client_profiles (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    business_name       TEXT,
    owner_name          TEXT,
    email               TEXT,
    phone               TEXT,
    city                TEXT,
    state_or_province   TEXT,
    country             TEXT,
    postal_code         TEXT,
    industry_type       TEXT,
    annual_revenue      INTEGER,
    employees           INTEGER,
    years_in_business   INTEGER,
    collected_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_client_profiles_session ON client_profiles(session_id);
CREATE INDEX IF NOT EXISTS idx_client_profiles_email   ON client_profiles(email);

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Classifications — classifier output (tiny but audit-critical)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS classifications (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    product         TEXT        NOT NULL,   -- general_liability | professional_liability | ...
    reason          TEXT,                   -- one-sentence LLM justification
    classified_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_classifications_session ON classifications(session_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Underwriting data — specialist output
-- 4 product schemas are non-overlapping shapes → JSONB typed by `product` column
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS underwriting_data (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id   UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    product      TEXT        NOT NULL,
    data         JSONB       NOT NULL,   -- shape determined by product column
    collected_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_underwriting_session ON underwriting_data(session_id);
CREATE INDEX IF NOT EXISTS idx_underwriting_data_gin ON underwriting_data USING GIN (data);

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. Quotes — underwriter output (normalized: actuaries query premiums directly)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS quotes (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    quote_ref        TEXT        UNIQUE NOT NULL,   -- human-readable ID from LLM
    product          TEXT        NOT NULL,
    coverage_limit   TEXT,
    annual_premium   INTEGER,
    monthly_premium  INTEGER,
    deductible       INTEGER,
    exclusions       TEXT[],                        -- native Postgres array
    notes            TEXT,
    valid_until      DATE,
    generated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_quotes_session ON quotes(session_id);
CREATE INDEX IF NOT EXISTS idx_quotes_product ON quotes(product);

-- ─────────────────────────────────────────────────────────────────────────────
-- 6. Conversation turns — one row per message, per node
-- This is the audit table: never store history as a JSONB array on session.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversation_turns (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id   UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    node         TEXT        NOT NULL,   -- receptionist | classifier | specialist | underwriter
    turn_index   INTEGER     NOT NULL,
    role         TEXT        NOT NULL,   -- user | assistant | tool | system
    content      TEXT        NOT NULL,
    tool_name    TEXT,                   -- populated when role = 'tool'
    tool_call_id TEXT,                   -- links AIMessage → ToolMessage pairs
    metadata     JSONB,                  -- token counts, model, latency_ms
    recorded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (session_id, node, turn_index)
);
CREATE INDEX IF NOT EXISTS idx_turns_session      ON conversation_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_session_node ON conversation_turns(session_id, node);

-- ─────────────────────────────────────────────────────────────────────────────
-- Audit query: replay a full session in node order
-- ─────────────────────────────────────────────────────────────────────────────
-- SELECT node, turn_index, role, content, tool_name, recorded_at
-- FROM   conversation_turns
-- WHERE  session_id = '<uuid>'
-- ORDER  BY node, turn_index;
