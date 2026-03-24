# Insurance PoC — Virtual Insurance Agency (LangGraph)

A conversational AI quote system that walks a small-business owner through an insurance intake and produces a bindable quote. Powered by **Claude Sonnet 4.6** via LangChain/LangGraph.

## Architecture

`main.py` implements a linear 4-node LangGraph pipeline with full Pydantic-typed I/O:

```
[receptionist] → [classifier] → [specialist] → [underwriter] → quote JSON
```

| Node | Mode | Role |
|---|---|---|
| `receptionist` | Conversational loop | Collects 9 business profile fields via chat |
| `classifier` | One-shot chain | Picks the best product from 4 options |
| `specialist` | Conversational loop | Gathers product-specific underwriting details |
| `underwriter` | One-shot chain | Generates a structured `InsuranceQuote` |

**Supported products:** `general_liability`, `professional_liability`, `commercial_auto`, `workers_comp`

**Key design choices:**
- All LLM I/O goes through Pydantic models — no raw JSON parsing in prompts
- Completion is signalled by a `tool_use` call, not `{"done": true}` sentinel
- LangGraph graph state checkpointed to Supabase (PostgreSQL) via `PostgresSaver`
- Structured outputs written to Supabase tables for audit (`sessions`, `client_profiles`, `classifications`, `underwriting_data`, `quotes`, `conversation_turns`)
- Each run saves a `quote_<id>.json` file next to `main.py`
- LangSmith tracing supported via env vars (zero-code)

## Setup

### 1. Create the virtual environment

```bash
python -m venv .venv
```

### 2. Activate it

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (cmd):**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

**Required:**
```powershell
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:SUPABASE_URL      = "https://<ref>.supabase.co"
$env:SUPABASE_KEY      = "<service-role-key>"
$env:SUPABASE_DB_URL   = "host=aws-0-<region>.pooler.supabase.com port=5432 dbname=postgres user=postgres.<ref> password=<password> sslmode=require"
```

**Optional — LangSmith tracing:**
```powershell
$env:LANGCHAIN_TRACING_V2 = "true"
$env:LANGCHAIN_API_KEY    = "ls__..."
$env:LANGCHAIN_PROJECT    = "InsurancePoC"
```

### 5. Apply the database schema

Run `schema.sql` once in the Supabase SQL editor. If upgrading from a previous version, run the numbered `migrate_NNN_*.sql` files instead.

### 6. Run

```bash
python main.py
```

The agent will ask questions interactively and print the final quote to the console and to a `quote_<id>.json` file.
