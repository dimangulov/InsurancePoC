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
- Sessions are persisted to `sessions.db` (SQLite) via LangGraph's `SqliteSaver`
- Each run saves a `quote_<id>.json` file next to `main.py`

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

### 4. Set your API key

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 5. Run

```bash
python main.py
```

The agent will ask questions interactively and print the final quote to the console and to a `quote_<id>.json` file.
