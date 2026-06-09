# GC Agent Regression Tester

A regression testing tool for Genesys Cloud Agentic Virtual Agents. Uses an LLM-as-judge methodology — an Ollama-hosted LLM plays a simulated user with a persona and goal, drives multi-turn conversations with your deployed agent via the Web Messaging API, and evaluates whether the goal was achieved across multiple attempts.

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) running locally with a model pulled (e.g., `ollama pull llama3.2`)
- A Genesys Cloud Web Messaging deployment (deployment ID + region)

## Setup

```bash
cd gc-agent-regression-tester
pip install -r requirements.txt
```

## Running the Web UI

```bash
python3 -m src.web_app
```

Open http://localhost:8899 in your browser. Fill in:
- **Deployment ID** — your Genesys Cloud Web Messaging deployment ID
- **Region** — e.g., `mypurecloud.com`
- **Ollama Model** — e.g., `llama3.2`
- **Allowed Origin** — the origin header for WebSocket auth (try `https://apps.mypurecloud.com`)
- **Test Suite File** — upload a YAML or JSON test suite

## Running via CLI

```bash
python3 -m src.cli run test_suite.yaml \
  --region mypurecloud.com \
  --deployment-id YOUR_DEPLOYMENT_ID \
  --ollama-model llama3.2
```

Backward-compatible form (suite path only) still works:

```bash
python3 -m src.cli test_suite.yaml
```

### Conversation cleanup commands

After each attempt, the runner resolves the Genesys `conversationId` and disconnects it via Platform API. Active conversations are tracked in `.gc-tester/active_conversations.json`.

```bash
python3 -m src.cli conversations list
python3 -m src.cli disconnect --id CONVERSATION_ID
python3 -m src.cli disconnect --all
```

## VA test suites (from test-scripts)

Pre-built suites grouped by reporter type live in [`test-suites/`](test-suites/):

- `pch_test_suite.yaml` — Primary Cardholder (45 scenarios)
- `poa_test_suite.yaml` — Power of Attorney (24 scenarios)
- `atd_test_suite.yaml` / `atm_test_suite.yaml` — Authority to Disclose (19 scenarios)
- `guardrail_test_suite.yaml` — Guardrail / unsupported (10 scenarios)

Regenerate from markdown with `python3 scripts/convert_test_scripts_to_yaml.py`.

## Test Suite Format

```yaml
name: My Regression Suite

scenarios:
  - name: Account Balance Inquiry
    persona: >
      Customer named Margaret. Her 8-digit login code is 12345678.
    goal: >
      Check account balance. Provide login code when asked.
      Goal is achieved when the agent provides a balance amount.
    first_message: "What is my account balance?"
    attempts: 3
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Scenario name shown in results |
| `persona` | Yes | Who the simulated user is, including any auth details they'd know |
| `goal` | Yes | What the user is trying to accomplish and how to know it's done |
| `first_message` | No | Exact first message to send (if omitted, LLM generates it) |
| `attempts` | No | Number of times to run this scenario (default: 5) |

## Configuration

Copy `.env.example` to `.env`, fill in your Genesys deployment ID and region, then run the CLI or web app (values are loaded automatically via `python-dotenv`).

You can also set defaults via environment variables or a `config.yaml` file:

| Env Variable | Config Key | Description |
|-------------|------------|-------------|
| `GC_REGION` | `gc_region` | Genesys Cloud region |
| `GC_DEPLOYMENT_ID` | `gc_deployment_id` | Web Messaging deployment ID |
| `GC_ORIGIN` | `gc_origin` | WebSocket Origin header (e.g. `https://apps.mypurecloud.com`) |
| `GC_CLIENT_ID` | `gc_client_id` | OAuth client ID for Platform API (conversation cleanup) |
| `GC_CLIENT_SECRET` | `gc_client_secret` | OAuth client secret for Platform API |
| `GC_TESTER_CONVERSATIONS_FILE` | `gc_conversations_file` | Path to active conversation registry (default: `.gc-tester/active_conversations.json`) |
| `OLLAMA_BASE_URL` | `ollama_base_url` | Ollama URL (default: http://localhost:11434) |
| `OLLAMA_MODEL` | `ollama_model` | Ollama model name |
| `GC_TESTER_DEFAULT_ATTEMPTS` | `default_attempts` | Default attempts per scenario (default: 5) |
| `GC_TESTER_MAX_TURNS` | `max_turns` | Max conversation turns (default: 20) |
| `GC_TESTER_RESPONSE_TIMEOUT` | `response_timeout` | Timeout in seconds (default: 30) |
| `GC_TESTER_SUCCESS_THRESHOLD` | `success_threshold` | Regression threshold (default: 0.8) |

Precedence: Web UI > Environment variables > config.yaml > defaults

### Genesys OAuth app (Platform API)

Create an OAuth client in Genesys Cloud (Client Credentials grant) with roles that allow:

- `GET /api/v2/conversations/messages/{messageId}/details`
- `POST /api/v2/conversations/{conversationId}/disconnect`

Set `GC_CLIENT_ID` and `GC_CLIENT_SECRET` in `.env`. These are required for CLI test runs so conversations are cleaned up after each attempt.

## Results

The results page shows per-scenario success rates with all attempts expandable to review the full conversation. Export to CSV or JSON from the results page.

The CLI exits with code 1 if any scenario falls below the success threshold, making it CI/CD friendly.
