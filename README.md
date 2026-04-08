# InboxPilot-OpenEnv

**A real-world email triage and response drafting environment for the Meta PyTorch OpenEnv Hackathon.**

An AI agent manages a shared university operations inbox: it reads emails, classifies them, assigns priority, extracts key fields, routes messages to the right internal queue, and optionally drafts compliant replies.

---

## Tasks

| ID | Difficulty | Description |
|----|-----------|-------------|
| `easy`   | ⭐ | Single email — classify, prioritise, extract, route |
| `medium` | ⭐⭐ | Classification + routing + constrained reply draft |
| `hard`   | ⭐⭐⭐ | 6-email inbox with phishing, spam, duplicates, step budget |

---

## Quick start (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the environment server
uvicorn app.server:app --host 0.0.0.0 --port 7860 --reload

# 3. Run the baseline agent (heuristic, no API key needed)
API_BASE_URL=http://localhost:7860 python inference.py --task easy

# With an LLM (set your key):
OPENAI_API_KEY=sk-... MODEL_NAME=gpt-4o-mini API_BASE_URL=http://localhost:7860 python inference.py --task hard
```

---

## API reference

### `POST /reset`
```json
{ "task_id": "easy" }
```
Returns an `Observation` object.

### `POST /step`
```json
{ "action": { "action": "SelectEmail", "email_id": "email_001" } }
```
Returns `{ "observation": ..., "reward": 0.15, "done": false, "info": { "reward_breakdown": {...} } }`

### `GET /state`
Returns full episode state including per-email triage status and cumulative reward breakdown.

### `GET /tasks`
Lists available task IDs.

---

## Action space

| Action | Parameters |
|--------|-----------|
| `SelectEmail` | `email_id` |
| `ClassifyEmail` | `category`: support / finance / placement / spam / phishing / urgent / general |
| `SetPriority` | `level`: low / medium / high |
| `ExtractFields` | `fields`: dict of extracted key-value pairs |
| `RouteEmail` | `target_queue`: tech_support / finance_office / placement_cell / security / archive |
| `DraftReply` | `reply_text` |
| `MarkResolved` | — |
| `RequestMoreInfo` | `question` |
| `FinishEpisode` | — |

---

## Reward shaping

| Event | Delta |
|-------|-------|
| Correct classification | +0.15 |
| Correct priority | +0.10 |
| Correct routing | +0.20 |
| Correct field extracted | +0.05 each |
| Reply semantic slot hit | +0.04 each |
| Email marked resolved | +0.10 |
| Invalid action | -0.05 |
| Repeat action | -0.08 |
| Phishing mislabelled | -0.12 |
| Steps over soft budget | -0.02/step |

Final score is normalised to **[0, 1]**.

---

## Running tests

```bash
python tests/test_env.py
python tests/test_graders.py
```

---

## Docker build & run

```bash
docker build -t inboxpilot-openenv .
docker run -p 7860:7860 inboxpilot-openenv
```

---

## Deploy to Hugging Face Spaces

1. Create a new Space → **Docker** SDK.
2. Push this repo:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/inboxpilot-openenv
   git push space main
   ```
3. The Space will build and expose the API at `https://<your-username>-inboxpilot-openenv.hf.space`.

---

## Grading methodology

All graders are **deterministic and rule-based** — no LLM-as-judge.

- **Classification / Priority / Routing**: exact match against gold labels.
- **Field extraction**: case-insensitive exact match per field.
- **Reply quality**: phrase-group coverage + forbidden-phrase penalties (no regex variance).
- **Efficiency**: small penalty per step over the soft budget.
