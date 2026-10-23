# LLM Chaining Pipelines

A command-line tool that chains several LLM calls together, where each step's output becomes the next step's input. It covers the basic pattern behind agentic systems: breaking a task into steps, sending each step to the model best suited for it, and having a stronger model check the work.

## Pipelines

| Command | What it does |
|---|---|
| `check` | Confirms the OpenAI API key is configured |
| `fact` | One-call smoke test using the fast model |
| `judge` | **Generate → Answer → Evaluate.** One model writes a hard reasoning question and answers it, then a stronger model judges whether the answer is correct (LLM-as-a-judge) |
| `opportunity` | **Industry → Pain point → Solution.** Picks a business area, identifies a concrete pain point in it, then designs a multi-agent AI solution with tools, handoffs and human-in-the-loop checkpoints |

## Model routing

Each step uses a model sized to the job, which keeps cost down without losing quality where it matters:

- `FAST_MODEL` (default `gpt-5.4-nano`): simple, low-stakes calls
- `REASONING_MODEL` (default `gpt-5.4-mini`): generating and solving problems
- `JUDGE_MODEL` (default `gpt-5.4`): evaluation and final solution design

You can override any of them in `.env`.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your OpenAI API key

python pipelines.py judge
python pipelines.py opportunity
```

## Example output (`opportunity`)

```
=== Business area ===
Customer Support

=== Pain point ===
Handling complex, multi-step issue resolution that spans several systems and teams,
for example a billing dispute that also needs a refund, account verification and a
shipment status check...

=== Proposed agentic solution ===
A multi-agent support orchestration system that acts like a digital case team:
an Intake & Triage agent, specialist agents for billing, refunds and logistics,
and human approval gates for high-risk actions...
```

## Tech

Python · OpenAI API · python-dotenv
