"""Multi-step LLM pipelines where each model call feeds the next.

Usage:
    python pipelines.py check        # verify the API key is configured
    python pipelines.py fact         # single-call smoke test
    python pipelines.py judge        # generate -> answer -> evaluate a hard question
    python pipelines.py opportunity  # industry -> pain point -> agentic AI solution
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

# Cheap model for simple tasks, mid-tier for reasoning, strongest for judging.
FAST_MODEL = os.getenv("FAST_MODEL", "gpt-5.4-nano")
REASONING_MODEL = os.getenv("REASONING_MODEL", "gpt-5.4-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.4")


def ask(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def section(title: str, body: str) -> None:
    print(f"\n=== {title} ===\n{body}")


def check_key() -> bool:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        print(f"OPENAI_API_KEY is set (starts with {key[:8]})")
        return True
    print("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return False


def run_fact(client: OpenAI) -> None:
    section("Fun fact", ask(client, FAST_MODEL, "Tell me a fun fact"))


def run_judge(client: OpenAI) -> None:
    question = ask(
        client,
        REASONING_MODEL,
        "Please propose a hard, challenging question to assess someone's IQ. "
        "Respond only with the question.",
    )
    section(f"Question ({REASONING_MODEL})", question)

    answer = ask(client, REASONING_MODEL, question)
    section(f"Answer ({REASONING_MODEL})", answer)

    verdict = ask(
        client,
        JUDGE_MODEL,
        f"Here is a question:\n{question}\n\n"
        f"And here is a possible answer that might be correct or incorrect:\n{answer}\n\n"
        "Please evaluate if the answer is correct or incorrect.",
    )
    section(f"Verdict ({JUDGE_MODEL})", verdict)


def run_opportunity(client: OpenAI) -> None:
    industry = ask(
        client,
        REASONING_MODEL,
        "Pick one business area that might be worth exploring for an Agentic AI "
        "opportunity. Respond with only the name of the business area.",
    )
    section("Business area", industry)

    pain_point = ask(
        client,
        REASONING_MODEL,
        f"Describe one specific, challenging pain point in the {industry} industry "
        "that might be ripe for an Agentic AI solution. Be concrete, in 3-5 sentences.",
    )
    section("Pain point", pain_point)

    solution = ask(
        client,
        JUDGE_MODEL,
        f"Industry: {industry}\nPain point: {pain_point}\n\n"
        "Propose an Agentic AI solution: the agents involved, the tools each one uses, "
        "how they hand off work, where a human stays in the loop, and how success is measured.",
    )
    section("Proposed agentic solution", solution)


PIPELINES = {"fact": run_fact, "judge": run_judge, "opportunity": run_opportunity}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", choices=["check", *PIPELINES])
    args = parser.parse_args()

    load_dotenv(override=True)
    if not check_key():
        sys.exit(1)
    if args.pipeline != "check":
        PIPELINES[args.pipeline](OpenAI())


if __name__ == "__main__":
    main()
