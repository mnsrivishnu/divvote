"""LLM interface (OpenAI, local vLLM).

Purpose
-------
Single entry point for querying a voting LLM with a causal prompt and reducing
its reply to a binary 1/0 vote (PLAN.md section 4.2, STEP 5). Default backend
is OpenAI GPT-4o-mini; local open-weight models (Llama 3.1-8B, Mistral-7B,
Phi-3 via vLLM) are the cross-LLM alternatives (sections 5, 7.4). Temperature
is pinned to 0 for determinism.

Maps to: Week 2, Days 3-4 (PLAN.md section 10) — "Implement LLM interface
(OpenAI GPT-4o-mini wrapper)."

Implements the skeleton given in PLAN.md section 8.3.
"""

from openai import OpenAI
client = OpenAI()


def llm_vote(prompt: str, model="gpt-4o-mini") -> int:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user",
                   "content": prompt + " Answer only Yes or No."}],
        temperature=0, max_tokens=5,
    )
    return 1 if "yes" in resp.choices[0].message.content.lower() else 0
