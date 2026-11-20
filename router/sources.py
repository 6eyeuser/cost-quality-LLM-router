"""Dataset loaders for the mixed benchmark.

Design rule: `query_text` is what a real user would type. Output-format instructions
(e.g. "end with 'Answer: X'") go in a separate *system prompt* stored in `meta`, so the
router cannot cheat by recognising a dataset's template.

Domains: math (GSM8K), knowledge (MMLU), science_qa (ARC-Challenge), code (MBPP),
chat (Alpaca, judge-graded), docqa (Dolly summarization/extraction, judge-graded).
"""
from __future__ import annotations

import json
import random
from typing import Callable

from .config import SEED

LETTERS = "ABCDEFGHIJ"
MCQ_SYSTEM = "Answer the multiple-choice question. Think briefly, then finish with a line of the form 'Answer: X' where X is the option letter."
MATH_SYSTEM = "Solve the problem. Finish with a line of the form 'Answer: <number>'."
CODE_SYSTEM = "Write a Python solution. Reply with a single ```python code block and nothing else."


def _item(qid, domain, src, text, reference, grader, system=None, **meta) -> dict:
    m = dict(meta)
    if system:
        m["system"] = system
    return {"query_id": qid, "domain": domain, "source_dataset": src, "query_text": text,
            "reference": reference, "grader": grader, "meta": json.dumps(m)}


def _sample_idx(n_total: int, n: int, salt: str) -> list[int]:
    return random.Random(f"{SEED}-{salt}").sample(range(n_total), min(n, n_total))


def load_gsm8k(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [_item(f"gsm8k_{i}", "math", "GSM8K", ds[i]["question"], ds[i]["answer"].split("####")[-1].strip().replace(",", ""),
                  "numeric", MATH_SYSTEM) for i in _sample_idx(len(ds), n, "gsm8k")]


def _mcq_text(question: str, choices: list[str], subject: str | None = None) -> str:
    head = f"The following is a multiple choice question about {subject.replace('_', ' ')}.\n\n" if subject else ""
    return head + question.strip() + "\n" + "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def load_mmlu(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    out = []
    for i in _sample_idx(len(ds), n, "mmlu"):
        r = ds[i]
        out.append(_item(f"mmlu_{i}", "knowledge", "MMLU", _mcq_text(r["question"], r["choices"], r["subject"]),
                         LETTERS[r["answer"]], "mcq", MCQ_SYSTEM))
    return out


def load_arc(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    out = []
    for i in _sample_idx(len(ds), n, "arc"):
        r = ds[i]
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labels or len(texts) > 10:
            continue
        out.append(_item(f"arc_{i}", "science_qa", "ARC-Challenge", _mcq_text(r["question"], texts),
                         LETTERS[labels.index(r["answerKey"])], "mcq", MCQ_SYSTEM))
    return out


def load_mbpp(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    out = []
    for i in _sample_idx(len(ds), n, "mbpp"):
        r = ds[i]
        tests = list(r["test_list"])
        text = f"{r['text']}\nYour code should pass this test:\n{tests[0]}"
        out.append(_item(f"mbpp_{i}", "code", "MBPP", text, json.dumps(tests), "code", CODE_SYSTEM,
                         test_setup_code=r.get("test_setup_code") or ""))
    return out


def load_alpaca(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out = []
    for i in _sample_idx(len(ds), n, "alpaca"):
        r = ds[i]
        text = r["instruction"] + (f"\n\n{r['input']}" if r.get("input") else "")
        out.append(_item(f"alpaca_{i}", "chat", "Alpaca", text, None, "judge"))
    return out


def load_dolly(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    keep = [i for i, c in enumerate(ds["category"]) if c in {"summarization", "information_extraction", "closed_qa"}]
    out = []
    for j in _sample_idx(len(keep), n, "dolly"):
        i = keep[j]
        r = ds[i]
        out.append(_item(f"dolly_{i}", "docqa", "Dolly", f"{r['context']}\n\n{r['instruction']}", r["response"], "judge"))
    return out


LOADERS: dict[str, Callable[[int], list[dict]]] = {
    "gsm8k": load_gsm8k, "mmlu": load_mmlu, "arc": load_arc, "mbpp": load_mbpp,
    "alpaca": load_alpaca, "dolly": load_dolly,
}
DEFAULT_COUNTS = {"gsm8k": 600, "mmlu": 600, "arc": 400, "mbpp": 400, "alpaca": 500, "dolly": 500}  # ~3,000
