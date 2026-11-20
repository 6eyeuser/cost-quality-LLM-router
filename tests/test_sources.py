"""Loader tests against fake `datasets` modules that follow the verified Hugging Face schemas."""
import json
import sys
import types

import pytest


class FakeDS:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        return [r[k] for r in self.rows] if isinstance(k, str) else self.rows[k]


def _fake_datasets():
    gsm = [{"question": f"Tom has {i} apples and buys 3 more. How many now?", "answer": f"reasoning...\n#### {1000 + i:,}"} for i in range(40)]
    mmlu = [{"question": f"Q{i}?", "subject": "high_school_physics", "choices": ["w", "x", "y", "z"], "answer": i % 4} for i in range(40)]
    arc = [{"id": f"a{i}", "question": f"Arc Q{i}?", "choices": {"text": ["p", "q", "r", "s"], "label": ["A", "B", "C", "D"] if i % 2 else ["1", "2", "3", "4"]},
            "answerKey": ("B" if i % 2 else "2")} for i in range(40)]
    mbpp = [{"task_id": i, "text": f"Write a function f{i}.", "code": "def f(): pass",
             "test_list": [f"assert f{i}(1) == 1", "assert 1", "assert 2"], "test_setup_code": "", "challenge_test_list": []} for i in range(40)]
    alpaca = [{"instruction": f"Give tips #{i}", "input": "ctx" if i % 2 else "", "output": "o", "text": "t"} for i in range(40)]
    cats = ["closed_qa", "open_qa", "summarization", "brainstorming", "information_extraction", "creative_writing"]
    dolly = [{"instruction": f"Summarize #{i}", "context": f"Passage {i}", "response": "A summary.", "category": cats[i % 6]} for i in range(60)]
    data = {"openai/gsm8k": gsm, "cais/mmlu": mmlu, "allenai/ai2_arc": arc, "google-research-datasets/mbpp": mbpp,
            "tatsu-lab/alpaca": alpaca, "databricks/databricks-dolly-15k": dolly}
    mod = types.ModuleType("datasets")
    mod.calls = []

    def load_dataset(name, *args, split=None, **kw):
        mod.calls.append((name, args, split))
        return FakeDS(data[name])

    mod.load_dataset = load_dataset
    return mod


@pytest.fixture
def sources(monkeypatch):
    fake = _fake_datasets()
    monkeypatch.setitem(sys.modules, "datasets", fake)
    from router import sources as s
    s.fake = fake
    return s


def _check_common(items, domain, grader, n):
    assert len(items) == n and len({i["query_id"] for i in items}) == n
    for it in items:
        assert it["domain"] == domain and it["grader"] == grader and it["query_text"].strip()
        json.loads(it["meta"])


def test_gsm8k(sources):
    items = sources.load_gsm8k(10)
    _check_common(items, "math", "numeric", 10)
    assert items[0]["reference"].isdigit() or "," not in items[0]["reference"]       # commas stripped from "1,000"
    assert all("Answer:" not in i["query_text"] for i in items)                      # format instruction must NOT leak to the router
    assert "Answer:" in json.loads(items[0]["meta"])["system"]
    assert ("openai/gsm8k", ("main",), "test") in sources.fake.calls


def test_mmlu_and_arc_letters_and_leakage(sources):
    mm = sources.load_mmlu(12)
    _check_common(mm, "knowledge", "mcq", 12)
    assert {i["reference"] for i in mm} <= set("ABCD") and "A. w" in mm[0]["query_text"] and "high school physics" in mm[0]["query_text"]
    arc = sources.load_arc(20)
    assert len(arc) == 20 and {i["reference"] for i in arc} <= set("ABCD")
    for it in arc:                                                    # numeric labels ("1".."4") map to the right letter
        assert it["reference"] == "B"
    assert all("Answer:" not in i["query_text"] for i in mm + arc)


def test_mbpp_reference_is_json_tests(sources):
    items = sources.load_mbpp(5)
    _check_common(items, "code", "code", 5)
    tests = json.loads(items[0]["reference"])
    assert tests[0].startswith("assert") and len(tests) == 3
    assert tests[0] in items[0]["query_text"] and "```python" in json.loads(items[0]["meta"])["system"]
    assert json.loads(items[0]["meta"])["test_setup_code"] == ""


def test_alpaca_and_dolly(sources):
    al = sources.load_alpaca(10)
    _check_common(al, "chat", "judge", 10)
    assert any("ctx" in a["query_text"] for a in al) and all(a["reference"] is None for a in al)
    do = sources.load_dolly(8)
    _check_common(do, "docqa", "judge", 8)
    kept = {"closed_qa", "summarization", "information_extraction"}
    ids = {int(d["query_id"].split("_")[1]) for d in do}
    assert all(("closed_qa", "open_qa", "summarization", "brainstorming", "information_extraction", "creative_writing")[i % 6] in kept for i in ids)
    assert do[0]["query_text"].startswith("Passage")


def test_sampling_is_deterministic_and_respects_n(sources):
    assert [i["query_id"] for i in sources.load_gsm8k(7)] == [i["query_id"] for i in sources.load_gsm8k(7)]
    assert len(sources.load_gsm8k(10_000)) == 40                       # n larger than the dataset: take everything
    assert set(sources.LOADERS) == set(sources.DEFAULT_COUNTS)
    assert 2800 <= sum(sources.DEFAULT_COUNTS.values()) <= 3200
