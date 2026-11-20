"""build_benchmark against a fake OpenAI client: grading, costs, resumability, failure handling."""
import json
import sys
import threading
import types

import pytest

from router import db as dbm
from router.config import CHEAP, JUDGE_MODEL, STRONG
import scripts.build_benchmark as bb


class FakeClient:
    """Mimics openai.OpenAI: client.chat.completions.create(...). Records every call."""

    def __init__(self):
        self.calls, self.lock, self.fail_ids = [], threading.Lock(), set()
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        with self.lock:
            self.calls.append(kw)
        msgs, model = kw["messages"], kw["model"]
        user = msgs[-1]["content"]
        if any(fid in user for fid in self.fail_ids) and model != JUDGE_MODEL:
            raise RuntimeError("simulated API outage")
        if model == JUDGE_MODEL:
            text = '{"score": 9, "reason": "good"}' if "GOOD ANSWER" in user else '{"score": 2, "reason": "bad"}'
            return self._resp(text, "stop", 80, 12)
        strong = model == STRONG.name
        if "apples" in user:                                    # numeric task: strong right, cheap wrong
            text = "Let me think.\nAnswer: 8" if strong else "Answer: 7"
        elif "function" in user:                                # code task: both right
            text = "```python\ndef add(a, b):\n    return a + b\n```"
        else:                                                   # chat task (judge graded): strong good, cheap bad
            text = "GOOD ANSWER about tips" if strong else "meh"
        return self._resp(text, "length" if (not strong and "truncate" in user) else "stop", 100, 50 if not strong else 400,
                          reasoning=300 if strong else 0)

    @staticmethod
    def _resp(text, finish, pt, ct, reasoning=0):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text), finish_reason=finish)],
            usage=types.SimpleNamespace(prompt_tokens=pt, completion_tokens=ct,
                                        completion_tokens_details=types.SimpleNamespace(reasoning_tokens=reasoning)))


def _items():
    def item(qid, dom, text, ref, grader, **meta):
        return {"query_id": qid, "domain": dom, "source_dataset": "fake", "query_text": text, "reference": ref,
                "grader": grader, "meta": json.dumps(meta)}
    return {
        "gsm8k": lambda n: [item("m1", "math", "Tom has 5 apples and gets 3 more. Total apples?", "8", "numeric", system="End with 'Answer: <number>'"),
                            item("m2", "math", "Ann has 2 apples and gets 6 more. Total apples? truncate", "8", "numeric", system="sys")],
        "mbpp": lambda n: [item("c1", "code", "Write a function add(a, b).\nYour code should pass this test:\nassert add(1,2)==3",
                                json.dumps(["assert add(1, 2) == 3"]), "code", system="code only", test_setup_code="")],
        "alpaca": lambda n: [item("a1", "chat", "Give me tips on sleeping", None, "judge")],
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(bb, "make_client", lambda: client)
    monkeypatch.setattr(bb.time, "sleep", lambda s: None)
    monkeypatch.setattr(bb, "LOADERS", _items())
    monkeypatch.setattr(bb, "DEFAULT_COUNTS", {"gsm8k": 2, "mbpp": 1, "alpaca": 1})
    db = tmp_path / "bench.db"

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["build", "--db", str(db), "--domains", "gsm8k", "mbpp", "alpaca", "--workers", "3", *extra])
        bb.main()
        return dbm.connect(db)
    return client, run


def test_builds_grades_and_prices_every_row(env):
    client, run = env
    conn = run()
    rows = {(q, m): (c, cost, rt, fin, rs) for q, m, c, cost, rt, fin, rs in conn.execute(
        "SELECT query_id, model_name, is_correct, cost_usd, reasoning_tokens, finish_reason, response_text FROM evaluation_cache")}
    assert len(rows) == 8                                                     # 4 queries x 2 models
    assert rows[("m1", STRONG.name)][0] == 1 and rows[("m1", CHEAP.name)][0] == 0     # numeric grading
    assert rows[("c1", STRONG.name)][0] == 1 and rows[("c1", CHEAP.name)][0] == 1     # code executed against tests
    assert rows[("a1", STRONG.name)][0] == 1 and rows[("a1", CHEAP.name)][0] == 0     # judge grading (score>=7)
    assert rows[("m1", STRONG.name)][1] == pytest.approx(STRONG.cost(100, 400))        # cost from the price config
    assert rows[("m1", STRONG.name)][2] == 300 and rows[("m1", CHEAP.name)][2] == 0    # reasoning tokens stored
    assert rows[("m2", CHEAP.name)][3] == "length"                                     # finish_reason stored for the audit
    assert dict(conn.execute("SELECT split, COUNT(*) FROM queries GROUP BY split").fetchall())


def test_api_parameters_are_the_documented_groq_ones(env):
    client, run = env
    run()
    gen = [c for c in client.calls if c["model"] != JUDGE_MODEL]
    assert all("max_completion_tokens" in c and "max_tokens" not in c and c["temperature"] == 0.0 for c in gen)
    strong_calls = [c for c in gen if c["model"] == STRONG.name]
    assert all(c["extra_body"] == {"reasoning_effort": "medium"} for c in strong_calls)
    assert all(not c.get("extra_body") for c in gen if c["model"] == CHEAP.name)       # only gpt-oss gets reasoning_effort
    # system prompt is sent separately from the router-visible query text
    m1 = next(c for c in gen if "5 apples" in c["messages"][-1]["content"])
    assert m1["messages"][0] == {"role": "system", "content": "End with 'Answer: <number>'"}
    assert "Answer:" not in m1["messages"][-1]["content"]
    judge = [c for c in client.calls if c["model"] == JUDGE_MODEL]
    assert judge and all(c["messages"][0]["role"] == "system" for c in judge)


def test_resumable_second_run_makes_no_calls(env):
    client, run = env
    run()
    n = len(client.calls)
    conn = run()
    assert len(client.calls) == n and conn.execute("SELECT COUNT(*) FROM evaluation_cache").fetchone()[0] == 8


def test_failed_calls_are_not_stored_and_are_retried_next_run(env):
    client, run = env
    client.fail_ids = {"Ann has 2 apples"}
    conn = run()
    stored = {r[0] for r in conn.execute("SELECT DISTINCT query_id FROM evaluation_cache")}
    assert "m2" not in stored and {"m1", "c1", "a1"} <= stored                  # one bad query doesn't sink the run
    client.fail_ids = set()
    conn = run()                                                                 # API recovered: only the missing rows are fetched
    assert conn.execute("SELECT COUNT(*) FROM evaluation_cache WHERE query_id='m2'").fetchone()[0] == 2


def test_skip_judge_leaves_judge_domains_empty(env):
    client, run = env
    conn = run("--skip-judge")
    assert conn.execute("SELECT COUNT(*) FROM evaluation_cache WHERE query_id='a1'").fetchone()[0] == 0
    assert not [c for c in client.calls if c["model"] == JUDGE_MODEL]


def test_unparseable_judge_output_is_not_recorded_as_a_label(env, monkeypatch):
    client, run = env
    orig = client.create
    client.create = lambda **kw: FakeClient._resp("I refuse to answer in JSON", "stop", 10, 5) if kw["model"] == JUDGE_MODEL else orig(**kw)
    client.chat.completions.create = client.create
    conn = run()
    assert conn.execute("SELECT COUNT(*) FROM evaluation_cache WHERE query_id='a1'").fetchone()[0] == 0
