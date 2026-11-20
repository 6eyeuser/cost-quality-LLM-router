import json

from router import graders as g


def test_numeric_float_compare_and_formats():
    assert g.grade_numeric("so the total is 18.00", "18").correct
    assert g.grade_numeric("Answer: $1,234", "1234").correct
    assert g.grade_numeric(r"final \boxed{42}", "42").correct
    assert g.grade_numeric("Answer: 17", "18").correct is False
    assert g.grade_numeric("no digits here", "5").correct is False
    assert g.grade_numeric("x\n#### 72", "72").correct
    assert g.grade_numeric("Answer: -3.5", "-3.5").correct


def test_numeric_uses_answer_line_not_last_number():
    assert g.grade_numeric("Answer: 10\nThat took 3 steps.", "10").correct


def test_mcq():
    assert g.grade_mcq("Reasoning...\nAnswer: B", "B").correct
    assert g.grade_mcq("The answer is (C).", "C").correct
    assert g.grade_mcq("Answer: A", "B").correct is False
    assert g.grade_mcq("I am not sure", "B").correct is False
    assert g.grade_mcq("a long text with a stray A in it\nAnswer: D", "D").correct


def test_code_pass_fail_timeout():
    tests = json.dumps(["assert add(2, 3) == 5"])
    good = "Here:\n```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert g.grade_code(good, tests, {}).correct
    assert not g.grade_code(bad, tests, {}).correct
    assert g.run_python_tests("def f():\n    while True: pass", ["f()"], timeout=1).detail == "timeout"
    assert not g.grade_code("", tests, {}).correct


def test_judge_parsing():
    assert g.parse_judge_score('{"score": 8, "reason": "ok"}') == 8
    assert g.parse_judge_score('Sure! {"score":3,"reason":"bad"} done') == 3
    assert g.parse_judge_score("score: 9") == 9
    assert g.parse_judge_score('{"score": 99}') is None
    assert g.parse_judge_score("garbage") is None
    ok = g.grade_judge(lambda m: '{"score": 7}', "q", "a")
    assert ok.correct and ok.score == 0.7
    assert g.grade_judge(lambda m: "nope", "q", "a") is None
    assert not g.grade_judge(lambda m: '{"score": 6}', "q", "a").correct
