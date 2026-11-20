"""Runs gateway/dashboard.py headlessly with streamlit's AppTest against a populated request_log."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASH = str(Path(__file__).resolve().parent.parent / "gateway" / "dashboard.py")

from router import db as dbm  # noqa: E402
from router.config import CHEAP, STRONG  # noqa: E402


def _populate(path, n=120):
    conn = dbm.init_db(path)
    for i in range(n):
        cheap = i % 3 != 0
        conn.execute(
            "INSERT INTO request_log (request_id, selected_model, reason, p_cheap, threshold, overhead_ms, upstream_ms, status, fallback, "
            "escalated, stream, prompt_chars, prompt_tokens, completion_tokens, cost_usd, est_saved_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"r{i}", (CHEAP if cheap else STRONG).name, "ml", 0.7 if cheap else 0.2, 0.6, 5.0 + i % 7, 300.0, 200, int(i % 40 == 0),
             int(i % 50 == 0), 0, 100, 50, 40, 1e-5, 4e-5 if cheap else 0.0))
    conn.execute("INSERT INTO shadow_log (request_id, prompt, p_cheap, cheap_response, strong_response, cheap_cost, strong_cost, agree) "
                 "VALUES ('r1','p',0.7,'8','8',1e-5,5e-5,1.0)")
    conn.commit()
    conn.close()


def test_dashboard_renders_metrics_without_exceptions(tmp_path, monkeypatch):
    db = tmp_path / "log.db"
    _populate(db)
    monkeypatch.setenv("ROUTER_LOG_DB", str(db))
    monkeypatch.setenv("CHEAP_MODEL", CHEAP.name)
    at = AppTest.from_file(DASH, default_timeout=30).run()
    assert not at.exception, at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Requests"] == "120" and labels["Routed cheap"] == "67%"
    assert "Numeric-answer agreement" in labels


def test_dashboard_shows_friendly_message_when_empty(tmp_path, monkeypatch):
    db = tmp_path / "empty.db"
    dbm.init_db(db).close()
    monkeypatch.setenv("ROUTER_LOG_DB", str(db))
    at = AppTest.from_file(DASH, default_timeout=30).run()
    assert not at.exception and any("No traffic" in i.value for i in at.info)
