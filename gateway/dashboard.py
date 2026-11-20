"""Live dashboard:  pip install streamlit && streamlit run gateway/dashboard.py"""
import os

import pandas as pd
import streamlit as st

from router import db as dbm
from router.config import DB_PATH

st.set_page_config(page_title="LLM Router", layout="wide")
st.title("Cost-Quality LLM Router")
conn = dbm.connect(os.environ.get("ROUTER_LOG_DB", DB_PATH), check_same_thread=False)
req = pd.read_sql_query("SELECT * FROM request_log ORDER BY id", conn)
if req.empty:
    st.info("No traffic yet.")
    st.stop()
req["ts"] = pd.to_datetime(req["ts"])
cheap = req["selected_model"].str.contains(os.environ.get("CHEAP_MODEL", "llama-3.1-8b-instant"))
c = st.columns(5)
c[0].metric("Requests", len(req))
c[1].metric("Routed cheap", f"{cheap.mean():.0%}")
c[2].metric("Est. $ saved", f"${req.est_saved_usd.fillna(0).sum():.4f}")
c[3].metric("Overhead p95", f"{req.overhead_ms.quantile(.95):.1f} ms")
c[4].metric("Fallbacks / escalations", f"{int(req.fallback.sum())} / {int(req.escalated.sum())}")
st.subheader("Routing mix over time (rolling % cheap)")
st.line_chart(cheap.rolling(50, min_periods=5).mean().set_axis(req["ts"]))
st.subheader("Routing reasons")
st.bar_chart(req["reason"].value_counts())
st.subheader("Mean P(cheap) (drift signal)")
st.line_chart(req["p_cheap"].rolling(50, min_periods=5).mean().set_axis(req["ts"]))
sh = pd.read_sql_query("SELECT * FROM shadow_log", conn)
if not sh.empty:
    st.subheader("Shadow mode")
    st.metric("Numeric-answer agreement", f"{sh.agree.dropna().mean():.0%}")
    st.dataframe(sh.tail(20))
