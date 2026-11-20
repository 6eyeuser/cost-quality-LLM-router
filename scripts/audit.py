"""Sanity-audit the benchmark BEFORE training: truncation, empty answers, per-domain accuracy, cost ratio."""
import pandas as pd

from router import db as dbm
from router.evaluation import load_frame


def main():
    conn = dbm.connect()
    df = load_frame(conn)
    ev = pd.read_sql_query("SELECT e.*, q.domain FROM evaluation_cache e JOIN queries q USING(query_id)", conn)
    cheap, strong = df.attrs["cheap"], df.attrs["strong"]
    print(f"cheap={cheap}  strong={strong}  queries with both={len(df)}\n")

    ev["truncated"] = ev["finish_reason"].eq("length")
    ev["empty"] = ev["response_text"].fillna("").str.strip().eq("")
    g = ev.groupby(["model_name", "domain"]).agg(n=("query_id", "size"), acc=("is_correct", "mean"),
                                                 truncated=("truncated", "mean"), empty=("empty", "mean"),
                                                 avg_cost=("cost_usd", "mean"), avg_out_tok=("output_tokens", "mean"),
                                                 avg_reasoning=("reasoning_tokens", "mean"))
    print(g.round(4).to_string(), "\n")
    print("Overall accuracy: cheap %.1f%%  strong %.1f%%" % (100 * df.correct_cheap.mean(), 100 * df.correct_strong.mean()))
    print("Effective cost ratio strong/cheap: %.2fx  (this caps your maximum savings)" % (df.cost_strong.mean() / df.cost_cheap.mean()))
    ceil = 1 - df.cost_cheap.mean() / df.cost_strong.mean()
    print("Absolute savings ceiling (everything cheap): %.1f%%" % (100 * ceil))
    oracle = ((df.correct_cheap >= df.correct_strong)).mean()
    print("Share of queries where cheap is as good as strong: %.1f%%" % (100 * oracle))
    for m, name in (("cheap", cheap), ("strong", strong)):
        t = ev[(ev.model_name == name) & ev.truncated].shape[0]
        if t:
            print(f"WARNING: {t} {m}-model answers were truncated (finish_reason=length) and probably graded wrong. "
                  f"Raise BENCH_MAX_TOKENS and re-run.")


if __name__ == "__main__":
    main()
