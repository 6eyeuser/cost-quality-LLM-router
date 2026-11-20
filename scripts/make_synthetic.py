"""Fill a throw-away DB with synthetic data to try the pipeline offline:
    ROUTER_DB=data/synthetic.db ROUTER_EMBEDDER=hash python -m scripts.make_synthetic
"""
import argparse

from router import db as dbm
from router.config import DB_PATH
from router.synthetic import populate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    populate(dbm.init_db(a.db), a.n)
    print(f"Synthetic data written to {a.db}. These numbers are NOT real results.")
