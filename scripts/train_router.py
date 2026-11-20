"""Train + compare routers (selection on VALIDATION only) and write the gateway artifact.

    python -m scripts.train_router
    ROUTER_EMBEDDER=onnx python -m scripts.train_router    # train on the same embedder you will serve
"""
import argparse

from router.config import DB_PATH
from router.training import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--target", type=float, default=0.95, help="quality-retention target for tuning tau")
    ap.add_argument("--safety", type=float, default=0.01, help="extra margin added to the target on val")
    ap.add_argument("--label", choices=["as_good", "cheap_correct"], default="as_good")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--augment", type=int, default=0, help="perturbed copies per training prompt (robustness; see scripts.robustness)")
    a = ap.parse_args()
    train(a.db, label_mode=a.label, target=a.target, safety=a.safety, calibrate=not a.no_calibrate, augment=a.augment)
