"""Central configuration. Everything is overridable with environment variables.

IMPORTANT: prices change. Verify them against https://groq.com/pricing before you
trust any cost-savings number, and override with the CHEAP_IN/CHEAP_OUT/STRONG_IN/
STRONG_OUT environment variables if they differ.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("ROUTER_DB", ROOT / "data" / "router_v2.db"))
ARTIFACT_PATH = Path(os.environ.get("ROUTER_ARTIFACT", ROOT / "gateway" / "models" / "router.joblib"))
RESULTS_DIR = Path(os.environ.get("ROUTER_RESULTS", ROOT / "results"))
ONNX_DIR = Path(os.environ.get("ROUTER_ONNX_DIR", ROOT / "gateway" / "models" / "onnx"))

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "https://api.groq.com/openai/v1")
SEED = int(os.environ.get("ROUTER_SEED", "42"))
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)  # train / val / test

# Generation settings used when building the benchmark. gpt-oss is a reasoning model:
# reasoning tokens count against max_tokens, so keep this generous and audit truncation.
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "2048"))
STRONG_REASONING_EFFORT = os.environ.get("STRONG_REASONING_EFFORT", "medium")


def get_api_key() -> str | None:
    for name in ("GROQ_API_KEY", "XAI_API_KEY"):  # XAI_API_KEY kept for backward compatibility
        if os.environ.get(name):
            return os.environ[name]
    return None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_per_m: float   # USD per 1M input tokens
    output_per_m: float  # USD per 1M output tokens

    def cost(self, in_tokens: int | None, out_tokens: int | None) -> float:
        return ((in_tokens or 0) * self.input_per_m + (out_tokens or 0) * self.output_per_m) / 1_000_000


# allam-2-7b is an Arabic-focused model; a general-purpose small model is a fairer "cheap tier".
# Run `python -m scripts.check_models` to see what your key can actually use.
CHEAP = ModelSpec(
    os.environ.get("CHEAP_MODEL", "llama-3.1-8b-instant"),
    float(os.environ.get("CHEAP_IN", "0.05")),
    float(os.environ.get("CHEAP_OUT", "0.08")),
)
STRONG = ModelSpec(
    os.environ.get("STRONG_MODEL", "openai/gpt-oss-120b"),
    float(os.environ.get("STRONG_IN", "0.15")),
    float(os.environ.get("STRONG_OUT", "0.60")),
)
# The judge should NOT be from the same family as the strong model (self-preference bias).
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama-3.3-70b-versatile")
JUDGE_PASS_SCORE = int(os.environ.get("JUDGE_PASS_SCORE", "7"))

MODELS = {"cheap": CHEAP, "strong": STRONG}
