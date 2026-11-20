"""Feature extraction shared by training AND serving (single source of truth).

Three feature modes, used for the ablation study:
    emb    - sentence embeddings only
    rules  - hand-written rule flags only
    both   - concatenation
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .config import ONNX_DIR

FEATURE_MODES = ("emb", "rules", "both")
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------------------------------------------------
# Rule-based features
# --------------------------------------------------------------------------------------------
_CODE_KW = re.compile(
    r"\b(def|class|import|return|lambda|elif|while|print\(|function|public static|#include|"
    r"select\b[^;\n]{0,120}?\bfrom|console\.log|=>|async|await|stdout|regex|algorithm|compile|"
    r"python|javascript|java|c\+\+|sql|bash)\b", re.I)
_PY_SIG = re.compile(r"\bdef\s+\w+\s*\(")
_MCQ_OPT = re.compile(r"(?m)^\s*\(?[A-D][\.\)]\s+\S")
_MATH_SYM = re.compile(r"[=+\-*/^%\$<>]|\\frac|\\sqrt|math\.")
_REASON = re.compile(
    r"\b(how many|how much|total|each|per|remaining|average|ratio|percent|solve|calculate|"
    r"prove|derive|integral|equation|probability|step by step|minimum|maximum|altogether|"
    r"twice|half|fraction)\b", re.I)
_WRITE = re.compile(r"\b(write|create|generate|compose|draft|brainstorm|story|poem|essay|email)\b", re.I)
_TRANSFORM = re.compile(r"\b(summari[sz]e|tl;dr|extract|rewrite|translate|classify|paraphrase|list the)\b", re.I)
_WH = re.compile(r"^\s*(what|why|how|who|when|where|which|is|are|can|does|do)\b", re.I)
_URL = re.compile(r"https?://")

RULE_FEATURE_NAMES = [
    "log_chars", "log_words", "n_lines", "digit_ratio", "n_numbers", "math_sym_count",
    "has_code_fence", "code_kw_count", "has_py_signature", "has_mcq_options", "n_question_marks",
    "starts_wh", "reasoning_cues", "write_cue", "transform_cue", "avg_word_len",
    "upper_ratio", "non_ascii_ratio", "has_url", "n_sentences", "has_math_module",
]


MAX_RULE_CHARS = 20_000  # bound CPU per prompt (and ReDoS exposure); length features still use the true length


def rule_features(text: str) -> np.ndarray:
    full = text or ""
    full_len = len(full)
    t = full[:MAX_RULE_CHARS]
    n = max(len(t), 1)
    words = t.split()
    n_words = max(len(words), 1)
    feats = [
        math.log1p(full_len),
        math.log1p(len(words) if full_len <= MAX_RULE_CHARS else full_len / 5.5),
        float(t.count("\n")),
        sum(c.isdigit() for c in t) / n,
        float(len(re.findall(r"\d+(?:\.\d+)?", t))),
        float(len(_MATH_SYM.findall(t))),
        float("```" in t),
        float(len(_CODE_KW.findall(t))),
        float(bool(_PY_SIG.search(t))),
        float(bool(_MCQ_OPT.search(t))),
        float(t.count("?")),
        float(bool(_WH.match(t))),
        float(len(_REASON.findall(t))),
        float(bool(_WRITE.search(t))),
        float(bool(_TRANSFORM.search(t))),
        sum(len(w) for w in words) / n_words,
        sum(c.isupper() for c in t) / n,
        sum(ord(c) > 127 for c in t) / n,
        float(bool(_URL.search(t))),
        float(len(re.findall(r"[.!?](?:\s|$)", t))),
        float("math." in t),
    ]
    return np.asarray(feats, dtype=np.float32)


def rule_matrix(texts: Sequence[str]) -> np.ndarray:
    return np.vstack([rule_features(t) for t in texts]) if len(texts) else np.zeros((0, len(RULE_FEATURE_NAMES)), np.float32)


# --------------------------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------------------------
class Embedder(Protocol):
    spec: dict

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, name: str = ST_MODEL_NAME):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy
        self.model = SentenceTransformer(name)
        self.spec = {"backend": "st", "name": name}

    def encode(self, texts):
        return np.asarray(
            self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


class HashEmbedder:
    """Dependency-free deterministic embedder. For tests / CI / offline demos only."""

    def __init__(self, dim: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vec = HashingVectorizer(n_features=dim, ngram_range=(1, 2), norm="l2", alternate_sign=False)
        self.spec = {"backend": "hash", "dim": dim}

    def encode(self, texts):
        return np.asarray(self.vec.transform(list(texts)).todense(), dtype=np.float32)


class OnnxEmbedder:
    """all-MiniLM-L6-v2 via onnxruntime (+ `tokenizers`): no torch needed at serving time.
    Create the model directory with `python -m gateway.onnx_export`."""

    def __init__(self, model_dir: str | Path = ONNX_DIR, max_len: int = 256, quantized: bool = True, batch_size: int = 32):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        fname = "model_int8.onnx" if quantized and (model_dir / "model_int8.onnx").exists() else "model.onnx"
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
        self.session = ort.InferenceSession(str(model_dir / fname), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding()
        self.batch_size = batch_size
        self.spec = {"backend": "onnx", "name": ST_MODEL_NAME, "model_dir": str(model_dir), "file": fname}

    def encode(self, texts):
        texts = list(texts)
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            enc = self.tok.encode_batch(batch)
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feed)[0]  # (B, T, H)
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
            chunks.append(pooled.astype(np.float32))
        return np.vstack(chunks)


class CachedEmbedder:
    """Thread-safe LRU cache keyed on the prompt hash (repeated prompts are common in prod)."""

    def __init__(self, inner: Embedder, max_items: int = 20000):
        self.inner, self.max_items = inner, max_items
        self.spec = inner.spec
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def _key(t: str) -> str:
        return hashlib.blake2b(t.encode("utf-8", "ignore"), digest_size=16).hexdigest()

    def encode(self, texts):
        texts = list(texts)
        keys = [self._key(t) for t in texts]
        out: list[np.ndarray | None] = [None] * len(texts)
        miss_idx: list[int] = []
        with self._lock:
            for i, k in enumerate(keys):
                v = self._cache.get(k)
                if v is None:
                    miss_idx.append(i)
                else:
                    self._cache.move_to_end(k)
                    out[i] = v
        if miss_idx:
            vecs = self.inner.encode([texts[i] for i in miss_idx])
            with self._lock:
                for i, v in zip(miss_idx, vecs):
                    out[i] = v
                    self._cache[keys[i]] = v
                while len(self._cache) > self.max_items:
                    self._cache.popitem(last=False)
        self.hits += len(texts) - len(miss_idx)
        self.misses += len(miss_idx)
        return np.vstack(out) if out else np.zeros((0, 0), np.float32)


def make_embedder(spec: dict | None = None, cached: bool = True) -> Embedder:
    """Build an embedder from a spec dict (as stored in the router artifact) or from the
    ROUTER_EMBEDDER env var (st | onnx | hash). Env var wins so you can swap backends."""
    backend = os.environ.get("ROUTER_EMBEDDER") or (spec or {}).get("backend") or "st"
    if backend == "hash":
        emb: Embedder = HashEmbedder((spec or {}).get("dim", 384))
    elif backend == "onnx":
        model_dir = Path((spec or {}).get("model_dir") or ONNX_DIR)
        if not model_dir.exists():
            model_dir = ONNX_DIR
        emb = OnnxEmbedder(model_dir)
    else:
        emb = SentenceTransformerEmbedder((spec or {}).get("name", ST_MODEL_NAME))
    return CachedEmbedder(emb) if cached else emb


def build_features(texts: Sequence[str], embedder: Embedder | None, mode: str,
                   emb: np.ndarray | None = None) -> np.ndarray:
    """Feature matrix for `mode`. Pass a precomputed `emb` to avoid re-embedding in training."""
    if mode not in FEATURE_MODES:
        raise ValueError(f"mode must be one of {FEATURE_MODES}")
    parts = []
    if mode in ("emb", "both"):
        parts.append(emb if emb is not None else embedder.encode(texts))
    if mode in ("rules", "both"):
        parts.append(rule_matrix(texts))
    return np.hstack(parts).astype(np.float32)