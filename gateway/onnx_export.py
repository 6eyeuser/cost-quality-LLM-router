"""Export all-MiniLM-L6-v2 to ONNX, quantize to int8, and verify parity with the PyTorch model.

    pip install -r requirements-export.txt
    python -m gateway.onnx_export

Then serve without torch:   ROUTER_EMBEDDER=onnx uvicorn gateway.app:app
IMPORTANT: embeddings from the int8 model differ slightly from fp32. Train the router with the same
backend you serve (ROUTER_EMBEDDER=onnx python -m scripts.train_router) so there is no train/serve skew.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from router.config import ONNX_DIR
from router.features import ST_MODEL_NAME

SAMPLES = ["What is 17 * 23?", "Write a haiku about autumn.", "def fib(n): return n if n < 2 else fib(n-1)+fib(n-2)",
           "Summarize the following paragraph in one sentence. " * 6, "Prove that the square root of 2 is irrational."]


def export_model(model, tok, out_dir: Path) -> tuple[Path, Path]:
    """Core of the export, independent of where the model came from (so it can be tested with a tiny random BERT).
    `tok` must be a fast tokenizer (it is saved as tokenizer.json for the `tokenizers`-only serving path)."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    class _OnnxWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=False,
            )[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    wrapped = _OnnxWrapper(model.eval()).eval()
    tok.save_pretrained(out_dir)  # writes tokenizer.json used by `tokenizers` at serving time
    batch = tok(["hello world", "a longer example sentence for tracing"], padding=True, return_tensors="pt")
    token_type_ids = batch.get("token_type_ids", torch.zeros_like(batch["input_ids"]))
    args = (batch["input_ids"], batch["attention_mask"], token_type_ids)
    fp32 = out_dir / "model.onnx"
    kwargs = dict(input_names=["input_ids", "attention_mask", "token_type_ids"], output_names=["last_hidden_state"],
                  dynamic_axes={k: {0: "batch", 1: "seq"} for k in
                                ("input_ids", "attention_mask", "token_type_ids", "last_hidden_state")},
                  opset_version=17)
    with torch.no_grad():
        try:
            torch.onnx.export(wrapped, args, str(fp32), dynamo=False, **kwargs)
        except TypeError:  # older torch has no `dynamo` argument
            torch.onnx.export(wrapped, args, str(fp32), **kwargs)
    int8 = out_dir / "model_int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print(f"Wrote {fp32} ({fp32.stat().st_size / 1e6:.1f} MB) and {int8} ({int8.stat().st_size / 1e6:.1f} MB)")
    return fp32, int8


def export(out_dir: Path = ONNX_DIR, check: bool = True) -> None:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ST_MODEL_NAME)
    model = AutoModel.from_pretrained(ST_MODEL_NAME)
    export_model(model, tok, out_dir)
    if check:
        from router.features import OnnxEmbedder, SentenceTransformerEmbedder
        ref = SentenceTransformerEmbedder().encode(SAMPLES)
        for quant in (False, True):
            emb = OnnxEmbedder(out_dir, quantized=quant).encode(SAMPLES)
            cos = (ref * emb).sum(1)
            print(f"{'int8' if quant else 'fp32'} cosine vs PyTorch: min={cos.min():.4f} mean={cos.mean():.4f}")
            if cos.min() < 0.98:
                print("  WARNING: <0.98 similarity. Retrain the router on this backend before serving it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ONNX_DIR))
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    export(Path(a.out), not a.no_check)