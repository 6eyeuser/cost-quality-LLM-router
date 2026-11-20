"""Exercises gateway.onnx_export.export_model end-to-end with a tiny randomly-initialised BERT
(the real all-MiniLM-L6-v2 needs a Hugging Face download). Verifies: torch -> ONNX -> int8 -> OnnxEmbedder
reproduces PyTorch mean-pooled, L2-normalised embeddings."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("onnx")
from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from gateway.onnx_export import export_model  # noqa: E402
from router.features import OnnxEmbedder  # noqa: E402

WORDS = ["hello", "world", "a", "longer", "example", "sentence", "for", "tracing", "foo", "bar", "what", "is", "the", "capital", "of", "france"]
VOCAB = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS)}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.manual_seed(0)
    model = BertModel(BertConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                 intermediate_size=64, max_position_embeddings=64)).eval()
    raw = Tokenizer(models.WordPiece(VOCAB, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    raw.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]")
    out = tmp_path_factory.mktemp("export")
    fp32, int8 = export_model(model, tok, out)
    return model, tok, out, fp32, int8


def _torch_embed(model, tok, texts):
    b = tok(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        h = model(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    e = (h * m).sum(1) / m.sum(1)
    return torch.nn.functional.normalize(e, dim=1).numpy()


TEXTS = ["hello world", "what is the capital of france", "foo bar foo", "a longer example sentence for tracing", "unseenword here"]


def test_files_written_and_int8_is_smaller(tiny):
    _, _, out, fp32, int8 = tiny
    assert fp32.exists() and int8.exists() and (out / "tokenizer.json").exists()
    assert int8.stat().st_size < fp32.stat().st_size


def test_fp32_onnx_matches_pytorch_exactly_enough(tiny):
    model, tok, out, *_ = tiny
    ours = OnnxEmbedder(out, quantized=False).encode(TEXTS)
    ref = _torch_embed(model, tok, TEXTS)
    cos = (ours * ref).sum(1)
    assert cos.min() > 0.9999, cos


def test_int8_onnx_is_close_to_pytorch(tiny):
    model, tok, out, *_ = tiny
    emb = OnnxEmbedder(out, quantized=True)
    assert emb.spec["file"] == "model_int8.onnx"
    cos = (emb.encode(TEXTS) * _torch_embed(model, tok, TEXTS)).sum(1)
    assert cos.min() > 0.9, cos          # tiny random weights quantise worse than a trained model; real export prints the true figure


def test_batching_does_not_change_embeddings(tiny):
    _, _, out, *_ = tiny
    emb = OnnxEmbedder(out, quantized=False)
    batch = emb.encode(TEXTS)
    single = np.vstack([emb.encode([t]) for t in TEXTS])
    assert np.allclose(batch, single, atol=1e-4)
