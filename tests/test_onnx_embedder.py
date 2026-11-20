"""Tests the OnnxEmbedder code path with a tiny hand-built model (the real MiniLM export needs HF access)."""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from router.features import OnnxEmbedder  # noqa: E402

VOCAB = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3, "foo": 4, "bar": 5}
H = 8


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    table = np.random.default_rng(0).normal(size=(len(VOCAB), H)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("Gather", ["emb", "input_ids"], ["last_hidden_state"], axis=0)], "tiny",
        [helper.make_tensor_value_info(n, TensorProto.INT64, ["b", "t"]) for n in ("input_ids", "attention_mask", "token_type_ids")],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["b", "t", H])],
        initializer=[numpy_helper.from_array(table, "emb")])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.save(m, str(d / "model.onnx"))
    tok = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(d / "tokenizer.json"))
    return d, table


def test_onnx_embedder_matches_manual_mean_pool(model_dir):
    d, table = model_dir
    emb = OnnxEmbedder(d, quantized=False)
    out = emb.encode(["hello world", "foo"])
    assert out.shape == (2, H) and np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-5)
    expected = table[[2, 3]].mean(0)
    expected /= np.linalg.norm(expected)
    assert np.allclose(out[0], expected, atol=1e-5)


def test_padding_does_not_change_a_sentences_embedding(model_dir):
    d, _ = model_dir
    emb = OnnxEmbedder(d, quantized=False)
    alone = emb.encode(["foo bar"])[0]
    padded = emb.encode(["foo bar", "hello world foo bar hello"])[0]
    assert np.allclose(alone, padded, atol=1e-5)


# ---------------------------------------------------------------------------------------------
# int8 quantisation path with real MatMuls (no torch needed): embedding -> linear -> tanh
# ---------------------------------------------------------------------------------------------
def _dense_model(path, vocab=len(VOCAB), h=64, seed=1):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(vocab, h)).astype(np.float32)
    w = (rng.normal(size=(h, h)) / np.sqrt(h)).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("Gather", ["emb", "input_ids"], ["e"], axis=0),
         helper.make_node("MatMul", ["e", "w"], ["m"]),
         helper.make_node("Tanh", ["m"], ["last_hidden_state"])], "dense",
        [helper.make_tensor_value_info(n, TensorProto.INT64, ["b", "t"]) for n in ("input_ids", "attention_mask", "token_type_ids")],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["b", "t", h])],
        initializer=[numpy_helper.from_array(emb, "emb"), numpy_helper.from_array(w, "w")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.save(m, str(path))
    return emb, w


def test_dynamic_int8_quantisation_keeps_embeddings_close_and_file_smaller(tmp_path_factory, model_dir):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    d = tmp_path_factory.mktemp("dense")
    emb, w = _dense_model(d / "model.onnx")
    quantize_dynamic(str(d / "model.onnx"), str(d / "model_int8.onnx"), weight_type=QuantType.QInt8)
    (d / "tokenizer.json").write_bytes((model_dir[0] / "tokenizer.json").read_bytes())
    fp, q = OnnxEmbedder(d, quantized=False), OnnxEmbedder(d, quantized=True)
    assert q.spec["file"] == "model_int8.onnx" and fp.spec["file"] == "model.onnx"
    texts = ["hello world", "foo bar foo", "hello", "bar bar world foo"]
    a, b = fp.encode(texts), q.encode(texts)
    ref = np.tanh(emb[[2, 3]] @ w).mean(0)                       # numpy reference for "hello world"
    assert np.allclose(a[0], ref / np.linalg.norm(ref), atol=1e-4)
    assert ((a * b).sum(1)).min() > 0.99
    assert (d / "model_int8.onnx").stat().st_size < (d / "model.onnx").stat().st_size
