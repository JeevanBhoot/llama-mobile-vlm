import urllib

import PIL
import torch
import transformers

from utility import (
    batches,
    distributed_batches,
    merge_params_mllama,
    set_padding_side_left,
    from_dict,
)


def test_batches() -> None:
    assert list(batches("abcdefg", 3)) == [list("abc"), list("def"), list("g")]
    assert list(batches("abcdefg", 3, drop_last=True)) == [list("abc"), list("def")]


def test_set_padding_side_left() -> None:
    processor = transformers.AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct"
    )
    text = ["Text", "Text text text text"]
    with set_padding_side_left(processor.tokenizer):
        tok_out = processor.tokenizer(text, padding=True, return_tensors="pt")
        assert tok_out["input_ids"].shape == (2, 5)
        assert tok_out["attention_mask"].shape == (2, 5)
        assert torch.all(
            tok_out["input_ids"][0, :3] == processor.tokenizer.pad_token_id
        )
        assert torch.all(~tok_out["input_ids"][0, :3])


def test_distributed_batches() -> None:
    items = range(8)
    world_size = 2
    batch_size = 4
    out = {}
    for rank in range(world_size):
        out[rank] = list(distributed_batches(items, batch_size, rank, world_size))
    assert out[0] == [[0, 1], [4, 5]]
    assert out[1] == [[2, 3], [6, 7]]


def test_merge_params_mllama() -> None:
    name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    m = transformers.MllamaVisionModel.from_pretrained(
        name,
        torch_dtype=torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Sanity check, not necessary
    for layer in m.global_transformer.layers:
        assert layer.is_gated
        assert hasattr(layer, "gate_attn")
        assert hasattr(layer, "gate_ffn")

    processor = transformers.AutoProcessor.from_pretrained(name)

    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    image = PIL.Image.open(urllib.request.urlopen(image_url))
    inp = processor.image_processor(image, return_tensors="pt")
    inp.pop("num_tiles")
    inp.to(m.device)

    with torch.no_grad():
        expected = m(**inp)["last_hidden_state"]

    # In-place
    merge_params_mllama(m)

    for layer in m.global_transformer.layers:
        assert not layer.is_gated
        assert not hasattr(layer, "gate_attn")
        assert not hasattr(layer, "gate_ffn")

    with torch.no_grad():
        actual = m(**inp)["last_hidden_state"]
    torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-4)


def test_from_dict() -> None:
    from dataclasses import dataclass, asdict

    @dataclass
    class ConfigB:
        f1: tuple[float]

    @dataclass
    class ConfigA:
        f1: int
        f2: list[float]
        f3: ConfigB | None

    cfg = ConfigA(f1=5, f2=[0.3, 0.5, 0.7], f3=ConfigB(f1=(-0.1, -0.3)))
    d = asdict(cfg)
    assert cfg == from_dict(ConfigA, d)
