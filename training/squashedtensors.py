# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Model serialization in the squashedtensors '.sqt' format."""

import argparse
import contextlib
import datetime
import io
import json
import math
import re
import struct
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal, Optional, Union

import torch
import transformers
import weight_formats.quantisation_training as T
from weight_formats.nearest_neighbour import nearest_neighbour
from torch import Tensor, nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mllama.image_processing_mllama import MllamaImageProcessor
from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration


FILE_VERSION = 2


def align(n: int, alignment: int) -> int:
    return n + (-n % alignment)


def safe_div(a: Tensor, b: Tensor) -> Tensor:
    return a / torch.where(b == 0, 1, b)


@dataclass
class TensorData:
    dtype: Literal["BF16", "INT8", "S3D8"]
    shape: tuple[int, ...]
    tensor: Tensor
    table: Optional["TensorData"]
    scale: Optional["TensorData"]

    def __post_init__(self) -> None:
        if self.dtype == "BF16":
            assert self.tensor.dtype == torch.bfloat16
            assert self.tensor.nelement() == math.prod(self.shape)
        elif self.dtype == "INT8":
            assert self.tensor.dtype == torch.int8
            assert self.tensor.nelement() == math.prod(self.shape)
        elif self.dtype == "S3D8":
            assert self.tensor.dtype == torch.uint8
            assert self.table is not None
            assert self.table.dtype == "INT8"
        assert self.tensor.ndim == 1
        if self.scale is not None:
            assert self.scale.dtype == "BF16"

    def __repr__(self) -> str:
        return f"TensorData({self.dtype}, {self.shape}, table={self.table}, scale={self.scale})"

    def to_bf16(self) -> Tensor:
        assert self.dtype != "S3D8", "S3D8 tensors do not support to_bf16()"
        data = self.tensor.to(torch.bfloat16).reshape(self.shape)
        if self.scale is not None:
            data = data * self.scale.to_bf16()
        return data

    def mul_(self, t: Tensor) -> None:
        assert (
            t.ndim == 0 and t.dtype == torch.bfloat16
        ), "Only scalar bfloat16 multiplication is supported"
        if self.scale is not None:
            self.scale.tensor.mul_(t.to(self.scale.tensor.device))
        else:
            self.tensor.mul_(t.to(self.tensor.device))


def to_tensor_data(t: Tensor | T.Weight) -> TensorData:
    if isinstance(t, Tensor):
        return TensorData(
            dtype="BF16",
            shape=tuple(t.shape),
            tensor=t.data.to(torch.bfloat16, copy=True).flatten(),
            table=None,
            scale=None,
        )

    if isinstance(t, T.UnquantisedWeight):
        return to_tensor_data(t.weight)

    if isinstance(t, T.Weight):
        # CHANNEL_INT8
        assert not hasattr(t, "sparse_idx"), "Sparse tensors are not supported"
        assert torch.allclose(
            t.centroids,
            torch.arange(-128, 128, device=t.centroids.device, dtype=t.centroids.dtype),
        ), "Only INT8 is supported"
        scale = t._get_scale()
        tensor = (
            safe_div(t.master.reshape(t._blocked_shape), scale)
            .round()
            .clip(-128, 127)
            .to(torch.int8)
            .flatten()
        )
        return TensorData(
            dtype="INT8",
            shape=tuple(t.shape),
            tensor=tensor,
            table=None,
            scale=to_tensor_data(T._squeeze_odd_dims(scale)),
        )

    if isinstance(t, T.Sign3D8Weight):
        # CHANNEL_S3D8
        scale = t._get_scale()
        tensor = safe_div(t.master.reshape(t._blocked_shape), scale).reshape(t.shape)
        # Pad output channels to multiple of 3, input channels to multiple of 16
        # Each vector is of 3 output channels, but after this it's output-major
        d_out, d_in = t.shape
        tensor = (
            torch.nn.functional.pad(tensor, (0, -(d_in % -16), 0, -(d_out % -3)))
            .unflatten(0, (-1, 3))
            .permute(0, 2, 1)
            .flatten(end_dim=1)
        )
        idx = nearest_neighbour(tensor.abs(), t.centroids)
        sign = tensor.lt(0)
        tensor_data = (
            (idx << 1)
            | sign[:, 0].to(idx.dtype)
            | (sign[:, 1].to(idx.dtype) << 6)
            | ((sign[:, 0] ^ sign[:, 2]).to(idx.dtype) << 7)
        ).to(torch.uint8)
        return TensorData(
            dtype="S3D8",
            shape=tuple(t.shape),
            tensor=tensor_data,
            table=TensorData(
                dtype="INT8",
                shape=tuple(t.centroids.shape),
                tensor=t.centroids.to(torch.int8, copy=True).flatten(),
                table=None,
                scale=None,
            ),
            scale=to_tensor_data(T._squeeze_odd_dims(scale)),
        )


def rope_angular_frequency(config: transformers.PretrainedConfig) -> Tensor:
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    freq = config.rope_theta ** -(
        torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim
    )
    if s := config.rope_scaling:
        z = (
            s["original_max_position_embeddings"] * freq / (2 * torch.pi)
            - s["low_freq_factor"]
        ) / (s["high_freq_factor"] - s["low_freq_factor"])
        freq *= torch.lerp(
            torch.tensor(1 / s["factor"]), torch.tensor(1.0), z.clip(0, 1)
        )
    return freq


def get_vocab_dict(tokenizer: transformers.PreTrainedTokenizerFast) -> dict[str, Any]:
    # Persist-to-file & load to get acccess to the tokenizer internals
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer.backend_tokenizer.save(tmp + "/tokenizer.json")
        with open(tmp + "/tokenizer.json") as f:
            data = json.load(f)

    # Checks, pre-tokenizer regex, special IDs
    assert data["model"]["type"] == "BPE"
    if not data["model"]["ignore_merges"]:
        warnings.warn(
            "The tokenizer does not set `ignore_merges` (which matches full tokens first)"
            ", but our codebase always behaves as if `ignore_merges` is `True`"
        )
    pre_split, pre_byte = data["pre_tokenizer"]["pretokenizers"]
    pre_regex = pre_split["pattern"]["Regex"]
    assert pre_byte["type"] == "ByteLevel"

    def _special_token_id(token: str, required: bool = True) -> int:
        matches = [t["id"] for t in data["added_tokens"] if t["content"] == token]
        if len(matches) > 1:
            raise ValueError(f"Multiple token IDs for token {token!r}")
        if required and not matches:
            raise ValueError(f"Special token {token!r} not found")
        return matches[0] if matches else None

    # Concatenate merges to single strings & de-duplicate
    merge_set = set([])
    merges = []
    for a, b in data["model"]["merges"]:
        merge = a + b
        if merge not in merge_set:
            merge_set.add(merge)
            merges.append(merge)

    # Convert vocab from a dict to a list
    vocab = [None] * len(data["model"]["vocab"])
    for token, id in data["model"]["vocab"].items():
        vocab[id] = token
    assert all(token is not None for token in vocab)

    return dict(
        begin_of_text_id=_special_token_id("<|begin_of_text|>"),
        end_of_text_id=_special_token_id("<|end_of_text|>"),
        image_id=_special_token_id("<|image|>", required=False),
        pre_tokenizer=pre_regex,
        merges=merges,
        vocab=vocab,
    )


def standardise_name(name: str) -> str:
    name = re.sub(
        r"^multi_modal_projector\.", "vision_model.multi_modal_projector.", name
    )
    name = re.sub(r"^language_model\.model\.", "text_model.", name)
    name = re.sub(r"^language_model\.lm_head\.", "text_model.lm_head.", name)
    name = re.sub(r"^model\.", "text_model.", name)
    name = re.sub(r"^lm_head\.weight$", "text_model.lm_head.weight", name)
    name = re.sub(r"\.(cross_attn|self_attn)\.", ".attn.", name)
    name = re.sub(r"\.(fc1)\.", ".up_proj.", name)
    name = re.sub(r"\.(fc2)\.", ".down_proj.", name)
    name = re.sub(r"\.input_layernorm\.", ".attn.norm.", name)
    name = re.sub(r"\.post_attention_layernorm\.", ".mlp.norm.", name)
    name = re.sub(r"\.transformer\.layers\.", ".layers0.", name)
    name = re.sub(r"\.global_transformer\.layers\.", ".layers1.", name)
    return name


@torch.no_grad()
def prepare_parameters(
    model: LlamaForCausalLM | MllamaForConditionalGeneration,
) -> dict[str, TensorData]:
    """Rename parameters for uniform storage between models."""
    params = {}
    force_bf16 = {
        "vision_model.multi_modal_projector.weight",
        "vision_model.gated_positional_embedding.tile_embedding.weight",
        "vision_model.pre_tile_positional_embedding.embedding.weight",
        "vision_model.post_tile_positional_embedding.embedding.weight",
    }

    def _visit(m: nn.Module, prefix: tuple[str, ...] = ()) -> None:
        for name, child in m.named_children():
            _visit(child, prefix + (name,))
        if isinstance(m, (T.UnquantisedWeight, T.Weight, T.Sign3D8Weight)):
            full_name = standardise_name(".".join(prefix))
            params[full_name] = to_tensor_data(m() if full_name in force_bf16 else m)
        else:
            for name, param in m.named_parameters(recurse=False):
                params[standardise_name(".".join(prefix + (name,)))] = to_tensor_data(
                    param
                )

    _visit(model)

    # Simplify downstream implementations by merging gates into linear projections
    # and combining embeddings. Note:
    # - Original embedding shapes are sometimes flattened, hence `unflatten()`
    # - Original model uses 1-based indexing of aspect ratios, hence `[1:]` to
    #   revert to 0-based indexing.
    if isinstance(model, MllamaForConditionalGeneration):
        # Merge cross-attention gates into output projections
        for i in model.config.text_config.cross_attention_layers:
            prefix = f"text_model.layers.{i}"
            gate = f"{prefix}.cross_attn_attn_gate"
            weight = f"{prefix}.attn.o_proj.weight"
            params[weight].mul_(params.pop(gate).to_bf16().view(()).tanh())
            gate = f"{prefix}.cross_attn_mlp_gate"
            weight = f"{prefix}.mlp.down_proj.weight"
            params[weight].mul_(params.pop(gate).to_bf16().view(()).tanh())

        # Merge global_transformer (layers1) gates into output projections
        for i in range(model.config.vision_config.num_global_layers):
            prefix = f"vision_model.layers1.{i}"
            gate = f"{prefix}.gate_attn"
            weight = f"{prefix}.attn.o_proj.weight"
            params[weight].mul_(params.pop(gate).to_bf16().view(()).tanh())
            gate = f"{prefix}.gate_ffn"
            gate_tensor = params.pop(gate).to_bf16().view(()).tanh()
            weight = f"{prefix}.mlp.down_proj.weight"
            params[weight].mul_(gate_tensor)
            bias = f"{prefix}.mlp.down_proj.bias"
            params[bias].mul_(gate_tensor)

        n_tiles = model.config.vision_config.max_num_tiles
        n_patches = (
            model.config.vision_config.image_size
            // model.config.vision_config.patch_size
        ) ** 2

        # Create merged positional_embedding and class_embedding
        prefix = "vision_model.gated_positional_embedding"
        embedding0 = params.pop(f"{prefix}.embedding").to_bf16()
        gate = params.pop(f"{prefix}.gate").to_bf16().view(()).tanh()
        tile_embedding = (
            params.pop(f"{prefix}.tile_embedding.weight")
            .to_bf16()[1:]
            .unflatten(1, (n_tiles, n_patches + 1, -1))
        )
        embedding0 = embedding0 * (1 - gate) + tile_embedding * gate
        prefix = "vision_model.pre_tile_positional_embedding"
        pre_tile_embedding = (
            params.pop(f"{prefix}.embedding.weight")
            .to_bf16()[1:]
            .unflatten(1, (n_tiles, -1))
        )
        gate = params.pop(f"{prefix}.gate").to_bf16().view(()).tanh()
        params["vision_model.positional_embedding.weight"] = to_tensor_data(
            pre_tile_embedding.unsqueeze(2) * gate + embedding0[:, :, 1:, :]
        )
        params["vision_model.class_embedding.weight"] = to_tensor_data(
            params.pop("vision_model.class_embedding").to_bf16()
            + embedding0[:, :, 0, :]
        )

        # Merge post_tile_positional_embedding gate
        prefix = "vision_model.post_tile_positional_embedding"
        weight = (
            params.pop(f"{prefix}.embedding.weight")
            .to_bf16()[1:]
            .unflatten(1, (n_tiles, -1))
        )
        gate = params.pop(f"{prefix}.gate").to_bf16().view(()).tanh()
        params[f"vision_model.tile_embedding_post.weight"] = to_tensor_data(
            weight * gate
        )

        # Permute the multi_modal_projector input dimensions.
        # In the original weights, the first block of 1280 elements is the final
        # hidden state of `layers1`, but after this, the "tapped" hidden states
        # are interleaved. This is bothersome to implement, so instead we make things
        # explicit with a (n_taps + 1, d_out, d_in) tensor.
        proj = params["vision_model.multi_modal_projector.weight"].to_bf16()
        hidden_size = model.config.vision_config.hidden_size
        params["vision_model.multi_modal_projector.weight"] = to_tensor_data(
            torch.cat(
                [
                    proj[:, hidden_size:].unflatten(1, (hidden_size, -1)).movedim(2, 0),
                    proj[None, :, :hidden_size],
                ],
                axis=0,
            )
        )

    return params


def get_config_dict(
    config: transformers.PretrainedConfig, image_processor: MllamaImageProcessor | None
) -> dict[str, Any]:
    if hasattr(config, "vision_config"):
        text, vision = config.text_config, config.vision_config
    else:
        text, vision = config, None
    return dict(
        text=dict(
            d_layers=text.num_hidden_layers,
            d_vocab=text.vocab_size,
            d_model=text.hidden_size,
            d_mlp=text.intermediate_size,
            d_attention_head=getattr(
                text, "head_dim", text.hidden_size // text.num_attention_heads
            ),
            d_attention_q=text.num_attention_heads // text.num_key_value_heads,
            d_attention_kv=text.num_key_value_heads,
            d_sequence_max=text.max_position_embeddings,
            norm_epsilon=text.rms_norm_eps,
            rope_angular_frequency=rope_angular_frequency(text).tolist(),
            tied_embeddings=text.tie_word_embeddings,
            cross_attention_layers=getattr(text, "cross_attention_layers", []),
        ),
        vision=(
            dict(
                # Input
                image_mean=image_processor.image_mean,
                image_std=image_processor.image_std,
                d_image=vision.image_size,
                d_patch=vision.patch_size,
                # Core
                d_layers0=vision.num_hidden_layers,
                d_layers1=vision.num_global_layers,
                d_model=vision.hidden_size,
                d_mlp=vision.intermediate_size,
                d_attention_head=vision.hidden_size // vision.attention_heads,
                d_attention_qkv=vision.attention_heads,
                norm_epsilon=vision.norm_eps,
                # Output
                output_taps=vision.intermediate_layers_indices,
            )
            if vision
            else None
        ),
    )


def save(
    model: LlamaForCausalLM | MllamaForConditionalGeneration,
    tokenizer: transformers.PreTrainedTokenizerFast,
    image_processor: MllamaImageProcessor | None,
    file_or_path: Union[str, Path, IO[bytes]],
    alignment: int = 32,
) -> None:
    """Save a Llama model to '.sqt' format."""
    header_dict = dict(
        __metadata__=dict(
            source=model.config._name_or_path,
            created=datetime.datetime.now().isoformat(timespec="seconds"),
            alignment=alignment,
            config=get_config_dict(model.config, image_processor),
            vocab=get_vocab_dict(tokenizer),
        )
    )
    params = prepare_parameters(model)

    # Measure serialized tensors, but discard them (for sake of memory usage)
    last_offset = 0

    def _generate_header(t: TensorData) -> dict[str, Any]:
        nonlocal last_offset
        n_bytes = t.tensor.element_size() * t.tensor.numel()
        d = dict(
            dtype=t.dtype,
            shape=list(t.shape),
            data_offsets=[last_offset, last_offset + n_bytes],
        )
        last_offset += align(n_bytes, alignment)
        return d

    for name, t in params.items():
        # Note: order MUST match the buffer write below
        header_dict[name] = _generate_header(t)
        if t.table is not None:
            header_dict[name]["table"] = _generate_header(t.table)
        if t.scale is not None:
            header_dict[name]["scale"] = _generate_header(t.scale)

    with contextlib.ExitStack() as stack:
        f = (
            file_or_path
            if isinstance(file_or_path, io.IOBase)
            else stack.enter_context(open(file_or_path, "wb"))
        )

        f.write(b".sqt")  # magic
        f.write(struct.pack("<I", FILE_VERSION))  # version
        header = json.dumps(header_dict, separators=(",", ":")).encode("utf8")
        nheader_pad = -(16 + len(header)) % alignment
        f.write(struct.pack("<Q", len(header) + nheader_pad))  # header length
        f.write(header + nheader_pad * b" ")  # header

        # buffer
        buffer_start = f.tell()
        assert buffer_start % alignment == 0
        for name, param_t in params.items():
            param_h = header_dict[name]
            for t, h in [
                (param_t, param_h),
                (param_t.table, param_h.get("table")),
                (param_t.scale, param_h.get("scale")),
            ]:
                if t is not None:
                    begin, end = h["data_offsets"]
                    assert f.tell() == buffer_start + begin
                    t_bytes = (
                        t.tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    )
                    f.write(t_bytes)
                    assert f.tell() == buffer_start + end
                    f.write((align(len(t_bytes), alignment) - len(t_bytes)) * b"\0")


def _run() -> None:
    parser = argparse.ArgumentParser(
        description="Serialize a Llama model to squashedtensors '.sqt' format"
    )
    parser.add_argument(
        "model_name_or_path",
        type=str,
        help="HuggingFace model name or path (e.g., 'meta-llama/Llama-3.2-1B-Instruct')",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output path for the '.sqt' file",
    )
    args = parser.parse_args()
    config = transformers.AutoConfig.from_pretrained(args.model_name_or_path)
    if config.model_type == "llama":
        model_cls = LlamaForCausalLM
        image_processor = None
    elif config.model_type == "mllama":
        model_cls = MllamaForConditionalGeneration
        image_processor = transformers.AutoProcessor.from_pretrained(
            args.model_name_or_path
        ).image_processor
    else:
        raise ValueError(
            f"Unsupported model type {config.model_type!r} for {args.model_name_or_path}"
            ", expected 'llama' or 'mllama'"
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = model_cls.from_pretrained(args.model_name_or_path)
    save(model, tokenizer, image_processor, args.output_path)


if __name__ == "__main__":
    _run()
