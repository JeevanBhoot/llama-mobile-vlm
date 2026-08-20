# Prebuilt Inference Models

The public release models for the C++ inference library are stored at:

```text
s3://graphcore-research-public/2026-llama-mobile/models/20260611/
```

They are `.sqt` files using the format documented in [`file_format.md`](file_format.md). The files contain model weights, quantized weight metadata, tokenizer data, chat-template metadata, and model configuration in a single file.

See the project [model card](MODEL_CARD.md) for intended use, limitations,
licensing, and release checksums.

## Models

| File | Size | Source model | Weights | Intended use |
| --- | ---: | --- | --- | --- |
| `vision-11B-s3d8.sqt` | 3.73 GB | `meta-llama/Llama-3.2-11B-Vision-Instruct` | S3D8 weights / INT8 activations, QAT | Trained model supporting the paper's main task results [proud-sponge-1878] |
| `vision-11B-bf16.sqt` | 21.32 GB | `meta-llama/Llama-3.2-11B-Vision-Instruct` | BF16 baseline | Vision-language baseline for quality and performance comparisons |
| `text-1B-s3d8.sqt` | 506 MB | `meta-llama/Llama-3.2-1B-Instruct` | S3D8 direct cast | Smallest text-only model for smoke testing (note: direct-cast performance of S3D8 is poor) |
| `text-1B-int8.sqt` | 1.50 GB | `meta-llama/Llama-3.2-1B-Instruct` | INT8 direct cast | Lightweight text-only CLI and benchmark smoke tests |
| `text-1B-bf16.sqt` | 3.00 GB | `meta-llama/Llama-3.2-1B-Instruct` | BF16 baseline | Text-only baseline for quality and performance comparisons |

## Download

From `inference/`:

```sh
mkdir -p models/
aws s3 sync --no-sign-request \
  s3://graphcore-research-public/2026-llama-mobile/models/20260611/ \
  models/

cd models
sha256sum --check SHA256SUMS
```

Or download individual files:

```sh
for name in \
  text-1B-bf16.sqt \
  text-1B-int8.sqt \
  text-1B-s3d8.sqt \
  vision-11B-bf16.sqt \
  vision-11B-s3d8.sqt
do
  aws s3 cp --no-sign-request \
    s3://graphcore-research-public/2026-llama-mobile/models/20260611/${name} \
    models/${name}
done
```

## Provenance

The `vision-11B-*.sqt` files are conversions of Meta Llama 3.2 11B Vision
Instruct. `vision-11B-bf16.sqt` is the BF16 baseline. `vision-11B-s3d8.sqt` is
the Llama-Mobile QAT vision-language model used as the primary compressed
inference artifact and stores the paper's S3D8 weight format for efficient Arm
CPU execution.

The `text-1B-*.sqt` files are conversions of Meta Llama 3.2 1B Instruct for
text-only testing and examples. `text-1B-bf16.sqt` is the BF16 baseline,
`text-1B-int8.sqt` is a direct-cast INT8 conversion, and `text-1B-s3d8.sqt` is a
direct-cast S3D8 conversion. They are useful for checking the inference library
and app without downloading the larger vision-language models.

The commands below document how project checkpoints were converted into `.sqt`
files for the C++ inference library. They require Meta Llama model access on
HuggingFace and, for QAT checkpoints, project checkpoint files. They were generated using [../training/squashedtensors.py](../training/squashedtensors.py).

**Conversion commands:**

```sh
mkdir -p models/v3/ tmp/

# Text | BF16 baseline
python squashedtensors.py meta-llama/Llama-3.2-1B-Instruct models/v3/text-1B-bf16.sqt --comment "Llama 3.2 1B Instruct, BF16"

# Text | INT8 direct cast
python direct_cast.py meta-llama/Llama-3.2-1B-Instruct tmp/text-1B-int8.safetensors --format int8
python squashedtensors.py meta-llama/Llama-3.2-1B-Instruct models/v3/text-1B-int8.sqt --checkpoint tmp/text-1B-int8.safetensors --comment "Llama 3.2 1B Instruct, INT8 (Direct Cast)"

# Text | S3D8 direct cast
python direct_cast.py meta-llama/Llama-3.2-1B-Instruct tmp/text-1B-s3d8.safetensors --format s3d8
python squashedtensors.py meta-llama/Llama-3.2-1B-Instruct models/v3/text-1B-s3d8.sqt --checkpoint tmp/text-1B-s3d8.safetensors --comment "Llama 3.2 1B Instruct, S3D8 (Direct Cast)"

# Vision | BF16 baseline
python squashedtensors.py meta-llama/Llama-3.2-11B-Vision-Instruct models/v3/vision-11B-bf16.sqt --comment "Llama 3.2 11B Vision Instruct, BF16"

# Vision | INT8 direct cast
python direct_cast.py meta-llama/Llama-3.2-11B-Vision-Instruct tmp/vision-11B-int8.safetensors --format int8
python squashedtensors.py meta-llama/Llama-3.2-11B-Vision-Instruct models/v3/vision-11B-int8.sqt --checkpoint tmp/vision-11B-int8.safetensors --comment "Llama 3.2 11B Vision Instruct, INT8 (Direct Cast)"

# Vision | QAT S3D8
python squashedtensors.py meta-llama/Llama-3.2-11B-Vision-Instruct models/v3/vision-11B-s3d8.sqt --checkpoint tmp/proud-sponge-1878.safetensors --comment "Llama 3.2 11B Vision Instruct, QAT (S3D8) [proud-sponge-1878]"
```


## HuggingFace-compatible checkpoints

We also provide checkpoints matching `vision-11B-s3d8.sqt`, in `.safetensors`
formats for the standalone Python demo.

The compressed demo checkpoint stores S3D8 weights packed as bytes, plus BF16
per-channel scales and small INT8 lookup tables. `demo.py` keeps these weights
packed in model state and dequantizes them layer-by-layer for PyTorch inference:

```text
s3://graphcore-research-public/2026-llama-mobile/hf_models/vision-11B-s3d8-packed.safetensors
```

The original training checkpoint stores QAT master weights and centroids:

```text
s3://graphcore-research-public/2026-llama-mobile/hf_models/vision-11B-s3d8-training.safetensors
```

An earlier compatibility checkpoint contains the S3D8 weights rounded back to
BF16 for loading by standard Transformers model classes:

```text
s3://graphcore-research-public/2026-llama-mobile/hf_models/vision-11B-s3d8-as-bf16.safetensors
```

**Conversion commands:**

To convert the training checkpoint into the compressed demo checkpoint:

```sh
python training/compress_s3d8_checkpoint.py vision-11B-s3d8-training.safetensors vision-11B-s3d8-packed.safetensors
```

The `hf_models/` prefix also contains `LICENSE`, `NOTICE`, `MODEL_CARD.md`, and
`SHA256SUMS`. Download and verify a checkpoint with:

```sh
for name in LICENSE NOTICE MODEL_CARD.md SHA256SUMS vision-11B-s3d8-packed.safetensors; do
  aws s3 cp --no-sign-request \
    s3://graphcore-research-public/2026-llama-mobile/hf_models/${name} .
done
sha256sum --check --ignore-missing SHA256SUMS
```
