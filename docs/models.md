# Prebuilt Inference Models

The public release models for the C++ inference library are stored at:

```text
s3://graphcore-research-public/2026-llama-mobile/models/20260611/
```

They are `.sqt` files using the format documented in [`file_format.md`](file_format.md). The files contain model weights, quantized weight metadata, tokenizer data, chat-template metadata, and model configuration in a single file.

## Models

| File | Size | Source model | Weights | Intended use |
| --- | ---: | --- | --- | --- |
| `vision-11B-s3d8.sqt` | 3.73 GB | `meta-llama/Llama-3.2-11B-Vision-Instruct` | S3D8 weights / INT8 activations, QAT | Trained model supporting the paper's main task results |
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
