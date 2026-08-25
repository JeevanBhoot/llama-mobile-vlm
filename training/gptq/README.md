# GPTQ Baselines

This directory provides two GPTQ workflows for
`meta-llama/Llama-3.2-11B-Vision-Instruct`.

## Choose a workflow

| Workflow | Command | Quantization scope | Output |
| --- | --- | --- | --- |
| GPTQModel | `gptq.standard` | Language model with C4 calibration | Packed GPTQModel checkpoint |
| Local PyTorch | `gptq.local` | Text layers or full Mllama Linear coverage | Dense dequantized Hugging Face checkpoint |

Use `gptq.standard` to create and evaluate a GPTQModel checkpoint. Use
`gptq.local` to experiment with ordinary INT, codebook INT, and S3D8 inside the
GPTQ algorithm. Both workflows use the `eval.vqa` evaluation harness.

## Set up the environment

Run from `training/`:

```sh
source .venv/bin/activate
uv pip install -r requirements.txt --torch-backend cu130
```

Install GPTQModel to use `gptq.standard` and run GPTQModel parity tests:

```sh
uv pip install setuptools wheel
uv pip install --no-build-isolation-package gptqmodel -r gptq/requirements.txt
```

Configure a Hugging Face token with access to
`meta-llama/Llama-3.2-11B-Vision-Instruct`.

## Quantize with GPTQModel

Create an INT4 GPTQModel checkpoint:

```sh
python -m gptq.standard quantize \
  --bits 4 \
  --output-dir out/gptq/llama-3.2-vision-gptq-int4-c4
```

`--bits` accepts `2`, `3`, `4`, `5`, `6`, or `8`. To quantize another model,
set `--model-name` to a Hugging Face model name or local checkpoint path.

The default calibration settings are:

- Dataset: `allenai/c4`, file `en/c4-train.00001-of-01024.json.gz`
- Samples: `512`
- Maximum tokens per sample: `1024`
- Sample order: descending length
- Minimum sample length: `10` tokens
- Calibration batch size: `1`
- Group size: `128`
- Checkpoint format: `gptq`
- Model load dtype: `bfloat16`

Common options:

- `--calibration-samples`, `--calibration-max-tokens`: set calibration size.
- `--calibration-concat-size`: concatenate tokens into fixed-size chunks.
- `--group-size`, `--sym`, `--desc-act`, `--static-groups`, `--mse`: configure
  quantization.
- `--format`, `--pack-dtype`: configure the saved GPTQModel checkpoint.
- `--offload-to-disk-path`, `--gc-mode`, `--wait-for-submodule-finalizers`:
  control memory use during quantization.

The output directory contains the GPTQModel checkpoint and `metadata.json`.

## Quantize with the local implementation

`gptq.local` applies quantization during the GPTQ column updates, then saves the
dequantized weights as a Hugging Face checkpoint. These checkpoints are used for
accuracy evaluation. `metadata.json` records the quantization settings and
estimated packed storage.

### Choose a weight format

| Format | Arguments | Quantization parameters |
| --- | --- | --- |
| Ordinary affine INT | `--format int --bits {2,3,4}` | Scale; asymmetric runs also store a zero point |
| Affine codebook INT | `--format int-codebook-affine --codepoints K` | Scale; asymmetric runs also store a zero point |
| Signed-centroid codebook INT | `--format int-codebook --codepoints K` | Scale |
| S3D8 | `--format s3d8` | Per-row scale and fitted centroids |

The two codebook modes work as follows:

- `int-codebook-affine` uses unsigned indices from `0` to `K - 1`. Symmetric
  quantization uses `scale = 2 * absmax / (K - 1)` and zero point
  `floor(K / 2)`. Asymmetric quantization derives scale and zero point from the
  row minimum and maximum. Under identical GPTQ settings and parameter dtype,
  `K=4`, `K=8`, and `K=16` match ordinary INT2, INT3, and INT4.
- `int-codebook` uses signed integer centroids that include zero and sets the
  per-row scale from `absmax / positive_endpoint`. For example, `K=6` uses
  `[-3, -2, -1, 0, 1, 2]`, so the scale denominator is `2`. `K=16` uses
  `[-8, ..., 7]`, so the scale denominator is `7`. No zero point is stored.

Use `--codepoints` for either codebook mode; do not combine it with `--bits`.
The intended affine-codebook sweep is `K=4..16`.
Packed-storage estimates treat the symmetric zero point as implicit because it
is determined by `K`. Runs using `--no-sym` include one stored zero point per
group.

### Quantize text layers

The default target is `--target-scope text-self` with C4 calibration. It
quantizes language-model self-attention and MLP projections.

Ordinary INT4:

```sh
python -m gptq.local quantize \
  --bits 4 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-int4-c4
```

Affine codebook with seven codepoints:

```sh
python -m gptq.local quantize \
  --format int-codebook-affine \
  --codepoints 7 \
  --group-size 128 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-int-affine-k7-g128-c4
```

Signed-centroid codebook with seven codepoints:

```sh
python -m gptq.local quantize \
  --format int-codebook \
  --codepoints 7 \
  --group-size 128 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-int-k7-g128-c4
```

S3D8:

```sh
python -m gptq.local quantize \
  --format s3d8 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-s3d8-c4
```

S3D8 also writes `gptq-s3d8.safetensors`. Pass this file to
`squashedtensors.py --checkpoint` for the S3D8 deployment path.

### Quantize the full multimodal model

Use `--target-scope full-multimodal` with `vqav2`, `synthetic`, or `eval-task`
calibration:

```sh
python -m gptq.local quantize \
  --target-scope full-multimodal \
  --calibration-source vqav2 \
  --bits 4 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-int4-vqav2-full
```

This target includes:

- Vision encoder and global-encoder Linear layers
- Text self-attention, cross-attention, and MLP Linear layers
- `model.multi_modal_projector`
- `lm_head`

VQAv2 calibration defaults to the `train` split from
`Multimodal-Fatima/VQAv2_sample_train`. Set `--calibration-split validation` to
use the validation split.

For multimodal runs:

- Keep `--batch-size 1` when image and text padding would exceed GPU memory.
- Enable `--calibration-gpu-cache` when the captured activations fit on the
  quantization GPU.
- Use `--calibration-max-tokens` to control text length.
- Repeat `--calibration-data-path` to provide synthetic calibration shards.

### Configure local GPTQ

The local defaults are:

- Calibration samples: `512`
- Maximum tokens per sample: `1024`
- C4 sample order: descending length
- C4 minimum sample length: `10` tokens
- Calibration batch size: `1`
- Group size: `128`
- GPTQ block size: `128`
- Activation order: disabled
- Act-group-aware ordering: enabled
- Symmetric quantization: enabled
- MSE shrink search: disabled
- Damp percent: `0.05`
- Damp auto increment: `0.01`
- Model dtype: `bfloat16`
- Scale and computed zero-point dtype: `bfloat16`

Set `--storage-scale-zero-dtype` to `bfloat16`, `float16`, or `float32`. This
dtype is applied to scales and computed zero points before GPTQ quantization.
Storage estimates use it for scales and, with `--no-sym`, stored zero points.

## Inspect outputs and storage

| Workflow | Primary artifact | Metadata |
| --- | --- | --- |
| GPTQModel | Packed checkpoint in `--output-dir` | `metadata.json` |
| Local INT/codebook | Dense dequantized Hugging Face checkpoint | `metadata.json` |
| Local S3D8 | Dense checkpoint and `gptq-s3d8.safetensors` | `metadata.json` |

Inspect a local storage estimate with:

```sh
jq .estimated_packed_storage \
  out/gptq/llama-3.2-vision-local-gptq-int4-c4/metadata.json
```

For codebook formats, `effective_weight_bits` is the ideal information rate
`log2(K)`. The storage metadata also reports a realizable byte-aligned radix
packing. For example, `K=6` packs three indices per byte and reports `8/3`
realizable weight bits. Full-model estimates include scales, stored zero points
for asymmetric affine formats, group indices, per-tensor tail chunks, and
unquantized tensors.

The saved local Hugging Face checkpoint contains dense dequantized weights. Use
`estimated_packed_storage` to compare quantization formats independently of the
dense checkpoint size.

## Evaluate a checkpoint

Evaluate a GPTQModel checkpoint with `gptq.standard`:

```sh
python -m gptq.standard evaluate \
  out/gptq/llama-3.2-vision-gptq-int4-c4 \
  --output-dir out/gptq/llama-3.2-vision-gptq-int4-c4/evaluation
```

Evaluate a local dense checkpoint with `gptq.local`:

```sh
python -m gptq.local evaluate \
  out/gptq/llama-3.2-vision-local-gptq-int4-c4 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-int4-c4/evaluation
```

Evaluation defaults:

- Tasks: `vqa chartqa docvqa ai2d`
- Examples per task: `1024`
- Batch size: `1`
- Device: `cuda`
- VQAv2 source: Hugging Face

Evaluation writes:

- `summary.json`
- `summary.partial.json` while a run is in progress
- One streamed JSONL file per task, such as `vqa.jsonl`

Use `--tasks`, `--n-examples`, `--batch-size`, `--resume`, and `--overwrite` to
control a run. The GPTQModel evaluator accepts `--backend` and `--dtype`; the
local evaluator accepts `--torch-dtype`.

## Command reference

```sh
python -m gptq.standard quantize --help
python -m gptq.standard evaluate --help
python -m gptq.local quantize --help
python -m gptq.local evaluate --help
```

## Run tests

GPTQModel parity tests require the optional GPTQModel installation:

```sh
pytest \
  tests/test_gptq_algorithm.py \
  tests/test_gptq_algorithm_parity.py \
  tests/test_gptq_local.py \
  tests/test_gptq_standard.py
```
