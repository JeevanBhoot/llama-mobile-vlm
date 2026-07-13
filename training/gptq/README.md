# GPTQ Baselines

This directory contains post-training GPTQ baselines for
`meta-llama/Llama-3.2-11B-Vision-Instruct`.

| Flow | Module | Quantizer | Artifact |
| --- | --- | --- | --- |
| GPTQModel reference | `gptq.standard` | GPTQModel | GPTQModel checkpoint |
| Local dense GPTQ | `gptq.local` | Local PyTorch GPTQ | Dense dequantized Hugging Face checkpoint |

Both flows support C4 text calibration and the `eval.vqa` benchmark harness.
The local flow also supports multimodal calibration and full Mllama Linear
coverage.

For full CLI options:

```sh
python -m gptq.standard quantize --help
python -m gptq.standard evaluate --help
python -m gptq.local quantize --help
python -m gptq.local evaluate --help
```

## Setup

From `training/`:

```sh
source .venv/bin/activate
uv pip install -r requirements.txt --torch-backend cu130
```

Install GPTQModel for the `gptq.standard` flow and GPTQModel parity tests:

```sh
uv pip install setuptools wheel
uv pip install --no-build-isolation-package gptqmodel -r gptq/requirements.txt
```

Set up a Hugging Face token with access to
`meta-llama/Llama-3.2-11B-Vision-Instruct`.

## GPTQModel Quantize

`gptq.standard` calls GPTQModel and writes a GPTQModel checkpoint plus
`metadata.json`.

INT4:

```sh
python -m gptq.standard quantize \
  --bits 4 \
  --output-dir out/gptq/llama-3.2-vision-gptq-int4-c4
```

INT3:

```sh
python -m gptq.standard quantize \
  --bits 3 \
  --output-dir out/gptq/llama-3.2-vision-gptq-int3-c4
```

Quantize a different model:

```sh
python -m gptq.standard quantize \
  --model-name /path/to/model \
  --bits 4 \
  --output-dir out/gptq/custom-model-gptq-int4-c4
```

Default quantization settings:

- Model: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- Calibration: `allenai/c4`, `en/c4-train.00001-of-01024.json.gz`
- Calibration samples: `512`
- Calibration max tokens: `1024`
- Calibration sort: `desc`
- Calibration min token length: `10`
- Calibration batch size: `1`
- Group size: `128`
- Format: `gptq`
- Load dtype: `bfloat16`

Useful options:

- `--calibration-samples`, `--calibration-max-tokens`: reduce calibration size.
- `--calibration-concat-size`: concatenate calibration tokens into fixed-size chunks.
- `--offload-to-disk-path`, `--gc-mode`, `--wait-for-submodule-finalizers`: control GPTQModel memory behavior.
- `--group-size`, `--desc-act`, `--act-group-aware`, `--static-groups`, `--damp-percent`, `--damp-auto-increment`, `--mse`: GPTQ algorithm settings.

## GPTQModel Evaluate

Evaluate the quantized checkpoint:

```sh
python -m gptq.standard evaluate \
  out/gptq/llama-3.2-vision-gptq-int4-c4 \
  --output-dir out/gptq/llama-3.2-vision-gptq-int4-c4/evaluation
```

The evaluator reuses `eval.vqa`, so prompts, splits, fixed subsets,
generation lengths, and metrics match the existing S3D8/QAT evaluation path.

Evaluation defaults:

- Tasks: `vqa chartqa docvqa ai2d`
- Examples per task: `1024`
- Batch size: `1`
- Device: `cuda`
- Processor: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- VQAv2 source: Hugging Face

Evaluation writes:

- `summary.json`
- `summary.partial.json`
- One streamed JSONL file per task, for example `vqa.jsonl`

Useful options:

- `--tasks`: choose evaluation tasks.
- `--n-examples`: set the per-task example count.
- `--batch-size`: set evaluation batch size.
- `--device`: set the evaluation device.
- `--processor-name`: use a different processor path.
- `--backend`, `--dtype`: control GPTQModel loading for evaluation.
- `--resume`: append missing examples to existing JSONL output.
- `--overwrite`: rerun tasks with existing JSONL output.
- `--vqa-s3-path`, `--vqa-s3-local-path`: load VQAv2 from an explicit S3 dataset copy.

## Local GPTQ Quantize

`gptq.local` runs the local PyTorch GPTQ implementation and saves dense
dequantized weights in a Hugging Face checkpoint. Use
`estimated_packed_storage` in `metadata.json` for the estimated packed tensor
storage.

By default, local GPTQ keeps the original decoder-only C4 path:
`--target-scope text-self` and `--calibration-source c4`. This quantizes
language-model self-attention and MLP projections, while skipping
cross-attention layers.

INT4:

```sh
python -m gptq.local quantize \
  --bits 4 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-int4-c4
```

INT3:

```sh
python -m gptq.local quantize \
  --bits 3 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-int3-c4
```

S3D8:

```sh
python -m gptq.local quantize \
  --format s3d8 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-s3d8-c4
```

S3D8 mode uses S3D8 inside the GPTQ column update and also writes
`gptq-s3d8.safetensors`, which can be passed to `squashedtensors.py
--checkpoint`.

### Full Multimodal

To quantize all layers, run:

```sh
python -m gptq.local quantize \
  --target-scope full-multimodal \
  --calibration-source vqav2 \
  --bits 4 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-int4-vqav2-full
```

The full multimodal path quantizes vision encoder and global-encoder Linear
layers, text self-attention layers, text cross-attention layers,
`model.multi_modal_projector`, and `lm_head`.

VQAv2 calibration defaults to the `train` split from
`Multimodal-Fatima/VQAv2_sample_train`. 
VQAv2 evaluation continues to use the `validation` split from `lmms-lab/VQAv2`.
To use the `validation` split for calibration, pass `--calibration-split validation`.

For large multimodal runs:

- Keep the default `--batch-size 1`. Larger batches increase transient GPU
  memory and image/text padding.
- Keep `--calibration-gpu-cache` disabled when the calibration cache is larger
  than available GPU memory.
- `--calibration-max-tokens` controls text length. It does not reduce the fixed
  per-image vision token count.

### Fractional-width affine INT

Use `int-codebook-affine` for fractional-width INT experiments:

```sh
python -m gptq.local quantize \
  --format int-codebook-affine \
  --codepoints 7 \
  --group-size 128 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-int-affine-k7-g128-c4
```

The affine codebook uses unsigned indices from `0` through `K - 1`. Symmetric
quantization uses `scale = 2 * absmax / (K - 1)` and zero point `floor(K / 2)`.
Asymmetric quantization derives the scale and zero point from each row's minimum
and maximum. Under the same GPTQ settings and scale dtype, `K=4`, `K=8`, and
`K=16` reproduce ordinary INT2, INT3, and INT4. The intended fractional sweep
is `K=4..16`.

Run that sweep with:

```sh
for k in 4 5 6 7 8 9 10 11 12 13 14 15 16; do
  for g in 32 64 128; do
    python -m gptq.local quantize \
      --format int-codebook-affine \
      --codepoints "$k" \
      --group-size "$g" \
      --output-dir "out/gptq/local-int-affine-k${k}-g${g}"
  done
done
```

The effective information rate is `log2(K)`. Metadata also reports a realizable
byte-aligned radix packing selected from one- through eight-byte chunks. For
example, `K=6` packs three indices per byte, or `8/3` realizable weight bits.
The full-model estimate includes each tensor's final partial chunk, scales, zero
points, group indices, and tensors that remain unquantized.

### Legacy safe-absmax INT codebook

Use `int-codebook` to reproduce the legacy scale-only safe-absmax quantizer:

```sh
python -m gptq.local quantize \
  --format int-codebook \
  --codepoints 7 \
  --group-size 128 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-int-k7-g128-c4
```

This format keeps its asymmetric signed centroids and positive-endpoint absmax
denominator. For example, `K=6` uses `[-3, -2, -1, 0, 1, 2]`; `K=16` uses
`[-8, ..., 7]` with scale denominator `7`. Even K therefore leaves part of the
negative endpoint range unused for symmetric rows.

Quantize a different model:

```sh
python -m gptq.local quantize \
  --model-name /path/to/model \
  --bits 4 \
  --output-dir out/gptq/custom-model-local-gptq-int4-c4
```

Local GPTQ uses the same calibration defaults as `gptq.standard`:

- Calibration samples: `512`
- Calibration max tokens: `1024`
- Calibration sort: `desc`
- Calibration min token length: `10`
- Calibration batch size: `1`
- Target scope: `text-self`
- Calibration source: `c4`

Local GPTQ-specific defaults:

- Bits: `2`, `3`, or `4` for ordinary affine INT; arbitrary `K >= 2` for
  either codebook format (`4` to `16` is the intended affine-K sweep)
- Group size: `128`
- Block size: `128`
- Activation order: disabled
- Act-group-aware ordering: enabled
- Static groups: disabled
- Symmetric quantization: enabled
- Damp percent: `0.05`
- Damp auto increment: `0.01`
- Torch dtype: `bfloat16`
- Quantizer scale/zero dtype: `bfloat16`; select `float16` or `float32` with
  `--storage-scale-zero-dtype`. This dtype also controls parameter storage
  estimates.

After quantization, inspect the packed storage estimate:

```sh
jq .estimated_packed_storage out/gptq/llama-3.2-vision-local-gptq-int4-c4/metadata.json
```

## Local GPTQ Evaluate

Evaluate the quantized checkpoint:

```sh
python -m gptq.local evaluate \
  out/gptq/llama-3.2-vision-local-gptq-int4-c4 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-int4-c4/evaluation
```

Local evaluation uses the same evaluation defaults, outputs, resume/overwrite
behavior, and dataset options as `gptq.standard evaluate`. Use `--torch-dtype`
to control the dense Hugging Face checkpoint load dtype.

## Tests

GPTQModel parity tests are marked `gptqmodel` and skip when GPTQModel is not
installed:

```sh
pytest \
  tests/test_gptq_algorithm.py \
  tests/test_gptq_algorithm_parity.py \
  tests/test_gptq_local.py \
  tests/test_gptq_standard.py
```

## Notes

INT2/INT3/INT4 and codebook local GPTQ checkpoints are accuracy-only dense PTQ
baselines. Local S3D8 GPTQ also writes a quantisation checkpoint for the existing
S3D8
`squashedtensors.py` deployment path.

Compare quality with `summary.json`. Compare storage with the GPTQModel
checkpoint size or the local `estimated_packed_storage` field.
