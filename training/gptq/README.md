# Standard GPTQ Baselines

This directory contains GPTQModel INT4 and INT3 baselines for
`meta-llama/Llama-3.2-11B-Vision-Instruct`.

The script quantizes with C4 calibration and evaluates with the existing
`eval.vqa` benchmark harness.

## Prerequisites

From `training/`:

```sh
source .venv/bin/activate
uv pip install -r requirements.txt --torch-backend cu130
uv pip install setuptools wheel
uv pip install --no-build-isolation-package gptqmodel -r gptq/requirements.txt
```

Set up a Hugging Face token with access to
`meta-llama/Llama-3.2-11B-Vision-Instruct`.

## Quantize

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

Each command writes a GPTQModel artifact to `--output-dir` and records run
metadata in `metadata.json`.

Quantization defaults:

- Model: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- Calibration dataset: `allenai/c4`
- Calibration file: `en/c4-train.00001-of-01024.json.gz`
- Calibration samples: `1024`
- Calibration max tokens per sample: `2048`
- Group size: `128`
- Quantization batch size: `1`
- Calibration GPU cache: disabled
- Torch dtype: `bfloat16`

The calibration batch size is already `1`. For lower memory use, reduce the
calibration sequence length:

```sh
python -m gptq.standard quantize \
  --bits 4 \
  --output-dir out/gptq/llama-3.2-vision-gptq-int4-c4 \
  --calibration-max-tokens 1024
```

If quantization still runs out of memory, add `--buffered-fwd`.

## Evaluate

Evaluate the INT4 artifact:

```sh
python -m gptq.standard evaluate \
  out/gptq/llama-3.2-vision-gptq-int4-c4 \
  --output-dir out/gptq/llama-3.2-vision-gptq-int4-c4/evaluation \
  --tasks vqa chartqa docvqa ai2d \
  --n-examples 1024
```

Evaluate the INT3 artifact:

```sh
python -m gptq.standard evaluate \
  out/gptq/llama-3.2-vision-gptq-int3-c4 \
  --output-dir out/gptq/llama-3.2-vision-gptq-int3-c4/evaluation \
  --tasks vqa chartqa docvqa ai2d \
  --n-examples 1024
```

The evaluator reuses `eval.vqa`, so prompts, splits, fixed subsets,
generation lengths, and metrics match the existing S3D8/QAT evaluation
path.

Evaluation writes:

- `summary.json`
- One JSONL file per task, for example `vqa.jsonl`

Evaluation defaults:

- Tasks: `vqa chartqa docvqa ai2d`
- Examples per task: `1024`
- Evaluation batch size: `1`
- Primary summary metric: `avg_primary`

## Notes

This is an accuracy-only PTQ baseline. The saved GPTQModel artifact is not
an Arm deployment artifact for the C++ inference library.

GPTQModel 2.0.0 Mllama quantization skips cross-attention layers that require
`pixel_values`.
