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

To quantize a different Hugging Face model, set `--model-name`:

```sh
python -m gptq.standard quantize \
  --bits 4 \
  --model-name meta-llama/Llama-3.2-3B \
  --output-dir out/gptq/llama-3.2-3b-gptq-int4-c4
```

Quantization defaults:

- Model: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- Calibration dataset: `allenai/c4`
- Calibration file: `en/c4-train.00001-of-01024.json.gz`
- Calibration samples: `512`
- Calibration max tokens per sample: `1024`
- Group size: `128`
- Format: `gptq`
- Quantization batch size: `1`
- Load dtype: `bfloat16`
- Activation-order setting: GPTQModel default, or override with
  `--desc-act` / `--no-desc-act`
- Static groups: disabled unless `--static-groups` is set
- Offload completed modules to disk: enabled by GPTQModel default
- GC mode: `interval`
- Calibration sort: `desc`

The calibration batch size is already `1`. For lower memory use, use one or
more of these options:

- Reduce `--calibration-samples`.
- Reduce `--calibration-max-tokens`.
- Keep `--offload-to-disk` enabled, or set `--offload-to-disk-path` to control
  where GPTQModel stores completed module state.
- Use `--gc-mode on_stage_end` to run GPTQModel garbage collection after each
  major quantization stage.
- Use `--wait-for-submodule-finalizers` if packing/offload finalization creates
  too much transient memory pressure.
- Use `--calibration-concat-size` to pack short calibration examples into fixed
  token chunks when that is appropriate for the baseline being run.

```sh
python -m gptq.standard quantize \
  --bits 4 \
  --output-dir out/gptq/llama-3.2-vision-gptq-int4-c4 \
  --calibration-samples 256 \
  --calibration-max-tokens 512 \
  --gc-mode on_stage_end \
  --wait-for-submodule-finalizers
```

The quantize command intentionally exposes only the practical GPTQ knobs for
paper baselines: number format, core GPTQ algorithm settings, calibration size,
device/backend, and memory controls. Advanced GPTQModel features are left at
their GPTQModel defaults so runs stay easy to reproduce.

Useful non-default algorithm knobs:

- `--group-size`
- `--format`
- `--pack-dtype`
- `--desc-act` / `--no-desc-act`
- `--act-group-aware` / `--no-act-group-aware`
- `--static-groups`
- `--damp-percent`
- `--damp-auto-increment`
- `--mse`

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

GPTQModel Mllama quantization skips cross-attention layers that require
`pixel_values`.
