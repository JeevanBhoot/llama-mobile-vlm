# GPTQ Baselines

This directory contains two GPTQ flows for
`meta-llama/Llama-3.2-11B-Vision-Instruct`.

| Flow | Module | Quantization implementation | Saved artifact | Primary use |
| --- | --- | --- | --- | --- |
| GPTQModel reference | `gptq.standard` | GPTQModel | GPTQModel artifact | Run the external GPTQModel baseline. |
| Local dense GPTQ | `gptq.local` | Local PyTorch GPTQ implementation | Dense dequantized Hugging Face model | Evaluate GPTQ quality with standard Transformers loading and report estimated packed tensor storage. |

Both flows use C4 calibration and the `eval.vqa` benchmark harness. GPTQModel's
Mllama definition quantizes language-model self-attention and MLP projections.
Cross-attention layers that require `pixel_values` are skipped.

## Prerequisites

From `training/`:

```sh
source .venv/bin/activate
uv pip install -r requirements.txt --torch-backend cu130
```

Install GPTQModel for the `gptq.standard` flow:

```sh
uv pip install setuptools wheel
uv pip install --no-build-isolation-package gptqmodel -r gptq/requirements.txt
```

Set up a Hugging Face token with access to
`meta-llama/Llama-3.2-11B-Vision-Instruct`.

## GPTQModel Reference Quantize

`gptq.standard` calls GPTQModel directly. It writes the artifact format produced
by GPTQModel and records run metadata in `metadata.json`.

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

Default quantization settings:

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

Use these options to reduce quantization memory:

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

Common algorithm options:

- `--group-size`: number of input columns per quantization group.
- `--format`: GPTQModel output format.
- `--pack-dtype`: packed weight dtype passed to GPTQModel.
- `--desc-act` / `--no-desc-act`: activation-order quantization.
- `--act-group-aware` / `--no-act-group-aware`: GPTQModel act-group-aware ordering.
- `--static-groups`: precompute group quantizers before activation ordering.
- `--damp-percent`: Hessian damping percentage.
- `--damp-auto-increment`: damping increment used during Cholesky recovery.
- `--mse`: MSE grid-search strength for quantizer scale selection.

## GPTQModel Reference Evaluate

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

The evaluator uses `eval.vqa`, so prompts, splits, fixed subsets, generation
lengths, and metrics are shared with the existing VQA evaluation code.

Evaluation writes:

- `summary.json`
- One JSONL file per task, for example `vqa.jsonl`

Evaluation resumes from existing task JSONL files in `--output-dir`. If a task
JSONL already contains records for the requested example ids, those examples are
used for the summary and skipped during generation. If a task JSONL is partial,
evaluation appends the missing records as they are generated.

Evaluation defaults:

- Tasks: `vqa chartqa docvqa ai2d`
- Examples per task: `1024`
- Evaluation batch size: `1`
- Primary summary metric: `avg_primary`

## Local GPTQ Quantize

`gptq.local` runs the local PyTorch GPTQ implementation and saves a dense
dequantized Hugging Face model. The quantized values are dequantized back into
the model weights before saving, so the artifact can be loaded with standard
Transformers APIs.

The saved model is dense. Use `estimated_packed_storage` in `metadata.json` for
the estimated packed tensor storage. The estimate counts quantized target
weights as packed `bits` values, scale/zero tensors using
`--storage-scale-zero-dtype`, optional `g_idx` storage separately, and all
unquantized tensors at their current dense dtype. Tokenizer files, config files,
and container overhead are excluded from the estimate.

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

The local flow quantizes the same Mllama language-model self-attention and MLP
projection layers as the GPTQModel Mllama definition. Cross-attention decoder
layers are skipped.

Local GPTQ defaults:

- Model: `meta-llama/Llama-3.2-11B-Vision-Instruct`
- Calibration dataset: `allenai/c4`
- Calibration file: `en/c4-train.00001-of-01024.json.gz`
- Calibration samples: `1024`
- Calibration max tokens per sample: `2048`
- Bits: `3` or `4`
- Group size: `128`
- Block size: `128`
- Activation order: disabled
- Act-group-aware activation ordering: enabled
- Static groups: disabled
- Symmetric quantization: enabled
- Damp percent: `0.05`
- Damp auto increment: `0.01`
- Torch dtype: `bfloat16`
- Packed storage scale/zero dtype assumption: `bfloat16`

After quantization, the command prints:

```text
Estimated packed tensor storage (without g_idx): ...
Estimated packed tensor storage (with g_idx): ...
Dense local artifact size: ...
```

The same information is written to:

```sh
jq .estimated_packed_storage out/gptq/llama-3.2-vision-local-gptq-int4-c4/metadata.json
```

## Local GPTQ Evaluate

Evaluate the local INT4 artifact:

```sh
python -m gptq.local evaluate \
  out/gptq/llama-3.2-vision-local-gptq-int4-c4 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-int4-c4/evaluation \
  --tasks vqa chartqa docvqa ai2d \
  --n-examples 1024
```

Evaluate the local INT3 artifact:

```sh
python -m gptq.local evaluate \
  out/gptq/llama-3.2-vision-local-gptq-int3-c4 \
  --output-dir out/gptq/llama-3.2-vision-local-gptq-int3-c4/evaluation \
  --tasks vqa chartqa docvqa ai2d \
  --n-examples 1024
```

## Notes

These are accuracy-only PTQ baselines. GPTQModel artifacts and local dense
artifacts are not Arm deployment artifacts for the C++ inference library.

The GPTQModel artifact size and the local dense artifact size are different
because the two flows save different artifact formats. Compare quality using the
evaluation summaries. Compare storage using the GPTQModel artifact size or the
local `estimated_packed_storage` field.
