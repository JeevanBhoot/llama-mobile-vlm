# GPTQ Baselines

This directory contains post-training GPTQ baselines for
`meta-llama/Llama-3.2-11B-Vision-Instruct`.

| Flow | Module | Quantizer | Artifact |
| --- | --- | --- | --- |
| GPTQModel reference | `gptq.standard` | GPTQModel | GPTQModel checkpoint |
| Local dense GPTQ | `gptq.local` | Local PyTorch GPTQ | Dense dequantized Hugging Face checkpoint |

Both flows use C4 calibration and the `eval.vqa` benchmark harness. Mllama
quantization covers language-model self-attention and MLP projections.
Cross-attention decoder layers are skipped.

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
- `--resume`: append missing examples to existing JSONL output.
- `--overwrite`: rerun tasks with existing JSONL output.
- `--vqa-s3-path`, `--vqa-s3-local-path`: load VQAv2 from an explicit S3 dataset copy.

## Local GPTQ Quantize

`gptq.local` runs the local PyTorch GPTQ implementation and saves dense
dequantized weights in a Hugging Face checkpoint. Use
`estimated_packed_storage` in `metadata.json` for the estimated packed tensor
storage.

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

Local GPTQ-specific defaults:

- Bits: `3` or `4`
- Group size: `128`
- Block size: `128`
- Activation order: disabled
- Act-group-aware ordering: enabled
- Static groups: disabled
- Symmetric quantization: enabled
- Damp percent: `0.05`
- Damp auto increment: `0.01`
- Torch dtype: `bfloat16`
- Packed scale/zero storage estimate dtype: `bfloat16`

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

Local evaluation uses the same evaluation defaults, outputs,
and useful options as `gptq.standard evaluate`.

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

These are accuracy-only PTQ baselines. The generated checkpoints are not Arm
deployment artifacts for the C++ inference library.

Compare quality with `summary.json`. Compare storage with the GPTQModel
checkpoint size or the local `estimated_packed_storage` field.
