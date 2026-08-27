# GPTQ

This directory provides a PyTorch GPTQ implementation for
`meta-llama/Llama-3.2-11B-Vision-Instruct`. It can target every
`Linear` weight matrix in the model, including vision encoders,
multimodal projector, cross-attention, and `lm_head`. The implementation
supports ordinary INT, codebook INT, and S3D8.

Saved Hugging Face checkpoints contain dense BF16 values that
preserve quantization error, and evaluation uses BF16 activations.

## Setup

Run from `training/``:

```sh
source .venv/bin/activate
uv pip install -r requirements.txt --torch-backend cu130
```

## Quantize a model

`python -m gptq quantize` runs GPTQ and saves the resulting dense BF16
values as a Hugging Face checkpoint.
`metadata.json` records the settings and estimated packed storage.

### Weight formats

| Format | Arguments | Parameters |
| --- | --- | --- |
| Ordinary affine INT | `--format int --bits {2,3,4}` | Scale and, for asymmetric runs, zero point |
| Affine codebook INT | `--format int-codebook-affine --codepoints K` | Scale and, for asymmetric runs, zero point |
| Signed codebook INT | `--format int-codebook --codepoints K` | Absmax scale |
| S3D8 | `--format s3d8` | Per-output-channel scale and fitted `32 x 3` centroids |

The default is symmetric INT4 (`--format int --bits 4`).

Ordinary INT selects a power-of-two number of levels with `--bits`. The
codebook formats set the number of levels directly with `--codepoints K`, so
they also support non-power-of-two sizes.

`int-codebook` uses signed integer values with absmax scaling.
`int-codebook-affine` stores unsigned indices and reconstructs them around a
fixed zero point. For even `K`, affine scaling gives finer spacing near zero,
while absmax scaling represents both `-absmax` and `+absmax` exactly.

### Text decoder quantization

The default scope is `text-self`, which only quantizes the text decoder
(self-attention and MLP projections) using C4 calibration.

Run ordinary INT4:

```sh
python -m gptq quantize \
  --format int \
  --bits 4 \
  --output-dir out/gptq/llama-3.2-vision-gptq-int4-c4
```

Quantize to an eight-level affine codebook with:

```sh
python -m gptq quantize \
  --format int-codebook-affine \
  --codepoints 8 \
  --group-size 128 \
  --output-dir out/gptq/llama-3.2-vision-gptq-affine-k8-c4
```

### Full-multimodal quantization

Set `--target-scope full-multimodal` to quantize all supported `Linear`
matrices in the local and global vision encoders, multimodal projector, text
self-attention, cross-attention, MLPs, and `lm_head`. This scope supports
`vqav2`, `synthetic`, and `eval-task` calibration.

Run S3D8 with VQAv2 training data from Hugging Face:

```sh
python -m gptq quantize \
  --format s3d8 \
  --target-scope full-multimodal \
  --calibration-source vqav2 \
  --calibration-split train \
  --output-dir out/gptq/llama-3.2-vision-gptq-s3d8-vqav2
```

VQAv2 is also an evaluation task, so this example is best used to try the
workflow. Use a separate image-text calibration set for independent
experiments.

The original experiments used synthetic image-text data generated through the
QAT data pipeline. It is not distributed with the repository, and the
original data links in the [training README](../README.md) require project
access. The generation logic is in [`train_data.py`](../train_data.py)
(`GenerationConfig` and `generate_data`) and
[`prompt_sampling.py`](../prompt_sampling.py). If you prepare compatible
data, use `--calibration-source synthetic` and `--calibration-data-path`.

### Common options

| Option | Default | Purpose |
| --- | ---: | --- |
| `--calibration-samples` | `512` | Number of calibration examples |
| `--calibration-max-tokens` | `1024` | Maximum tokens per example |
| `--batch-size` | `1` | Calibration batch size |
| `--group-size` | `128` | Weights per quantization group |
| `--blocksize` | `128` | GPTQ input-column block size |
| `--damp-percent` | `0.05` | Initial Hessian damping |
| `--damp-auto-increment` | `0.01` | Damping increment after factorization failure |
| `--act-group-aware` | enabled | Reorder groups using activation statistics |
| `--desc-act` | disabled | Process columns in descending activation order |
| `--sym` | enabled | Use symmetric quantization |
| `--mse` | `0.0` | MSE shrink search; `0.0` disables it |
| `--torch-dtype` | `bfloat16` | Model and saved-weight dtype |

Use `--calibration-gpu-cache` when the captured activations fit on the
quantization GPU. Otherwise, the flow streams activation caches through CPU
memory.

## Evaluate a checkpoint

Evaluate a saved Hugging Face checkpoint:

```sh
python -m gptq evaluate \
  out/gptq/llama-3.2-vision-gptq-int4-c4 \
  --output-dir out/gptq/llama-3.2-vision-gptq-int4-c4/evaluation
```

Evaluation defaults to:

- tasks: `vqa chartqa docvqa ai2d`;
- 1024 examples per task;
- batch size 1;
- CUDA with BF16 activations;
- the Hugging Face VQAv2 validation split.

Evaluation writes `summary.json`, `summary.partial.json` while a run is in
progress, and one JSONL file per task. Use `--tasks`, `--n-examples`,
`--batch-size`, `--resume`, and `--overwrite` to control the run.

## Artifacts and storage

| Artifact | Purpose |
| --- | --- |
| Hugging Face checkpoint | Dense BF16 values preserving quantization error; input to `python -m gptq evaluate` |
| `metadata.json` | Quantization settings, module log, and packed-storage estimates |
| `gptq-s3d8.safetensors` | S3D8 master and format tensors, including fitted scales and centroids |
| `.sqt` | Packed deployment artifact produced separately by `squashedtensors.py` |

For codebook formats, `effective_weight_bits` is the ideal rate `log2(K)`.
