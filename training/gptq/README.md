# GPTQ

This directory provides a PyTorch GPTQ implementation for
`meta-llama/Llama-3.2-11B-Vision-Instruct`. It can target every
`Linear` weight matrix in the model, including those in the vision encoders,
multimodal projector, cross-attention layers, and `lm_head`. It supports
ordinary INT, codebook INT, and S3D8.

Saved Hugging Face checkpoints store dense BF16 weights that preserve the
quantization error. Evaluation also uses BF16 activations.

## Setup

Run from `training/`:

```sh
source .venv/bin/activate
uv pip install -r requirements.txt --torch-backend cu128
```

## Quantize a model

`python -m gptq quantize` runs GPTQ and saves a Hugging Face checkpoint.

### Weight formats

| Format | Arguments | Parameters |
| --- | --- | --- |
| Ordinary affine INT | `--format int --bits {2,3,4}` | Scale and, for asymmetric runs, zero point |
| Affine codebook INT | `--format int-codebook-affine --codepoints K` | Scale and, for asymmetric runs, zero point |
| Signed codebook INT | `--format int-codebook --codepoints K` | Absmax scale |
| S3D8 | `--format s3d8` | Per-output-channel scale and fitted `32 x 3` centroids |

The default is symmetric INT4 (`--format int --bits 4`).

Ordinary INT uses `--bits`, so its number of levels is a power of two. The
codebook formats set the number of levels directly with `--codepoints K` and
also support non-power-of-two sizes.

With `int-codebook`, codes represent signed integer values and use absmax
scaling. `int-codebook-affine` stores unsigned codes and reconstructs them
around a fixed zero point. For even `K`, affine scaling gives finer spacing
near zero, while absmax scaling represents both `-absmax` and `+absmax`
exactly.

### Text decoder quantization

The default scope, `text-self`, quantizes only the text decoder
(self-attention and MLP projections) using C4 calibration.

Run ordinary INT4:

```sh
python -m gptq quantize \
  --format int \
  --bits 4 \
  --output-dir out/gptq/llama-3.2-vision-gptq-int4-c4
```

To quantize to an eight-level affine codebook:

```sh
python -m gptq quantize \
  --format int-codebook-affine \
  --codepoints 8 \
  --group-size 128 \
  --output-dir out/gptq/llama-3.2-vision-gptq-affine-k8-c4
```

### Full-multimodal quantization

With `--target-scope full-multimodal`, GPTQ quantizes all supported `Linear`
matrices in the local and global vision encoders, multimodal projector, text
self-attention, cross-attention, MLPs, and `lm_head`. This scope supports
`vqav2`, `synthetic`, and `eval-task` calibration.

Run GPTQ-S3D8 with VQAv2 training data for calibration:

```sh
python -m gptq quantize \
  --format s3d8 \
  --target-scope full-multimodal \
  --calibration-source vqav2 \
  --calibration-split train \
  --output-dir out/gptq/llama-3.2-vision-gptq-s3d8-vqav2
```

VQAv2 (val subset) is also an evaluation task, so this command is mainly useful for testing
the workflow. Use a separate image-text calibration set for independent
experiments.

The original experiments used synthetic image-text data from the QAT data
pipeline. This dataset is not distributed with the repository, and the data
links in the [training README](../README.md) require project access. The
generation code is in [`train_data.py`](../train_data.py)
(`GenerationConfig` and `generate_data`) and
[`prompt_sampling.py`](../prompt_sampling.py). To use compatible data, pass
`--calibration-source synthetic` and `--calibration-data-path`.

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
quantization GPU. Otherwise, activation caches are streamed through CPU
memory.

## Evaluate a checkpoint

Evaluate a GPTQ checkpoint:

```sh
python -m gptq evaluate \
  out/gptq/llama-3.2-vision-gptq-int4-c4 \
  --output-dir out/gptq/llama-3.2-vision-gptq-int4-c4/evaluation
```

By default, evaluation uses:

- tasks: `vqa chartqa docvqa ai2d`;
- 1024 examples per task;
- batch size 1;
- CUDA with BF16 activations;

Evaluation writes `summary.json`, `summary.partial.json` while a run is in
progress, and one JSONL file per task. Control the run with `--tasks`,
`--n-examples`, `--batch-size`, `--resume`, and `--overwrite`.

## Artifacts and storage

| Artifact | Purpose |
| --- | --- |
| Hugging Face checkpoint | Dense BF16 values preserving quantization error; input to `python -m gptq evaluate` |
| `metadata.json` | Quantization settings, module log, and packed-storage estimates |
| `gptq-s3d8.safetensors` | S3D8 master and format tensors, including fitted scales and centroids |
| `.sqt` | Packed deployment artifact produced separately by `squashedtensors.py` |

For codebook formats, `effective_weight_bits` is the ideal rate `log2(K)`.
