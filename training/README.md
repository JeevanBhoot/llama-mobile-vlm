# Llama-Mobile Training Reference

This directory contains the quantization-aware training, evaluation, and model
conversion code used during the Llama-Mobile project.

It is provided for transparency and provenance. It is not intended to be a
supported external training pipeline: the original workflow depends on internal
datasets, checkpoints, and cluster infrastructure.

For runnable public artifacts, see the [parent README](../README.md).

## Structure

- `train.py` — quantization-aware training and checkpoint generation.
- `train_data.py` and `prompt_sampling.py` — dataset preparation, prompt
  generation, and sampling configuration.
- `direct_cast.py` — directly quantizes Hugging Face Llama weights.
- `compress_s3d8_checkpoint.py` — packs trained S3D8 checkpoints for inference.
- `squashedtensors.py` — reads and writes the project's `.sqt` model format.
- `eval/` — VQA metrics and reference-output comparisons.
- `vqa_interactive.py` — notebook-oriented utilities for inspecting model output.
- `utility.py` — shared S3, distributed-training, model, and batching utilities.

---

## Developer guide

Our development setup requires access to our S3 datasets at `s3://graphcore-research-us-east-1/2024-10-squashedllama/data/`, and a HuggingFace token with access to the `meta-llama` repository.

First-time setup:
```sh
uv python install 3.11
uv venv .venv --python 3.11
echo "export PYTHONPATH=\${PYTHONPATH}:\$(dirname \${VIRTUAL_ENV})" >> .venv/bin/activate
echo "export TOKENIZERS_PARALLELISM=true" >> .venv/bin/activate

source .venv/bin/activate
uv pip install -r requirements.txt --index https://download.pytorch.org/whl/cu128

uv run pytest tests/
```

S3 paths default to the original internal project bucket, but can be overridden
for compatible bucket layouts:

```sh
export LLAMA_MOBILE_S3_REPO_PATH=s3://my-bucket/llama-mobile
```

Sync from AWS S3:
```sh
aws s3 sync --region=eu-west-1 s3://graphcore-research-us-east-1/2024-10-squashedllama/data/ data/
```

If missing `aws` client (for fetching data from S3 buckets):
```sh
sudo apt-get update
sudo apt install awscli
```

Create and run a Python script:

```py
import train

if __name__ == "__main__":
    settings = train.Settings.default()
    settings.run_name = "dev"
    settings.training.n_steps = 16
    settings.training.batch_size = 128
    settings.training.optimiser.lr = 2**-16
    train.run_experiment(settings)
```

### Internal resources

 - Dataset: `s3://graphcore-research-us-east-1/2024-10-squashedllama/data/`
 - Trained checkpoints: `s3://graphcore-research-us-east-1/2024-10-squashedllama/checkpoints/`
 - Converted models for inference: `s3://graphcore-research-us-east-1/2024-10-squashedllama/models/`
