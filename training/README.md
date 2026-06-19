# Llama-Mobile Training Reference

This directory contains the quantization-aware training, evaluation, and model
conversion code used during the Llama-Mobile project.

It is provided for transparency and provenance. It is not intended to be a
supported external training pipeline: the original workflow depends on internal
datasets, checkpoints, and cluster infrastructure.

For runnable public artifacts, see the [parent README](../README.md).

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
