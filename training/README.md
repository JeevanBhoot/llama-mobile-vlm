# Squashed Llama training

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
aws s3 sync s3://graphcore-research-us-east-1/2024-10-squashedllama/data/ data/
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

## Create .sqt files for the C++ inference library

```sh
python squashedtensors.py meta-llama/Llama-3.2-1B-Instruct ../models/Llama-3.2-1B-Instruct-BF16.sqt
python squashedtensors.py meta-llama/Llama-3.2-11B-Vision-Instruct ../models/Llama-3.2-11B-Vision-Instruct-BF16.sqt
```

## Resources

 - Dataset: `s3://graphcore-research-us-east-1/2024-10-squashedllama/data/`
 - Trained checkpoints: `s3://graphcore-research-us-east-1/2024-10-squashedllama/checkpoints/`
