# Squashed Llama training

## Development

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

## Model conversion for the C++ inference library

Models are saved for private access to `s3://graphcore-research-us-east-1/2024-10-squashedllama/models/`. For public models, see the top-level [README](../README.md).

```sh
mkdir -p models/v3/ tmp/

# Text | BF16 Baseline
python squashedtensors.py meta-llama/Llama-3.2-1B-Instruct models/v3/text-1B-bf16.sqt --comment "Llama 3.2 1B Instruct, BF16"

# Text | INT8 Direct Cast
python direct_cast.py meta-llama/Llama-3.2-1B-Instruct tmp/text-1B-int8.safetensors --format int8
python squashedtensors.py meta-llama/Llama-3.2-1B-Instruct models/v3/text-1B-int8.sqt --checkpoint tmp/text-1B-int8.safetensors --comment "Llama 3.2 1B Instruct, INT8 (Direct Cast)"

# Text | S3D8 Direct Cast
python direct_cast.py meta-llama/Llama-3.2-1B-Instruct tmp/text-1B-s3d8.safetensors --format s3d8
python squashedtensors.py meta-llama/Llama-3.2-1B-Instruct models/v3/text-1B-s3d8.sqt --checkpoint tmp/text-1B-s3d8.safetensors --comment "Llama 3.2 1B Instruct, S3D8 (Direct Cast)"

# Vision | BF16 Baseline
python squashedtensors.py meta-llama/Llama-3.2-11B-Vision-Instruct models/v3/vision-11B-bf16.sqt --comment "Llama 3.2 11B Vision Instruct, BF16"

# Vision | INT8 Direct Cast
python direct_cast.py meta-llama/Llama-3.2-11B-Vision-Instruct tmp/vision-11B-int8.safetensors --format int8
python squashedtensors.py meta-llama/Llama-3.2-11B-Vision-Instruct models/v3/vision-11B-int8.sqt --checkpoint tmp/vision-11B-int8.safetensors --comment "Llama 3.2 11B Vision Instruct, INT8 (Direct Cast)"

# Vision | QAT
aws s3 cp s3://graphcore-research-us-east-1/2024-10-squashedllama/checkpoints/proud-sponge-1878.safetensors tmp/
python squashedtensors.py meta-llama/Llama-3.2-11B-Vision-Instruct models/v3/vision-11B-s3d8.sqt --checkpoint tmp/proud-sponge-1878.safetensors --comment "Llama 3.2 11B Vision Instruct, QAT (S3D8) [proud-sponge-1878]"
```

## Resources

 - Dataset: `s3://graphcore-research-us-east-1/2024-10-squashedllama/data/`
 - Trained checkpoints: `s3://graphcore-research-us-east-1/2024-10-squashedllama/checkpoints/`
