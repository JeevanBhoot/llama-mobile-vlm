# Squashed Llama training

First-time setup:

```sh
git submodule update --init

python3 -m venv .venv

echo "export PYTHONPATH=\${PYTHONPATH}:\$(dirname \${VIRTUAL_ENV})" >> .venv/bin/activate
echo "export PYTHONPATH=\${PYTHONPATH}:\$(dirname \${VIRTUAL_ENV})/optimal_weight_formats" >> .venv/bin/activate
echo "export TOKENIZERS_PARALLELISM=true" >> .venv/bin/activate

source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

Sync from AWS S3:
```sh
aws s3 sync s3://graphcore-research/2024-10-squashedllama/data/ data/
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
    settings.training.batch_size = 64
    settings.training.optimiser.lr = 2**-16
    train.run_experiment(settings)
```

## Resources

 - Dataset: `s3://graphcore-research/2024-10-squashedllama/data/`
 - Trained checkpoints: `s3://graphcore-research/2024-10-squashedllama/checkpoints/`
