# Squashed Llama training

```sh
python3 -m venv .venv
echo "export PYTHONPATH=\${PYTHONPATH}:\$(dirname \${VIRTUAL_ENV})" >> .venv/bin/activate
echo "export PYTHONPATH=\${PYTHONPATH}:\$(dirname \${VIRTUAL_ENV})/optimal_weight_formats" >> .venv/bin/activate
echo "export TOKENIZERS_PARALLELISM=true" >> .venv/bin/activate
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
cd optimal_weight_formats/ && git apply ../optimal_weight_formats.patch && cd ..
aws s3 sync s3://graphcore-research/2024-10-squashedllama/data/ data/
```
