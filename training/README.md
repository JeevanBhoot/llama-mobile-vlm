# Squashed Llama training

```sh
python3 -m venv .venv
echo "export PYTHONPATH=\${PYTHONPATH}:\$(dirname \${VIRTUAL_ENV})" >> .venv/bin/activate
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```
