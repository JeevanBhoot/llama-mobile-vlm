# Squashed Llama

This is a monorepo for squashed llama work.

Components:
 - [`/android`](android) - client demo app for Android phones (Kotlin)
 - [`/inference`](inference) - on-device inference library (C++)
 - [`/training`](training) - on-server quantisation & training (PyTorch)

Other:
 - [`/docs`](docs)  - design docs (.md)
 - [`/models`](https://eu-west-1.console.aws.amazon.com/s3/buckets/graphcore-research?prefix=2024-10-squashedllama/models/) (s3 bucket) - models for tests & demos (.sqt)
 - [`/notebooks`](https://github.com/graphcore-research/squashed-llama/tree/notebooks) (branch) - notes and reports (.ipynb)

In addition to cloning the repo, please run:

```sh
git clone git@github.com:graphcore-research/squashed-llama.git --branch notebooks notebooks
aws s3 sync s3://graphcore-research/2024-10-squashedllama/models/ models/
```
