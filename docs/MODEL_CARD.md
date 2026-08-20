# Llama-Mobile Model Card

## Model Summary

Llama-Mobile is a collection of derivative Meta Llama 3.2 model artifacts for
research into efficient CPU inference. It includes text-only conversions of
`meta-llama/Llama-3.2-1B-Instruct` and vision-language conversions and
quantization-aware-trained weights derived from
`meta-llama/Llama-3.2-11B-Vision-Instruct`.

The `.sqt` artifacts bundle weights, quantization metadata, tokenizer data, chat
templates, and model configuration for the project's C++ inference library. The
`.safetensors` artifacts expose the vision QAT checkpoint in formats used by the
standalone Transformers demo and conversion workflow.

## Released Artifacts

| Artifact | Source model | Representation | Purpose |
| --- | --- | --- | --- |
| `text-1B-bf16.sqt` | Llama 3.2 1B Instruct | BF16 | Text baseline |
| `text-1B-int8.sqt` | Llama 3.2 1B Instruct | Direct-cast INT8 | Lightweight text demo and benchmark |
| `text-1B-s3d8.sqt` | Llama 3.2 1B Instruct | Direct-cast S3D8 | S3D8 smoke testing; quality is expected to be poor without QAT |
| `vision-11B-bf16.sqt` | Llama 3.2 11B Vision Instruct | BF16 | Vision-language baseline |
| `vision-11B-s3d8.sqt` | Llama 3.2 11B Vision Instruct | QAT S3D8 weights, INT8 activations | Primary compressed vision-language model |
| `vision-11B-s3d8-as-bf16.safetensors` | Llama 3.2 11B Vision Instruct | S3D8 values expanded to BF16 | Legacy Transformers-compatible checkpoint |
| `vision-11B-s3d8-packed.safetensors` | Llama 3.2 11B Vision Instruct | Packed S3D8, BF16 scales, INT8 lookup tables | Compressed standalone demo checkpoint |
| `vision-11B-s3d8-training.safetensors` | Llama 3.2 11B Vision Instruct | QAT master weights and centroids | Training and conversion checkpoint |

The source repository's `docs/models.md` contains detailed conversion and
download commands. The `.sqt` format is documented in `docs/file_format.md`.

## Intended Use

These artifacts are intended for research and evaluation of quantized language
and vision-language inference, reproduction of Llama-Mobile results, and
non-production demonstrations on Arm CPUs. The text model accepts text and
produces text. The vision model accepts English text with an image and produces
text; text-only use inherits the base model's supported languages.

The artifacts are not presented as production-ready models or complete AI
systems. Applications must add safeguards appropriate to their users and use
case and comply with the Llama 3.2 Community License, Acceptable Use Policy, and
applicable law.

## Evaluation and Limitations

`vision-11B-s3d8.sqt` is the QAT artifact used for the project's primary task
results. The BF16 artifacts are baselines, while the direct-cast text artifacts
are primarily functional and performance test inputs. Refer to the associated
paper for the reported evaluation methodology and results.

Quantization can reduce quality relative to the BF16 source models, especially
for the direct-cast S3D8 text model. Outputs may be inaccurate, incomplete,
biased, or inappropriate. The artifacts have not been comprehensively evaluated
for every device, language, safety risk, or downstream use case and must not be
relied upon for production, safety-critical, medical, legal, or financial
decisions.

For base-model capabilities and limitations, see Meta's official
[Llama 3.2 text model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md)
and [Llama 3.2 Vision model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md).

## License and Attribution

**Built with Llama.** Llama-Mobile is not affiliated with, sponsored by, or
endorsed by Meta.

The derivative model artifacts are distributed under the official
[Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE),
Copyright © Meta Platforms, Inc. All Rights Reserved, and are subject to Meta's
[Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy). The required
attribution is retained in `NOTICE`. The repository includes the
agreement as `LICENSE_LLAMA_3.2`; each S3 model prefix includes it as `LICENSE`.
The project's source code is separately licensed under the MIT License.

## Download and Integrity

The `.sqt` models are published under:

```text
s3://graphcore-research-public/2026-llama-mobile/models/20260611/
```

The `.safetensors` checkpoints are published under:

```text
s3://graphcore-research-public/2026-llama-mobile/hf_models/
```

Each prefix contains the Llama license as `LICENSE`, the attribution as
`NOTICE`, this model card, and a `SHA256SUMS` manifest. After downloading
artifacts from one prefix, verify them with:

```sh
sha256sum --check --ignore-missing SHA256SUMS
```
