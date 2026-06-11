# The squashedtensors `.sqt` format (version 3)

Squashedtensors is a derivative of [safetensors](https://github.com/huggingface/safetensors), for saving model weights, quantised weights, tokenizer data, chat-template metadata, and model hyperparameters in one file.

## Differences from safetensors

 - Prepend file magic and version number.
 - The header key `__metadata__` is an arbitrary JSON dict, not just strings.
 - Holes are allowed, and required for alignment.
 - `h.__metadata__.alignment: INTEGER` gives the minimum byte alignment of tensor starts in the file.
 - Quantised tensor dtypes are specified explicitly.

## The format

Data is little-endian. Tensor memory layout is row(dim0)-major ('C').

**File**

| Size (bytes) | Format | Description | Padding |
| --- | --- | -- | --- |
| 4 | literal | `.sqt` `[0x2e, 0x73, 0x71, 0x74]` | - |
| 4 | uint32 | File version (`3`) | - |
| 8 | uint64 | Size of header `N`, including padding | - |
| `N` | char (JSON) | Header | `0x20` (' ') |
| `L` | byte | Buffer | `0x00` |

**Header**

```json
{
    "__metadata__": {
        "source": "meta-llama/Llama-3.2-1B-Instruct",
        "created": "2026-06-11T10:20:00",
        "alignment": 32,
        "comment": "optional free-form comment",
        "config": { ... },
        "vocab": { ... }
    },
    "TENSOR_NAME": {
        "dtype": "BF16" | "INT8" | "S3D8",
        "shape": [DIM0, DIM1, ...],
        "data_offsets": [BEGIN, END],
        "table": {                            // optional (S3D8 only)
            "dtype": "INT8",
            "shape": [32, 3],
            "data_offsets": [BEGIN, END]
        },
        "scale": {                            // optional (INT8 and S3D8 only)
            "dtype": "BF16",
            "shape": [DIM0, DIM1, ...],
            "data_offsets": [BEGIN, END]
        }
    }, ...
}
```

`data_offsets` are relative to the start of the buffer, not the start of the file.

## Metadata

| Key | Format | Description |
| --- | --- | -- |
| `source` | `STRING` | Original model name, usually the Hugging Face model path |
| `created` | `STRING` | Timestamp of file creation, ISO 8601 |
| `alignment` | `INTEGER` | Byte alignment of the buffer and all tensor start offsets within it |
| `comment` | `STRING` | Optional free-form comment supplied by `squashedtensors.py --comment` |
| `config.text.*` | * | Text model hyperparameters |
| `config.vision` | `OBJECT` or `null` | Vision model hyperparameters, or `null` for text-only models |
| `vocab.chat_template` | `STRING` | Hugging Face tokenizer chat template used as model provenance and native prompt-format metadata |
| `vocab.begin_of_text_id` | `INTEGER` | Special token ID for begin of text |
| `vocab.start_header_id` | `INTEGER` | Special token ID for chat role header start |
| `vocab.end_header_id` | `INTEGER` | Special token ID for chat role header end |
| `vocab.eot_id` | `INTEGER` | Special token ID for end of chat turn |
| `vocab.stop_token_ids` | `LIST[INTEGER]` | Token IDs that terminate generation; may include end-of-text, end-of-message, and end-of-turn IDs |
| `vocab.image_id` | `INTEGER` or `null` | Special token ID for image input, if the tokenizer defines one |
| `vocab.pre_tokenizer` | `STRING` | Regex for pre-tokenization |
| `vocab.merges` | `LIST[STRING]` | BPE merge rules |
| `vocab.vocab` | `LIST[STRING]` | Vocabulary list, an ID-to-token mapping |

## Dtypes

`BF16` data is stored as contiguous little-endian bfloat16.

`INT8` data is stored as contiguous int8. It must have a nested `scale` entry. For a 2D tensor with shape `[DIM0, DIM1]`, the scale shape is `[DIM0, 1]` for per-output-channel scaling.

`S3D8` data is stored as a contiguous packed byte stream. It requires:

 - an `INT8` lookup `table` entry of shape `[32, 3]`, and
 - a nested `BF16` `scale` entry using the same per-output-channel convention as `INT8`.
