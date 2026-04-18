# The squashedtensors `.sqt` format (version 2)

Squashedtensors is a derivative of [safetensors](https://github.com/huggingface/safetensors), for saving (quantised) model weights, tokenizer and hyperparameters.

## Differences from safetensors

 - Prepend file magic & version number.
 - The header key `__metadata__` is an arbitary JSON dict, not just strings.
 - Holes are allowed, and required for alignment.
 - `h.__metadata__.alignment: INTEGER` gives the minimum byte alignment of tensor (starts) in the file.
 - We will specify quantised tensor dtypes.

## The format

Data is little-endian. Tensor memory layout is row(dim0)-major ('C').

**File**

| Size (bytes) | Format | Description | Padding |
| --- | --- | -- | --- |
| 4 | literal | ".sqt" `[0x2e, 0x73, 0x71, 0x74]` | - |
| 4 | uint32 | File version (2) | - |
| 8 | uint64 | Size of header `N`, including padding | - |
| `N` | char (JSON) | Header | `0x20` (' ') |
| `L` | byte | Buffer | `0x00` |

**Header**

```json
{
    "__metadata__": { ... },
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

Note that `data_offsets` are relative to the start of the buffer.

**Metadata**

| Key | Format | Description |
| --- | --- | -- |
| `source` | `STRING` | Original model name (e.g. huggingface model path) |
| `created` | `STRING` | Timestamp of file creation (model quantisation), ISO 8601 |
| `alignment` | `INTEGER` | Byte alignment of the buffer & all tensor start offsets within it |
| `config.*` | * | Model hyperparameters (model-dependent) |
| `vocab.begin_of_text_id` | `INTEGER` | Special token ID for begin of text |
| `vocab.end_of_text_id` | `INTEGER` | Special token ID for end of text |
| `vocab.image_id` | `INTEGER` or `null` | Special token ID for image input, if the tokenizer defines one |
| `vocab.pre_tokenizer` | `STRING` | Regex for pre-tokenization |
| `vocab.merges` | `LIST[STRING]` | BPE merge rules |
| `vocab.vocab` | `LIST[STRING]` | Vocabulary list, an ID-to-token mapping |

**Dtypes**

`BF16` data is stored as contiguous (little-endian) bfloat16.

`INT8` data is stored as contiguous int8.

`S3D8` data is stored as a contiguous packed byte stream. It requires an `INT8` lookup `"table"` entry of shape `[32, 3]`, as described above.

_Scaling:_ `INT8` and `S3D8` must also be scaled per-output-channel, with a nested `"scale"` entry of shape `[DIM0, 1]` for a 2D tensor `[DIM0, DIM1]`.
