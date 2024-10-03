# The squashedtensors `.sqt` format (version 0)

Squashedtensors is a derivative of [safetensors](https://github.com/huggingface/safetensors), for saving (quantised) model weights, tokenizer and hyperparameters.

**TODO - tokenizer.**

## Differences from safetensors

 - Prepend file magic & version number.
 - The header key `__metadata__` is an arbitary JSON dict, not just strings.
 - Holes are allowed, and required for alignment.
 - `h.__metadata__.alignment: INTEGER` gives the minimum byte alignment of tensor (starts) in the file.
 - We will specify quantised tensor dtypes.

## The format

Data is little-endian. Tensor memory layout is row-major ('C').

**File**

| Size (bytes) | Format | Description | Padding |
| --- | --- | -- | --- |
| 4 | literal | ".sqt" `[0x2e, 0x73, 0x71, 0x74]` | - |
| 4 | uint32 | File version (0) | - |
| 8 | uint64 | Size of header `N`, including padding | - |
| `N` | char (JSON) | Header | `0x20` (' ') |
| `L` | byte | Buffer | `0x00` |

**Header**

```json
{
    "__metadata__": { ... },
    "TENSOR_NAME": {
        "dtype": "DTYPE",
        "shape": [DIM0, DIM1, ...],
        "data_offsets": [BEGIN, END]
    }, ...
}
```

Note that `data_offsets` are relative to the start of the buffer.

**Metadata**

| Key | Format | Description |
| --- | --- | -- |
| source | STRING | Original model name (e.g. huggingface model path) |
| created | STRING | Timestamp of file creation (model quantisation), ISO 8601 |
| alignment | INTEGER | Byte alignment of the buffer & all tensor start offsets within it |
| config.* | * | Model hyperparameters |
