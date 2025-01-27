from typing import Iterable, TypeVar

T = TypeVar("T")


def batches(iterable: Iterable[T], batch_size: int) -> Iterable[list[T]]:
    """Chunks `iterable` into batches of consecutive values."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
