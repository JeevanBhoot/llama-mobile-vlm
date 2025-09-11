import json
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import datasets
import numpy as np
import torch
import torch.multiprocessing as mp
import transformers
from PIL.ImageFile import ImageFile

from utility import set_padding_side_left


@dataclass
class SamplingSettings:
    do_sample: bool
    temperature: Optional[float]
    top_p: Optional[float]

    @classmethod
    def default(cls) -> "SamplingSettings":
        return cls(do_sample=True, temperature=0.6, top_p=0.9)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImageDataset:
    """Generic dataclass yielding image-prompt pairs"""

    HF_NAME: str
    HF_SPLIT: str
    S3_DIRNAME: str
    REMOVE_COLUMNS: list[str]
    PROMPT: str
    DEFAULT_SEED: int

    @classmethod
    def _get_paths(cls) -> tuple[str]:
        s3_path = f"s3://graphcore-research/2024-10-squashedllama/data/{cls.S3_DIRNAME}"
        local_path = f"data/{cls.S3_DIRNAME}"
        return s3_path, local_path

    @classmethod
    def _download_and_sync(cls) -> None:
        # This can be slow as it downloads the whole dataset
        ds = datasets.load_dataset(
            cls.HF_NAME,
            split=cls.HF_SPLIT,
            trust_remote_code=True,
        )

        ds = ds.remove_columns(cls.REMOVE_COLUMNS)

        # Add an index column
        ds = ds.add_column(name="index", column=np.arange(len(ds)))

        # Add a prompt column
        ds = ds.add_column(
            name="prompt", column=[f"<|image|>{cls.PROMPT}" for _ in range(len(ds))]
        )

        ds.save_to_disk(f"data/{cls.S3_DIRNAME}")

        # Upload to s3
        s3_path, local_path = cls._get_paths()
        command = ["aws", "s3", "sync", local_path, s3_path]
        result = subprocess.run(command)
        result.check_returncode()

    @classmethod
    def data(cls, seed: Optional[int] = None) -> datasets.Dataset:
        if seed is None:
            seed = cls.DEFAULT_SEED

        # Sync local folder
        s3_path, local_path = cls._get_paths()
        result = subprocess.run(["aws", "s3", "sync", s3_path, local_path])
        result.check_returncode()

        ds = datasets.load_from_disk(local_path).shuffle(seed)
        return ds


class ImageNet(ImageDataset):
    """ImageNet (test) dataset with custom prompts"""

    HF_NAME = "ILSVRC/imagenet-1k"
    HF_SPLIT = "test"
    S3_DIRNAME = "imagenet"
    REMOVE_COLUMNS = ["label"]
    PROMPT = "Describe the image:\n"
    DEFAULT_SEED = 480932


class Coco(ImageDataset):
    """Coco (validation) dataset with custom prompts"""

    HF_NAME = "ariG23498/coco2017"
    HF_SPLIT = "validation"
    S3_DIRNAME = "coco"
    REMOVE_COLUMNS = ["objects"]
    PROMPT = "Describe the image:\n"
    DEFAULT_SEED = 644346


IMAGE_DATASETS: dict[str, ImageDataset] = {"imagenet": ImageNet, "coco": Coco}


@dataclass
class Datum:
    index: int
    image: ImageFile
    prompt: str
    out: str


@dataclass
class Dataset:
    model_name: str
    dataset_name: str
    limit: Optional[int]
    _out: dict[int, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "Dataset":
        # NOTE: maybe should change this
        d = json.loads(text)
        d["_out"] = {int(k): v for k, v in d["_out"].items()}
        return cls(**d)

    def save(self, path: str) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> "Dataset":
        return cls.from_json(Path(path).read_text())

    def get_datums(self) -> Iterable[Datum]:
        data = IMAGE_DATASETS[self.dataset_name].data()

        if self.limit is not None:
            data = data.select(range(self.limit))

        for x in data:
            if x["index"] not in self._out:
                warnings.warn(
                    f"No text generation found for image index {x['index']}."
                    "Stopping iterator."
                )
                break
            yield Datum(
                index=x["index"],
                image=x["image"],
                prompt=x["prompt"],
                out=self._out[x["index"]],
            )


def _generate(
    model: transformers.PreTrainedModel,
    data: datasets.Dataset,
    n_generated_tokens: int,
    sampling_settings: dict[str, Any],
    batch_size: int,
    log_path: str | Path,
) -> Iterable[tuple[int, str]]:
    processor = transformers.AutoProcessor.from_pretrained(model.name_or_path)

    start_t = time.time()
    n_batches = len(data) // batch_size
    with open(log_path, "w") as g:
        for i, batch in enumerate(data.iter(batch_size), start=1):
            t0 = time.time()

            # NOTE: Each example gets a list of images
            imgs = [[img] for img in batch["image"]]
            prompts = batch["prompt"]
            idxs = batch["index"]
            with set_padding_side_left(processor.tokenizer):
                inp = processor(imgs, prompts, padding=True, return_tensors="pt").to(
                    model.device
                )
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=n_generated_tokens, **sampling_settings
                )
            t = time.time()
            print(
                f"Batch: {i}/{n_batches}, Time: {t - t0:.2f}"
                f", Total time: {t - start_t:.2f}",
                file=g,
                flush=True,
            )
            for idx, text in zip(idxs, processor.batch_decode(out)):
                yield idx, text


def _generate_worker(
    rank: int,
    world_size: int,
    out_path: str,
    model_name: str,
    dataset_name: str,
    limit: Optional[int],
    n_generated_tokens: int,
    sampling_settings: dict[str, Any],
    batch_size: int,
    dtype: str,
) -> None:
    assert batch_size % world_size == 0

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    model = transformers.MllamaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=getattr(torch, dtype), device_map=device
    )

    data = IMAGE_DATASETS[dataset_name].data()
    if limit is not None:
        data = data.select(range(limit))

    # Drop last global batch
    data = data.select(range(len(data) - len(data) % batch_size))

    # Shard data across devices
    data = data.shard(world_size, rank)

    log_path = Path(out_path).parent / "log" / f"log_{rank}"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for idx, text in _generate(
            model,
            data,
            n_generated_tokens,
            sampling_settings,
            batch_size // world_size,
            log_path,
        ):
            print(json.dumps({idx: text}), file=f, flush=True)


def generate_dataset(
    model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    dataset_name: str = "imagenet",
    limit: Optional[int] = None,
    n_generated_tokens: int = 1024,
    sampling_settings: dict[str, Any] = SamplingSettings.default().to_dict(),
    batch_size: int = 16,
    dtype: str = "bfloat16",
    world_size: int = torch.cuda.device_count(),
) -> Dataset:
    assert world_size <= torch.cuda.device_count()

    ctx = mp.get_context("spawn")

    out_path = "data/{}-generation/tmp/out{}.jsonl"

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_generate_worker,
            args=(
                rank,
                world_size,
                out_path.format(dataset_name, rank),
                model_name,
                dataset_name,
                limit,
                n_generated_tokens,
                sampling_settings,
                batch_size,
                dtype,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    combined_out = {}
    for rank in range(world_size):
        with open(out_path.format(dataset_name, rank), "r") as f:
            for line in f:
                d = json.loads(line)
                combined_out.update({int(k): v for k, v in d.items()})

    return Dataset(
        model_name=model_name,
        dataset_name=dataset_name,
        limit=limit,
        _out=combined_out,
    )
