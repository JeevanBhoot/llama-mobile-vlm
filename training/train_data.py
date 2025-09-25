import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import datasets
import torch
import torch.multiprocessing as mp
import transformers
from PIL.ImageFile import ImageFile

from utility import (
    LLAMA_PROMPT_TEMPLATES,
    LOCAL_DATA_PATH,
    S3_DATA_PATH,
    set_padding_side_left,
)

DEFAULT_PROMPT = "Describe the image:\n"


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


@dataclass(init=False)
class ImageDataset:
    """Generic dataset wrapper yielding image-prompt pairs"""

    name: str
    hf_name: str
    split: str
    remove_columns: list[str]

    def _get_paths(self) -> tuple[str, str]:
        dir_name = f"{self.name}-{self.split}"
        s3_path = f"{S3_DATA_PATH}/{dir_name}"
        local_path = f"{LOCAL_DATA_PATH}/{dir_name}"
        return s3_path, local_path

    def download(self, sync: bool = True) -> None:
        # This can be slow as it downloads the whole dataset
        ds = datasets.load_dataset(
            self.hf_name,
            split=self.split,
            trust_remote_code=True,
        )

        ds = ds.remove_columns(self.remove_columns)

        # Add an index column
        ds = ds.add_column(name="index", column=range(len(ds)))

        s3_path, local_path = self._get_paths()

        ds.save_to_disk(local_path)

        # Upload to s3
        if sync:
            subprocess.run(["aws", "s3", "sync", local_path, s3_path], check=True)

    def data(
        self,
        prompt: Optional[str],
        seed: Optional[int] = None,
        sync: bool = True,
    ) -> datasets.Dataset:
        s3_path, local_path = self._get_paths()
        if sync:
            try:
                # Check if files exist on S3
                subprocess.run(
                    ["aws", "s3", "ls", s3_path],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                print(
                    "Warning: Dataset not found on S3, "
                    "downloading locally and uploading to S3",
                    file=sys.stderr,
                    flush=True,
                )
                self.download(sync=True)
            subprocess.run(["aws", "s3", "sync", s3_path, local_path], check=True)

        try:
            ds = datasets.load_from_disk(local_path)
        except FileNotFoundError:
            print(
                "Warning: Dataset not found on disk, downloading from HuggingFace",
                file=sys.stderr,
                flush=True,
            )
            self.download(sync=False)
            ds = datasets.load_from_disk(local_path)

        # Avoid shuffling data when seed is not passed
        if seed is not None:
            ds = ds.shuffle(seed)

        # Add prompt column
        if prompt:
            ds = ds.add_column(name="prompt", column=[prompt for _ in range(len(ds))])

        return ds


class ImageNet(ImageDataset):
    def __init__(self, split: str = "train"):
        self.name = "imagenet"
        self.hf_name = "ILSVRC/imagenet-1k"
        self.split = split
        self.remove_columns = ["label"]


class Coco(ImageDataset):
    def __init__(self, split: str = "validation"):
        self.name = "coco"
        self.hf_name = "ariG23498/coco2017"
        self.split = split
        self.remove_columns = ["objects"]


@dataclass
class GenerationConfig:
    model_name: str
    dataset_name: str
    split: str
    prompt: str
    seed: Optional[int]
    data_range: Optional[tuple[int, int]]
    n_generated_tokens: int
    sampling_settings: SamplingSettings

    @classmethod
    def default(cls, **kwargs: Any) -> "GenerationConfig":
        default_args = dict(
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
            dataset_name="imagenet",
            split="train",
            prompt=LLAMA_PROMPT_TEMPLATES["instruct"].format(prompt=DEFAULT_PROMPT),
            seed=None,
            data_range=None,
            n_generated_tokens=1024,
            sampling_settings=SamplingSettings.default(),
        )
        default_args.update(kwargs)

        return GenerationConfig(**default_args)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_hash(self) -> str:
        config_str = json.dumps(self.to_dict())
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:6]


def _generate(
    model: transformers.PreTrainedModel,
    data: datasets.Dataset,
    n_generated_tokens: int,
    sampling_settings: SamplingSettings,
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
            with set_padding_side_left(processor.tokenizer):
                inp = processor(
                    imgs, batch["prompt"], padding=True, return_tensors="pt"
                ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=n_generated_tokens,
                    **sampling_settings.to_dict(),
                )
            t = time.time()
            print(
                f"Batch: {i}/{n_batches}, Time: {t - t0:.2f}"
                f", Total time: {t - start_t:.2f}",
                file=g,
                flush=True,
            )
            for idx, out_text in zip(batch["index"], processor.batch_decode(out)):
                yield idx, out_text.replace(processor.tokenizer.pad_token, "")


def _generate_worker(
    rank: int,
    world_size: int,
    out_path: str,
    config: GenerationConfig,
    batch_size: int,
    dtype: str,
) -> None:
    assert batch_size % world_size == 0

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    model = transformers.MllamaForConditionalGeneration.from_pretrained(
        config.model_name, torch_dtype=getattr(torch, dtype), device_map=device
    )

    data = IMAGE_DATASETS[config.dataset_name](split=config.split).data(
        prompt=config.prompt, seed=config.seed
    )
    if config.data_range is not None:
        data = data.select(range(*config.data_range))

    # Drop last global batch
    data = data.select(range(len(data) - len(data) % batch_size))

    # Shard data across devices
    data = data.shard(world_size, rank)

    log_path = Path(out_path).parent.parent / "log" / f"log_{rank}"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for idx, out_text in _generate(
            model,
            data,
            config.n_generated_tokens,
            config.sampling_settings,
            batch_size // world_size,
            log_path,
        ):
            print(json.dumps([idx, out_text]), file=f, flush=True)


def generate_data(
    config: GenerationConfig,
    batch_size: int = 16 * torch.cuda.device_count(),
    dtype: str = "bfloat16",
    world_size: int = torch.cuda.device_count(),
    data_path: str = LOCAL_DATA_PATH,
) -> str:
    n_examples = config.data_range[1] - config.data_range[0]
    if n_examples % batch_size != 0:
        orig_range = config.data_range
        config.data_range = (orig_range[0], orig_range[1] - n_examples % batch_size)

        print(
            f"Warning: Requested data range {orig_range} is not divisible "
            f"by batch size {batch_size}. Changing to {config.data_range}.",
            file=sys.stderr,
            flush=True,
        )

    assert world_size <= torch.cuda.device_count()

    ctx = mp.get_context("spawn")

    dir_path = (
        f"{data_path}/{config.dataset_name}-{config.split}-generation"
        f"/{config.to_hash()}"
    )
    out_path = dir_path + "/out/out{}.jsonl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_generate_worker,
            args=(
                rank,
                world_size,
                out_path.format(rank),
                config,
                batch_size,
                dtype,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Save config info
    d = config.to_dict()
    d["created at"] = datetime.now().isoformat(timespec="seconds")
    config_path = f"{dir_path}/config.json"
    with open(config_path, "w") as f:
        json.dump(d, f, indent=2)

    return dir_path


IMAGE_DATASETS: dict[str, ImageDataset] = {"imagenet": ImageNet, "coco": Coco}


@dataclass
class Datum:
    index: int
    image: ImageFile
    prompt: str
    out: str


class Dataset:
    def __init__(
        self, paths: list[str], n_examples: list[Optional[int]], seed: int = 563673
    ):
        data_to_concat: list[datasets.Dataset] = []
        for path, n in zip(paths, n_examples):
            path = Path(path)
            config_path = path / "config.json"
            with open(config_path) as f:
                config = json.loads(f.read())

            data = (
                IMAGE_DATASETS[config["dataset_name"]](split=config["split"])
                .data(prompt=config["prompt"], seed=config["seed"])
                .select(range(*config["data_range"]))
            )

            if n is None:
                n = len(data)

            out_path = path / "out"
            out = []
            for filename in out_path.glob("out*.jsonl"):
                with open(filename) as f:
                    for line in f:
                        idx, text = json.loads(line)
                        out.append((idx, text))

            # This is kinda dangerous - assumes indices are corresponding to data
            out.sort(key=lambda x: x[0])
            data = data.add_column(name="out", column=[x[1] for x in out])
            data_to_concat.append(data.select(range(n)))

        self.data = datasets.concatenate_datasets(data_to_concat).shuffle(seed)

    # TODO: This function is unnecessary now!
    def get_datums(self) -> Iterable[Datum]:
        for x in self.data:
            yield Datum(**x)
