import hashlib
import json
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import datasets
import torch
import torch.multiprocessing as mp
import transformers
from PIL.ImageFile import ImageFile

from prompt_sampling import INSTRUCTIONS, PROMPTS, get_prompts
from utility import (
    LLAMA_PROMPT_TEMPLATES,
    LOCAL_DATA_PATH,
    S3_DATA_PATH,
    check_s3_access,
    set_padding_side_left,
)

DEFAULT_PROMPT = "Describe the image:\n"


@dataclass
class SamplingSettings:
    do_sample: bool
    temperature: float | None
    top_p: float | None

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
        prompt_fn: Callable[[int], list[str]] | None,
        seed: int | None = None,
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
        if prompt_fn is not None:
            ds = ds.add_column(name="prompt", column=prompt_fn(len(ds)))

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


class VQA(ImageDataset):
    def __init__(self, split: str = "validation"):
        self.name = "vqav2"
        self.hf_name = "lmms-lab/VQAv2"
        self.split = split
        self.remove_columns = ["image_id", "question", "answers"]
        self._template = LLAMA_PROMPT_TEMPLATES["instruct"]

    def data(
        self,
        prompt_fn: Callable[[int], list[str]] | None = None,
        seed: int | None = None,
        sync: bool = True,
    ) -> datasets.Dataset:
        import eval.vqa

        assert prompt_fn is None, "VQA dataset assumes default prompts"
        if seed is None:
            print(
                "Warning: Seed should be specified  to avoid repeated images.",
                flush=True,
                file=sys.stderr,
            )

        data = eval.vqa.VQA.data(
            shuffle_seed=None, split=self.split, limit=None, load_from_s3=sync
        )

        # Use question_id as the index
        data = data.rename_column("question_id", "index")

        # Add system template
        prompts = [
            self._template.format(
                prompt=eval.vqa.VQA.QUESTION_TEMPLATE.format(question=q)
            )
            for q in data["question"]
        ]
        data = data.add_column("prompt", prompts)

        data = data.remove_columns(self.remove_columns)

        if seed is not None:
            data = data.shuffle(seed)

        return data


@dataclass
class PromptConfig:
    templates: list[str]
    template_weights: list[float]
    instructions: list[str]
    prompts: list[str]
    instruction_prob: float
    seed: int

    @classmethod
    def default(cls, **kwargs: Any) -> "PromptConfig":
        templates = [
            LLAMA_PROMPT_TEMPLATES["simple"],
            LLAMA_PROMPT_TEMPLATES["instruct"],
        ]
        default_args = dict(
            templates=templates,
            template_weights=[0.25, 0.75],
            instructions=INSTRUCTIONS,
            prompts=PROMPTS,
            instruction_prob=0.75,
            seed=693785,
        )
        default_args.update(kwargs)
        return PromptConfig(**default_args)


@dataclass
class GenerationConfig:
    model_name: str
    dataset_name: str
    split: str
    prompt_config: PromptConfig | None
    seed: int | None
    data_range: tuple[int, int] | None
    n_generated_tokens: int
    sampling_settings: SamplingSettings

    @classmethod
    def default(cls, **kwargs: Any) -> "GenerationConfig":
        default_args = dict(
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
            dataset_name="imagenet",
            split="train",
            prompt_config=PromptConfig.default(),
            seed=None,
            data_range=None,
            n_generated_tokens=512,
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
                pad_token = processor.tokenizer.special_tokens_map["pad_token"]
                bos_token = processor.tokenizer.special_tokens_map["bos_token"]
                yield idx, out_text.replace(pad_token, "").replace(bos_token, "")


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

    # NOTE: data method expects prompt_fn with only size argument
    data = IMAGE_DATASETS[config.dataset_name](split=config.split).data(
        prompt_fn=(
            (lambda n: (get_prompts(n, **asdict(config.prompt_config))))
            if config.prompt_config
            else None
        ),
        seed=config.seed,
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
    batch_size: int = 128,
    dtype: str = "bfloat16",
    world_size: int = torch.cuda.device_count(),
    data_path: str = LOCAL_DATA_PATH,
    sync_to_s3: bool = True,
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

    dir_path = f"{config.dataset_name}-{config.split}-generation/{config.to_hash()}"
    path = f"{data_path}/{dir_path}"
    out_path = path + "/out/out{}.jsonl"
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

    if all(p.exitcode == 0 for p in processes):
        (Path(path) / "log" / "_SUCCESS").touch()

    # Save config info
    d = config.to_dict()
    d["created at"] = datetime.now().isoformat(timespec="seconds")
    config_path = f"{path}/config.json"
    with open(config_path, "w") as f:
        json.dump(d, f, indent=2)

    if sync_to_s3:
        check_s3_access()
        s3_path = f"{S3_DATA_PATH}/{dir_path}"
        subprocess.run(["aws", "s3", "sync", path, s3_path], check=True)

    return path


IMAGE_DATASETS: dict[str, ImageDataset] = {
    "imagenet": ImageNet,
    "coco": Coco,
    "vqav2": VQA,
}


@dataclass
class Datum:
    index: int
    image: ImageFile
    prompt: str
    out: str


def load_config(path: str | Path) -> GenerationConfig:
    from utility import from_dict

    path = Path(path)
    config_path = path / "config.json"
    with open(config_path) as f:
        return from_dict(GenerationConfig, json.loads(f.read()))


class Dataset:
    def __init__(
        self, paths: list[str], n_examples: list[int | None], seed: int = 563673
    ):
        data_to_concat: list[datasets.Dataset] = []
        self._configs = []
        for path, n in zip(paths, n_examples):
            local_path = Path(LOCAL_DATA_PATH) / path
            if not local_path.exists():
                check_s3_access()
                s3_path = f"{S3_DATA_PATH}/{path}"
                subprocess.run(
                    ["aws", "s3", "sync", s3_path, str(local_path)], check=True
                )

            config = load_config(local_path)
            self._configs.append(config)

            data = (
                IMAGE_DATASETS[config.dataset_name](split=config.split)
                .data(
                    prompt_fn=(
                        (lambda n: (get_prompts(n, **asdict(config.prompt_config))))
                        if config.prompt_config
                        else None
                    ),
                    seed=config.seed,
                )
                .select(range(*config.data_range))
            )

            if n is None:
                n = len(data)

            out_path = local_path / "out"
            out = {}
            for filename in out_path.glob("out*.jsonl"):
                with open(filename) as f:
                    for line in f:
                        idx, text = json.loads(line)
                        out[idx] = text

            data = data.add_column(
                name="out", column=[out[idx] for idx in data["index"]]
            )
            data_to_concat.append(data.select(range(n)))

        self.data = datasets.concatenate_datasets(data_to_concat).shuffle(seed)

    def __len__(self):
        return len(self.data)

    def get_datums(self) -> Iterable[Datum]:
        for x in self.data:
            yield Datum(**x)


def join_data(paths: list[str | Path], out_path: str | Path) -> None:
    """Join generated data from a distributed process and write to a new path.
    Requires:
    - config fields for each path should match (apart from data_range)
    - data_range fields should be joinable
    (such as [0, 16] and [16, 32], but not [0, 16], [14, 32])
    """
    # Load configs, check if they can be joined
    configs = []
    for p in paths:
        configs.append(load_config(p))
    configs.sort(key=lambda x: x.data_range[0])

    # Check if metadata matches and data ranges are joinable
    for c1, c2 in zip(configs[:-1], configs[1:]):
        d1 = c1.to_dict()
        d2 = c2.to_dict()
        if d1.pop("data_range")[1] != d2.pop("data_range")[0]:
            return
        if d1 != d2:
            return

    config = deepcopy(configs[0])
    config.data_range = (configs[0].data_range[0], configs[-1].data_range[1])

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save config info
    d = config.to_dict()
    d["created at"] = datetime.now().isoformat(timespec="seconds")
    with open(out_path / "config.json", "w") as f:
        json.dump(d, f, indent=2)

    json_path = out_path / "out" / "out.jsonl"
    json_path.parent.mkdir()
    with open(json_path, "w") as f:
        for p in paths:
            for in_json_path in (p / "out").iterdir():
                with open(in_json_path, "r") as g:
                    for line in g:
                        f.write(line)

    success_path = out_path / "log" / "_SUCCESS"
    success_path.parent.mkdir()
    success_path.touch()
