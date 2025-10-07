import gc
import itertools as it
import json
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

import datasets
import torch
import torch.distributed as dist
import torch.distributed.fsdp as fsdp
import torch.multiprocessing as mp
import transformers
import weight_formats.quantisation as Q
import weight_formats.quantisation_training as QT
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from weight_formats.experiments.qat import _compute_kl_loss

import wandb
from eval import vqa
from train_data import Dataset, Datum, GenerationConfig
from utility import (
    LOCAL_DATA_PATH,
    check_s3_access,
    distributed_batches,
    get_unsharded_quantised_params,
    record_memory,
    save_params_to_s3,
)

WANDB_PROJECT = "llama-mobile"
CHECKPOINT_PATH = (
    "s3://graphcore-research/2024-10-squashedllama/checkpoints/{name}.safetensors"
)


def _log(*msg: Any) -> None:
    """Log message to stderr (rank 0 only)."""
    if dist.get_rank() == 0:
        print(*msg, flush=True, file=sys.stderr)


@dataclass
class DataShard:
    path: str
    n_examples: int | None
    config: GenerationConfig = field(init=False)

    def __post_init__(self):
        config_path = f"{LOCAL_DATA_PATH}/{self.path}/config.json"
        with open(config_path) as f:
            self.config = json.loads(f.read())


@dataclass
class DataSettings:
    train: list[DataShard]
    validation: list[DataShard] | None


@dataclass
class OptimiserSettings:
    lr: float
    betas: tuple[float] = (0.9, 0.95)
    weight_decay: float = 0.0


@dataclass
class LRScheduleSettings:
    type: str = "cosine"
    n_warmup_steps: int = 0


@dataclass
class TrainingSettings:
    n_steps: int
    batch_size: int
    optimiser: OptimiserSettings
    lr_schedule: LRScheduleSettings


@dataclass
class ExecutionSettings:
    world_size: int = torch.cuda.device_count()
    params_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    teacher_dtype: str = "bfloat16"
    compile: str | None = "default"
    recomputation: bool = True
    wrap_teacher: bool = True


@dataclass
class QuantisationSettings:
    fmt: Q.TensorFormat
    scaling_mode: QT.ScalingMode = "dynamic"
    clip_gradient: bool = False
    trainable_centroids: bool = False


@dataclass
class Task:
    name: str
    n_examples: int


@dataclass
class Settings:
    run_name: str
    model_name: str
    data: DataSettings
    quantisation: QuantisationSettings | None
    training: TrainingSettings
    execution: ExecutionSettings
    wandb: bool
    downstream_tasks: list[Task] | None
    memory_profile: bool
    save_checkpoint: bool

    @classmethod
    def default(cls) -> "Settings":
        return cls(
            run_name="test",
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
            data=DataSettings(
                train=[
                    DataShard("imagenet-train-generation/870805", None),
                    DataShard("imagenet-train-generation/98cf95", None),
                ],
                validation=[
                    DataShard("coco-validation-generation/9ba9e8", 128),
                    DataShard("coco-validation-generation/76bcc0", 512),
                ],
            ),
            quantisation=QuantisationSettings(
                fmt=Q.LinearScalingFormat(
                    Q.parse("E0M2"),
                    scale_format=Q.parse("BFLOAT16"),
                    block_shape=(1, 64),
                    scaling="absmax",
                )
            ),
            training=TrainingSettings(
                n_steps=16,
                batch_size=64,
                optimiser=OptimiserSettings(lr=2**-16),
                lr_schedule=LRScheduleSettings(),
            ),
            execution=ExecutionSettings(),
            wandb=True,
            downstream_tasks=[Task("vqa", 1024)],
            memory_profile=False,
            save_checkpoint=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compile_model(model: MllamaForConditionalGeneration, mode: str) -> None:
    transformer_modules = [
        model.vision_model.transformer,
        model.vision_model.global_transformer,
        model.language_model.model,
    ]

    for module in transformer_modules:
        for layer_id, layer in module.layers.named_children():
            layer = torch.compile(layer, mode=mode)
            module.layers.register_module(layer_id, layer)


def _apply_fsdp(model: MllamaForConditionalGeneration, **kwargs) -> None:
    transformer_modules = [
        model.vision_model.transformer,
        model.vision_model.global_transformer,
        model.language_model.model,
    ]
    for module in transformer_modules:
        for layer in module.layers:
            fsdp.fully_shard(layer, **kwargs)

    emb_layers = [model.language_model.model.embed_tokens, model.language_model.lm_head]
    for layer in emb_layers:
        fsdp.fully_shard(layer, **kwargs)

    fsdp.fully_shard(model, **kwargs)


def _quantise(
    model: torch.nn.Module, settings: QuantisationSettings
) -> torch.nn.Module:
    QT.convert(
        model,
        fmt_spec=settings.fmt,
        scaling_mode=settings.scaling_mode,
        clip_gradient=settings.clip_gradient,
        error_weight=None,
    )
    if not settings.trainable_centroids:
        for _, p in QT.get_named_parameters(model, "centroids"):
            p.requires_grad_(False)

    return model


def run_validation(
    teacher: MllamaForConditionalGeneration,
    student: MllamaForConditionalGeneration,
    processor: MllamaProcessor,
    batches: Iterable[list[Datum]],
) -> Optional[float]:
    """Return total validation loss to rank 0"""

    device = torch.get_default_device()
    loss = torch.tensor(0.0, dtype=torch.float32)
    n_tokens = torch.tensor(0, dtype=torch.int64)
    with torch.no_grad():
        for batch in batches:
            imgs = [[x.image] for x in batch]
            texts = [x.out for x in batch]
            inps = processor(imgs, texts, return_tensors="pt", padding=True).to(device)
            loss += _compute_kl_loss(student, teacher, inps).item()
            n_tokens += inps["attention_mask"].sum()

    if torch.distributed.is_initialized():
        dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(n_tokens, dst=0, op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        return loss.item() / n_tokens.item()


def run_downstream(
    model: MllamaForConditionalGeneration,
    processor: MllamaProcessor,
    tasks: list[Task],
    local_batch_size: int = 16,
) -> Optional[dict[str, Any]]:
    rank, world_size = 0, 1
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    results = {}
    for task in tasks:
        if task.name == "vqa":
            data = vqa.VQA.data(limit=task.n_examples).shard(
                num_shards=world_size, index=rank
            )
            out = list(vqa.evaluate(model, processor, data, local_batch_size))
            acc = torch.tensor([x["accuracy"] for x in out]).mean()
            dist.all_reduce(acc, dist.ReduceOp.AVG)
            results["vqa"] = dict(accuracy=acc)
        else:
            # TODO: Reasonable default data mix for outcompare
            raise NotImplementedError

    if rank == 0:
        return results


def load_dataset(data_shards: list[DataShard]) -> Dataset:
    paths = [f"{LOCAL_DATA_PATH}/{x.path}" for x in data_shards]
    n_examples = [x.n_examples for x in data_shards]
    return Dataset(paths, n_examples)


def fsdp_train(rank: int, init_method: str, settings: Settings) -> None:
    if settings.save_checkpoint:
        check_s3_access()
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        world_size=settings.execution.world_size,
        rank=rank,
    )
    total_t, total_val_t = None, None
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank != 0:
            transformers.utils.logging.disable_progress_bar()
            datasets.utils.logging.disable_progress_bar()

        if settings.wandb and rank == 0:
            config = settings.to_dict()
            config["name"] = settings.run_name
            config["n_devices"] = world_size
            mode = "offline" if settings.wandb == "offline" else "online"
            run = wandb.init(
                config=config,
                mode=mode,
                entity="graphcore",
                project=WANDB_PROJECT,
            )

        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        torch.set_default_device(device)

        processor = transformers.AutoProcessor.from_pretrained(settings.model_name)

        with record_memory() if settings.memory_profile else nullcontext():
            # Original model
            teacher = transformers.MllamaForConditionalGeneration.from_pretrained(
                settings.model_name,
                torch_dtype=getattr(torch, settings.execution.teacher_dtype),
                device_map="cpu" if settings.execution.wrap_teacher else device,
            )
            for p in teacher.parameters():
                p.requires_grad_(False)

            teacher.eval()

            if settings.execution.wrap_teacher:
                _apply_fsdp(teacher)

            # Quantised model
            student = transformers.MllamaForConditionalGeneration.from_pretrained(
                settings.model_name,
                torch_dtype=getattr(torch, settings.execution.params_dtype),
                device_map="cpu",
            )

            # NOTE: Dropout should be set to 0 by the config
            student.train()

            if settings.quantisation is not None:
                student = _quantise(student, settings.quantisation)
                if settings.wandb and rank == 0:
                    # TODO: This is incorrect for compressed formats
                    n_bits = QT.count_bits(
                        student,
                        compute_dtype=getattr(torch, settings.execution.compute_dtype),
                    )
                    run.config["n_bits"] = n_bits

            if settings.execution.recomputation:
                student.gradient_checkpointing_enable({"use_reentrant": False})

            _apply_fsdp(
                student,
                mp_policy=fsdp.MixedPrecisionPolicy(
                    param_dtype=getattr(torch, settings.execution.compute_dtype)
                ),
            )

            if settings.execution.compile:
                torch._dynamo.config.cache_size_limit = 64
                teacher = torch.compile(
                    teacher,
                    mode=settings.execution.compile,
                    fullgraph=not settings.execution.wrap_teacher,
                )
                _compile_model(student, mode=settings.execution.compile)

            train_data = load_dataset(settings.data.train)

            opt = torch.optim.AdamW(
                student.parameters(),
                lr=settings.training.optimiser.lr,
                betas=settings.training.optimiser.betas,
                weight_decay=settings.training.optimiser.weight_decay,
            )
            lr_scheduler = transformers.get_scheduler(
                settings.training.lr_schedule.type,
                opt,
                num_warmup_steps=settings.training.lr_schedule.n_warmup_steps,
                num_training_steps=settings.training.n_steps,
            )

            batches = it.islice(
                distributed_batches(
                    train_data.get_datums(),
                    settings.training.batch_size,
                    rank=rank,
                    world_size=world_size,
                ),
                settings.training.n_steps,
            )

            if settings.data.validation:
                val_data = load_dataset(settings.data.validation)
                # Assuming n_validation_steps == n_training_steps
                # TODO: Make this flexible
                n_val_steps = len(val_data.data) // settings.training.batch_size
                val_batches = list(
                    distributed_batches(
                        val_data.get_datums(),
                        settings.training.batch_size,
                        rank=rank,
                        world_size=world_size,
                    )
                )

            _log("training")
            total_t0 = time.time()
            total_val_t = 0.0
            total_n_toks = 0
            for step, batch in tqdm(
                enumerate(batches), total=settings.training.n_steps, disable=bool(rank)
            ):
                # TODO: Make this flexible
                if settings.data.validation and step % n_val_steps == 0:
                    t0 = time.time()
                    val_loss = run_validation(teacher, student, processor, val_batches)
                    val_step_t = time.time() - t0
                    total_val_t += val_step_t
                else:
                    val_loss = None

                t0 = time.time()

                imgs = [[x.image] for x in batch]
                texts = [x.out for x in batch]
                inps = processor(imgs, texts, return_tensors="pt", padding=True).to(
                    device
                )

                opt.zero_grad()

                loss = _compute_kl_loss(student, teacher, inps)

                loss.backward()
                opt.step()
                lr_scheduler.step()

                # Log step data
                out = {}
                total_loss = loss.detach().clone()
                dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
                with torch.no_grad():
                    n_toks = inps["attention_mask"].sum()
                    dist.reduce(n_toks, dst=0, op=dist.ReduceOp.SUM)
                    total_n_toks += n_toks.item()
                if settings.wandb and rank == 0:
                    out["train/tokens"] = total_n_toks
                    out["train/toks_per_step"] = n_toks.item()
                    out["perf/step_time"] = time.time() - t0
                    out["perf/toks_per_s"] = (
                        out["train/toks_per_step"] / out["perf/step_time"]
                    )
                    out["train/loss"] = total_loss.item() / n_toks.item()
                    if val_loss:
                        out["val/loss"] = val_loss
                        out["perf/val_step_time"] = val_step_t
                    wandb.log(out, step=step)
            total_t = time.time() - total_t0

            del opt, teacher
            gc.collect()
            torch.cuda.empty_cache()

            # Unshard model in either case
            if settings.downstream_tasks or settings.save_checkpoint:
                params = get_unsharded_quantised_params(
                    student, dtype=getattr(torch, settings.execution.compute_dtype)
                )
                del student
                gc.collect()
                torch.cuda.empty_cache()

            if settings.save_checkpoint:
                _log("save checkpoint")
                path = None
                if rank == 0:
                    path = CHECKPOINT_PATH.format(
                        name=run.name if settings.wandb else settings.run_name
                    )
                save_params_to_s3(params, path)

            if settings.downstream_tasks:
                _log("downstream tasks")
                eval_model = (
                    transformers.MllamaForConditionalGeneration.from_pretrained(
                        settings.model_name,
                        torch_dtype=settings.execution.compute_dtype,
                        device_map=device,
                    )
                )
                QT.load_convert(eval_model, params)
                results = run_downstream(
                    eval_model, processor, tasks=settings.downstream_tasks
                )
                if settings.wandb and rank == 0:
                    run.summary["downstream"] = results

    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        tb = traceback.format_exc()
        if rank == 0:
            _log(error_type, error_message, "\n", tb)
            if settings.wandb:
                run.summary["error_type"] = error_type
                run.summary["error_message"] = error_message
                run.summary["traceback"] = tb
    finally:
        if settings.wandb and rank == 0:
            run.summary["total_time"] = total_t
            run.summary["total_val_time"] = total_val_t
            wandb.finish(1 if "error_type" in run.summary.keys() else 0)
        dist.destroy_process_group()


def run_experiment(settings: Settings) -> None:
    try:
        mp.spawn(
            fsdp_train,
            args=("tcp://localhost:12345", settings),
            nprocs=settings.execution.world_size,
            join=True,
        )
    except Exception as e:
        print(e, file=sys.stderr, flush=True)
        traceback.print_exc()
