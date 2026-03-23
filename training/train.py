import gc
import itertools as it
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import datasets
import torch
import torch.distributed as dist
import torch.distributed.fsdp as fsdp
import torch.multiprocessing as mp
import transformers
import wandb
import weight_formats.fit as F
import weight_formats.quantisation as Q
import weight_formats.quantisation_training as QT
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from weight_formats.experiments.qat import _compute_kl_loss

from eval import vqa
from train_data import Dataset, Datum
from utility import (
    S3_REPO_PATH,
    check_s3_access,
    distributed_batches,
    get_unsharded_quantised_params,
    record_memory,
    save_params_to_s3,
)

WANDB_PROJECT = "llama-mobile"
CHECKPOINT_PATH = S3_REPO_PATH + "/checkpoints/{name}.safetensors"


def _log(*msg: Any) -> None:
    """Log message to stderr (rank 0 only)."""
    if dist.get_rank() == 0:
        print(*msg, flush=True, file=sys.stderr)


@dataclass
class DataShard:
    path: str
    n_examples: int | None


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
    validation_interval: int | None = None
    freeze_params: list[str] = field(default_factory=list)


@dataclass
class ExecutionSettings:
    world_size: int | Literal["auto"] = "auto"
    params_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    teacher_dtype: str = "bfloat16"
    compile: str | None = None
    recomputation: bool = True
    wrap_teacher: bool = True


@dataclass
class QuantisationSettings:
    fmt: Q.TensorFormat | F.Scaled
    exclude: list[str] = field(default_factory=list)
    scaling_mode: QT.ScalingMode = "dynamic"
    clip_gradient: bool = False
    trainable_centroids: bool = False
    mode: Literal["qat", "one-shot"] = "qat"


@dataclass
class Task:
    name: str
    n_examples: int


@dataclass
class EvaluationSettings:
    tasks: list[Task]
    batch_size: int
    save_outputs: bool
    include_relaxed_metrics: bool


@dataclass
class Settings:
    run_name: str
    model_name: str
    data: DataSettings
    quantisation: QuantisationSettings | None
    training: TrainingSettings
    execution: ExecutionSettings
    evaluation: EvaluationSettings | None
    wandb: bool
    memory_profile: bool
    save_checkpoint: bool

    @classmethod
    def default(cls) -> "Settings":
        gen_path = "generation/llama-3.2-11b-vision-instruct"
        return cls(
            run_name="test",
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
            data=DataSettings(
                train=[
                    DataShard(
                        f"{gen_path}/imagenet-train/new-prompts-1280k",
                        None,
                    )
                ],
                validation=[DataShard(f"{gen_path}/coco-validation/default-4k", 4096)],
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
                batch_size=128,
                optimiser=OptimiserSettings(lr=2**-17),
                lr_schedule=LRScheduleSettings(),
            ),
            execution=ExecutionSettings(),
            evaluation=EvaluationSettings(
                tasks=[
                    Task("vqa", 1024),
                    Task("chartqa", 1024),
                    Task("docvqa", 1024),
                    Task("ai2d", 1024),
                ],
                batch_size=128,
                save_outputs=True,
                include_relaxed_metrics=True,
            ),
            wandb=True,
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
    fmt_spec = defaultdict(lambda: settings.fmt)
    param_names = [x[0] for x in model.named_parameters()]
    exclude = tuple(settings.exclude)
    if exclude:
        for p_name in param_names:
            if p_name.startswith(exclude):
                fmt_spec[p_name] = Q.TorchFormat("bfloat16")

    QT.convert(
        model,
        fmt_spec=fmt_spec,
        scaling_mode=settings.scaling_mode,
        clip_gradient=settings.clip_gradient,
        error_weight=None,
        mode=settings.mode,
    )
    if not settings.trainable_centroids:
        for _, p in QT.get_named_parameters(model, "centroids"):
            p.requires_grad_(False)

    return model


def _tokenise_and_add_mask(
    batch: list[Datum], processor: MllamaProcessor
) -> dict[str, torch.Tensor]:
    imgs = [[x.image] for x in batch]
    prompts_tok = processor(imgs, [x.prompt for x in batch])["input_ids"]

    # TODO: Save outputs without <bot> tokens
    inp = processor(
        imgs,
        [x.out.replace("<|begin_of_text|>", "") for x in batch],
        return_tensors="pt",
        padding=True,
    )
    mask = torch.ones_like(inp["attention_mask"])
    for i, n in enumerate([len(x) for x in prompts_tok]):
        mask[i, :n] = 0
    inp["prompt_mask"] = mask
    return inp


def run_validation(
    teacher: MllamaForConditionalGeneration,
    student: MllamaForConditionalGeneration,
    processor: MllamaProcessor,
    batches: list[list[Datum]],
) -> Optional[float]:
    """Return total validation loss to rank 0"""
    rank = dist.get_rank()

    loss = torch.tensor(0.0, dtype=torch.float32)
    n_tokens = torch.tensor(0, dtype=torch.int64)
    with torch.no_grad():
        for batch in tqdm(batches, desc="Validation", leave=False, disable=bool(rank)):
            inps = _tokenise_and_add_mask(batch, processor)
            prompt_mask = inps.pop("prompt_mask")
            loss += _compute_kl_loss(student, teacher, inps, prompt_mask)
            n_tokens += (inps["attention_mask"] & prompt_mask).sum()

    if torch.distributed.is_initialized():
        dist.all_reduce(loss)
        dist.all_reduce(n_tokens)

    if rank == 0:
        return loss.item() / n_tokens.item()


def run_downstream(
    model: MllamaForConditionalGeneration,
    processor: MllamaProcessor,
    tasks: list[Task],
    batch_size: int,
    include_relaxed_metrics: bool,
    out_path: Path | str | None,
) -> Optional[dict[str, Any]]:
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1

    if out_path:
        out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)

    results = {}
    for task in tasks:
        if task.name not in vqa.TASKS:
            raise NotImplementedError

        data = (
            vqa.TASKS[task.name]
            .data(limit=task.n_examples)
            .shard(num_shards=world_size, index=rank)
        )
        out = list(
            vqa.evaluate(
                model,
                processor,
                task.name,
                data,
                batch_size // world_size,
                include_relaxed_metrics=include_relaxed_metrics,
                disable_progress=bool(rank),
            )
        )
        if out_path:
            with open(out_path / f"{task.name}.{rank}", "w") as f:
                for x in out:
                    print(json.dumps(x), file=f)
                # Make sure per-rank files are visible at the end
                f.flush()
                os.fsync(f.fileno())

        results[task.name] = {}
        metrics = list(vqa.TASKS[task.name].METRICS)
        if include_relaxed_metrics:
            metrics += vqa.TASKS[task.name].RELAXED_METRICS
        for metric in metrics:
            m = torch.tensor([x[metric] for x in out]).sum()
            n = torch.tensor(len(out), device=m.device)
            if dist.is_initialized():
                dist.all_reduce(m, dist.ReduceOp.SUM)
                dist.all_reduce(n, dist.ReduceOp.SUM)
            results[task.name][metric] = m / n

    # all_reduce should already sync ranks, but just in case
    if dist.is_initialized():
        dist.barrier()

    if rank != 0:
        return

    if not out_path:
        return results

    import fileinput

    # Join shards into per-task jsonl files
    for task in tasks:
        join_path = out_path / f"{task.name}.jsonl"
        parts = (out_path / f"{task.name}.{i}" for i in range(world_size))
        with open(join_path, "w") as f:
            for line in fileinput.input(parts):
                f.write(line)

    return results


def load_dataset(data_shards: list[DataShard]) -> Dataset:
    paths = [x.path for x in data_shards]
    n_examples = [x.n_examples for x in data_shards]
    return Dataset(paths, n_examples)


def _sync_datasets(settings: Settings) -> None:
    import subprocess

    from train_data import IMAGE_DATASETS, load_config
    from utility import LOCAL_DATA_PATH, S3_DATA_PATH

    check_s3_access()

    train = settings.data.train
    val = settings.data.validation if settings.data.validation else []
    for ds in it.chain(train, val):
        local_path = f"{LOCAL_DATA_PATH}/{ds.path}"
        s3_path = f"{S3_DATA_PATH}/{ds.path}"
        subprocess.run(["aws", "s3", "sync", s3_path, local_path], check=True)
        config = load_config(local_path)
        IMAGE_DATASETS[config.dataset_name](split=config.split).data(prompt_fn=None)

    if settings.evaluation:
        for task in settings.evaluation.tasks:
            vqa.TASKS[task.name].data()


def fsdp_train(rank: int, init_method: str, settings: Settings) -> None:
    if settings.save_checkpoint:
        check_s3_access()

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    dist.init_process_group(
        "nccl",
        init_method=init_method,
        world_size=settings.execution.world_size,
        rank=rank,
        device_id=device,
    )
    total_t, total_val_t = None, None
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank != 0:
            transformers.utils.logging.disable_progress_bar()
            datasets.utils.logging.disable_progress_bar()

        obj = [None]
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
            obj[0] = run.id

        _log("Downloading datasets")
        if rank == 0:
            _sync_datasets(settings)

        dist.barrier()

        # Broadcast run id so all ranks can write outputs
        dist.broadcast_object_list(obj, src=0)
        run_id = obj[0]

        processor = transformers.AutoProcessor.from_pretrained(settings.model_name)

        with record_memory() if settings.memory_profile else nullcontext():
            # Original model
            teacher = transformers.MllamaForConditionalGeneration.from_pretrained(
                settings.model_name,
                torch_dtype=getattr(torch, settings.execution.teacher_dtype),
                device_map=device,
            )
            for p in teacher.parameters():
                p.requires_grad_(False)

            teacher.eval()

            if settings.execution.wrap_teacher:
                _apply_fsdp(teacher)

            # Quantised model
            student = transformers.MllamaForConditionalGeneration.from_pretrained(
                settings.model_name,
                torch_dtype=getattr(torch, settings.execution.teacher_dtype),
                device_map=device,
            )
            student.to(dtype=getattr(torch, settings.execution.params_dtype))

            student.train()
            # Sanity check to make sure dropout is disabled
            for m in student.modules():
                if hasattr(m, "dropout"):
                    assert m.dropout == 0

            if settings.quantisation is not None:
                student = _quantise(student, settings.quantisation)
                if settings.wandb and rank == 0:
                    # TODO: This is incorrect for compressed formats
                    n_bits = QT.count_bits(
                        student,
                        compute_dtype=getattr(torch, settings.execution.compute_dtype),
                    )
                    run.config["n_bits"] = n_bits

            # Optionally freeze parameters
            if settings.training.freeze_params:
                freeze_params = tuple(settings.training.freeze_params)
                for p_name, p in student.named_parameters():
                    if p_name.startswith(freeze_params):
                        p.requires_grad_(False)

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
                # By default total num validation steps == total num training steps
                val_interval = settings.training.validation_interval
                if val_interval is None:
                    val_interval = len(val_data.data) // settings.training.batch_size
                val_batches = list(
                    distributed_batches(
                        val_data.get_datums(),
                        settings.training.batch_size,
                        rank=rank,
                        world_size=world_size,
                    )
                )

            total_t0 = time.time()
            total_val_t = 0.0
            total_n_toks = 0
            for step, batch in tqdm(
                enumerate(batches),
                desc="Training",
                total=settings.training.n_steps,
                disable=bool(rank),
            ):
                if settings.data.validation and step % val_interval == 0:
                    t0 = time.time()
                    val_loss = run_validation(teacher, student, processor, val_batches)
                    val_step_t = time.time() - t0
                    total_val_t += val_step_t
                else:
                    val_loss = None

                t0 = time.time()

                inps = _tokenise_and_add_mask(batch, processor)
                prompt_mask = inps.pop("prompt_mask")

                opt.zero_grad()

                n_toks = (inps["attention_mask"] & prompt_mask).sum()
                dist.all_reduce(n_toks)
                loss = _compute_kl_loss(student, teacher, inps, prompt_mask) / n_toks

                loss.backward()
                opt.step()
                lr_scheduler.step()

                # Log step data
                out = {}
                total_n_toks += n_toks.item()
                dist.all_reduce(loss)
                if settings.wandb and rank == 0:
                    out["train/tokens"] = total_n_toks
                    out["train/toks_per_step"] = n_toks.item()
                    out["perf/step_time"] = time.time() - t0
                    out["perf/toks_per_s"] = (
                        out["train/toks_per_step"] / out["perf/step_time"]
                    )
                    out["train/loss"] = loss.item()
                    if val_loss is not None:
                        out["val/loss"] = val_loss
                        out["perf/val_step_time"] = val_step_t
                    wandb.log(out, step=step)
            total_t = time.time() - total_t0

            del opt, teacher
            gc.collect()
            torch.cuda.empty_cache()

            # Unshard model in either case
            if settings.evaluation or settings.save_checkpoint:
                params = get_unsharded_quantised_params(
                    student, dtype=getattr(torch, settings.execution.compute_dtype)
                )
                del student
                gc.collect()
                torch.cuda.empty_cache()

            if settings.evaluation:
                _log("Evaluating downstream tasks")
                out_path = (
                    Path(f"out/evaluation/{run_id}")
                    if settings.evaluation.save_outputs
                    else None
                )
                eval_model = (
                    transformers.MllamaForConditionalGeneration.from_pretrained(
                        settings.model_name,
                        torch_dtype=settings.execution.compute_dtype,
                        device_map=device,
                    )
                )
                QT.load_convert(eval_model, params)
                results = run_downstream(
                    eval_model,
                    processor,
                    tasks=settings.evaluation.tasks,
                    batch_size=settings.evaluation.batch_size,
                    include_relaxed_metrics=settings.evaluation.include_relaxed_metrics,
                    out_path=out_path,
                )
                if settings.wandb and rank == 0:
                    run.summary["downstream"] = results

                    # Save generated outputs as wandb artifact
                    if out_path:
                        artifact = wandb.Artifact(
                            f"task-outputs-{run.id}", "task_outputs"
                        )
                        for task in settings.evaluation.tasks:
                            path = out_path / f"{task.name}.jsonl"
                            if path.exists():
                                artifact.add_file(path)
                        wandb.log_artifact(artifact)

            if settings.save_checkpoint:
                _log("Saving checkpoint")
                path = None
                if rank == 0:
                    path = CHECKPOINT_PATH.format(
                        name=run.name if settings.wandb else settings.run_name
                    )
                save_params_to_s3(params, path)

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
    if settings.execution.world_size == "auto":
        settings.execution.world_size = torch.cuda.device_count()
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
