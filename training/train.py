import itertools as it
import math
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist
import torch.distributed.fsdp as fsdp
import torch.multiprocessing as mp
import transformers
import weight_formats.quantisation as Q_new
import weight_formats.quantisation_training as QT
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, MllamaProcessor

import wandb
from quantisation import quantisation as Q_old
from quantisation.layers import quantise_linear_layers
from training_data import Dataset, Datum
from utility import (
    check_s3_access,
    compute_kl_loss,
    distributed_batches,
    record_memory,
    save_model_to_s3,
)

# from weight_formats.quantisation_training import (
#     ScalingMode,
#     convert,
#     get_named_parameters,
# )


WANDB_PROJECT = "llama-mobile"
CHECKPOINT_PATH = (
    "s3://graphcore-research/2024-10-squashedllama/checkpoints/{name}.safetensors"
)


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


# NOTE: Temporarily allow both implementations
@dataclass
class QuantisationSettings:
    el_fmt: str
    scale_fmt: str
    block_shape: tuple[int]
    implementation: str  # "old"/"new"
    trainable_centroids: bool


# @dataclass
# class QuantisationSettings:
#     fmt: Q.TensorFormat
#     scaling_mode: ScalingMode
#     clip_gradient: bool


@dataclass
class Settings:
    run_name: str
    model_name: str
    train_dataset: str
    val_dataset: Optional[str]
    quantisation: QuantisationSettings | None
    training: TrainingSettings
    execution: ExecutionSettings
    wandb: bool
    n_val_examples: int
    memory_profile: bool
    save_checkpoint: bool

    @classmethod
    def default(cls) -> "Settings":
        return cls(
            run_name="test",
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
            train_dataset="imagenet",
            val_dataset="coco",
            quantisation=QuantisationSettings(
                el_fmt="E0M2",
                scale_fmt="BFLOAT16",
                block_shape=(1, 64),
                implementation="new",
                trainable_centroids=False,
            ),
            # quantisation=QuantisationSettings(
            #     fmt=Q.LinearScalingFormat(
            #         Q.parse("E5M2"),
            #         scale_format=Q.parse("BFLOAT16"),
            #         block_shape=(1, 32),
            #         scaling="absmax",
            #     ),
            #     scaling_mode="dynamic",
            #     clip_gradient=False,
            # ),
            training=TrainingSettings(
                n_steps=10,
                batch_size=torch.cuda.device_count(),
                optimiser=OptimiserSettings(lr=1e-4),
                lr_schedule=LRScheduleSettings(),
            ),
            execution=ExecutionSettings(),
            wandb=True,
            n_val_examples=512,
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
    if settings.implementation == "new":
        fmt = Q_new.LinearScalingFormat(
            Q_new.parse(settings.el_fmt),
            scale_format=Q_new.parse("BFLOAT16"),
            block_shape=settings.block_shape,
            scaling="absmax",
        )
        QT.convert(
            model,
            fmt,
            scaling_mode="dynamic",
            clip_gradient=False,
            error_weight=None,
        )
        if not settings.trainable_centroids:
            for _, p in QT.get_named_parameters(model, "centroids"):
                p.requires_grad_(False)
    elif settings.implementation == "old":
        fmt = Q_old.LinearScalingFormat(
            Q_old.parse(settings.el_fmt),
            group_shapes=[settings.block_shape],
            scale_format=Q_old.parse(settings.scale_fmt),
            scale_combiner=None,
        )
        model = quantise_linear_layers(model, fmt)
    else:
        print(f"Invalid implementation {settings.implementation}")

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
            loss += compute_kl_loss(student, teacher, inps).item()
            n_tokens += inps["attention_mask"].sum()

    if torch.distributed.is_initialized():
        dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(n_tokens, dst=0, op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        return loss.item() / n_tokens.item()


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
                    # TODO: Get rid of this
                    if settings.quantisation.implementation == "new":
                        n_bits = QT.count_bits(
                            student,
                            compute_dtype=getattr(
                                torch, settings.execution.compute_dtype
                            ),
                        )
                    else:
                        n_bits = None
                    run.config["n_bits"] = n_bits
                # convert(
                #     student,
                #     settings.quantisation.fmt,
                #     scaling_mode=settings.quantisation.scaling_mode,
                #     clip_gradient=settings.quantisation.clip_gradient,
                #     error_weight=None,
                # )
                # # Do not train centroids
                # for _, p in get_named_parameters(student, "centroids"):
                #     p.requires_grad_(False)

            if settings.execution.recomputation:
                student.gradient_checkpointing_enable({"use_reentrant": False})

            _apply_fsdp(
                student,
                mp_policy=fsdp.MixedPrecisionPolicy(
                    param_dtype=getattr(torch, settings.execution.compute_dtype)
                ),
            )

            # TODO: Move compilation before FSDP
            if settings.execution.compile:
                torch._dynamo.config.cache_size_limit = 64
                teacher = torch.compile(
                    teacher,
                    mode=settings.execution.compile,
                    fullgraph=not settings.execution.wrap_teacher,
                )
                _compile_model(student, mode=settings.execution.compile)

            dataset_path = "data/{}-generation/{}.json"
            model_name = settings.model_name.replace("meta-llama/", "").lower()

            data = Dataset.load(dataset_path.format(settings.train_dataset, model_name))

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
                    data.get_datums(),
                    settings.training.batch_size,
                    rank=rank,
                    world_size=world_size,
                ),
                settings.training.n_steps,
            )

            val_data = Dataset.load(
                dataset_path.format(settings.val_dataset, model_name)
            )
            # Assuming n_validation_steps == n_training_steps
            n_val_steps = settings.n_val_examples // settings.training.batch_size

            val_batches = list(
                it.islice(
                    distributed_batches(
                        val_data.get_datums(),
                        settings.training.batch_size,
                        rank=rank,
                        world_size=world_size,
                    ),
                    n_val_steps,
                )
            )

            total_t0 = time.time()
            total_val_t = 0.0
            total_n_toks = 0
            for step, batch in tqdm(
                enumerate(batches), total=settings.training.n_steps
            ):
                # TODO: Change this
                if step % n_val_steps == 0:
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
                loss = compute_kl_loss(student, teacher, inps) / math.prod(
                    inps["attention_mask"].shape
                )

                loss.backward()
                opt.step()
                lr_scheduler.step()

                # Log step data
                out = {}
                out_loss = loss.detach().clone()
                dist.reduce(out_loss, dst=0, op=dist.ReduceOp.AVG)
                with torch.no_grad():
                    n_toks = inps["attention_mask"].sum()
                    dist.reduce(n_toks, dst=0, op=dist.ReduceOp.SUM)
                    total_n_toks += n_toks.item()
                if settings.wandb and rank == 0:
                    out["train/tokens"] = total_n_toks
                    out["perf/step_time"] = time.time() - t0
                    out["perf/toks_per_s"] = n_toks.item() / out["perf/step_time"]
                    out["train/loss"] = out_loss.item()
                    if val_loss:
                        out["val/loss"] = val_loss
                        out["perf/val_step_time"] = val_step_t
                    wandb.log(out, step=step)
            total_t = time.time() - total_t0
            del opt  # save memory
            if settings.save_checkpoint:
                path = None
                if rank == 0:
                    path = CHECKPOINT_PATH.format(
                        name=run.name if settings.wandb else settings.run_name
                    )
                save_model_to_s3(
                    student,
                    path,
                    dtype=getattr(torch, settings.execution.compute_dtype),
                )

    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        tb = traceback.format_exc()
        if rank == 0:
            print(error_type, error_message, file=sys.stderr, flush=True)
            print(tb, file=sys.stderr, flush=True)
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
