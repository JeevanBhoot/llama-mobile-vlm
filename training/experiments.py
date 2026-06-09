import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
import transformers
import wandb
import weight_formats.quantisation as Q
import weight_formats.quantisation_training as QT

from eval import vqa

WANDB_PROJECT = "llama-mobile"


@dataclass
class Task:
    name: str
    n_examples: Optional[int]
    metrics: list[str]

    @classmethod
    def vqa(cls, n_examples: int = 1000) -> "Task":
        return cls(name="vqa", n_examples=n_examples, metrics=["accuracy"])


@dataclass
class Execution:
    device: str
    batch_size: int
    wandb: bool


@dataclass
class Experiment:
    name: str
    model: str
    task: Task
    quantisation: Q.TensorFormat
    execution: Execution
    notes: Optional[str] = None


Results = dict[str, Any]


def run_experiment(xp: Experiment) -> Results:
    config = asdict(xp)
    notes = config.pop("notes")
    if xp.execution.wandb:
        mode = "offline" if xp.execution.wandb == "offline" else "online"
        wandb.init(
            config=config,
            mode=mode,
            entity="graphcore",
            project=WANDB_PROJECT,
            reinit=True,
            notes=notes,
        )

    dtype = dict(cpu=torch.float32, cuda=torch.bfloat16)[xp.execution.device]
    if "Llama" in xp.model and "Vision" in xp.model:
        model = transformers.MllamaForConditionalGeneration.from_pretrained(
            xp.model, torch_dtype=dtype, device_map=xp.execution.device
        )
    else:
        raise ValueError("Unsupported model")
    processor = transformers.AutoProcessor.from_pretrained(xp.model)

    # Quantise the model
    # TODO: Allow variable quantisation
    QT.convert(
        model,
        fmt_spec=xp.quantisation,
        scaling_mode="dynamic",
        clip_gradient=False,
        error_weight=None,
        activation_fmt=None,
    )
    n_bytes = QT.count_bits(model, torch.bfloat16) / 8

    out = {}
    out["n_params"] = sum(p.nelement() for p in model.parameters())
    out["n_bytes"] = n_bytes

    t0 = time.time()
    if xp.task.name == "outcompare":
        raise NotImplementedError
    elif xp.task.name == "vqa":
        results = list(
            vqa.evaluate(
                model=model,
                processor=processor,
                data=vqa.VQA.data(limit=xp.task.n_examples),
                batch_size=xp.execution.batch_size,
            )
        )
        accuracy = sum(x["accuracy"] for x in results) / len(results)
        out.update(dict(results=results, n_examples=len(results), accuracy=accuracy))
    else:
        raise ValueError("Task not supported")
    out["duration"] = time.time() - t0

    if xp.execution.wandb:
        wandb.summary.update(out)
        assert wandb.run is not None
        out["wandb"] = dict(
            id=wandb.run.id, name=wandb.run.name, url=wandb.run.get_url()
        )
        wandb.finish()

    return dict(**config, notes=notes, **out)
