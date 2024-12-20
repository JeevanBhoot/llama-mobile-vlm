import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Union

import torch
import transformers

import training.quantisation.quantisation as Q
import wandb
from training.eval import outcompare

WANDB_PROJECT = "llama-mobile"
logger = logging.getLogger(__name__)


@dataclass
class Task:
    name: str
    n_samples: Optional[int] = None
    metrics: list[str] = field(default_factory=lambda: outcompare.METRICS)


@dataclass
class Execution:
    device: str
    batch_size: int
    wandb: Union[bool, str]  # False | True | "offline"


@dataclass
class Experiment:
    name: str
    model: str
    task: Task
    quantisation: list[Q.ParameterRule]
    execution: Execution
    notes: Optional[str] = None


Results = Dict[str, Any]


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

    # TODO: Move this to utility.py as its also done in eval.outcomapre
    dtype = dict(cpu=torch.float32, cuda=torch.bfloat16)[xp.execution.device]
    if "Llama" in xp.model and "Vision" in xp.model:
        model = transformers.MllamaForConditionalGeneration.from_pretrained(
            xp.model, torch_dtype=dtype, device_map=xp.execution.device
        )
    elif "paligemma" in model:
        model = transformers.PaliGemmaForConditionalGeneration.from_pretrained(
            model, torch_dtype=dtype, device_map=xp.execution.device
        )
    else:
        raise ValueError("Unsupported model")

    # Quantise the model
    # TODO: Change the pattern, add format for vectors
    n_bytes = Q.quantise_model(model, xp.quantisation)

    results = {}
    results["n_params"] = sum(p.nelement() for p in model.parameters())
    results["n_bytes"] = n_bytes

    t0 = time.time()
    if xp.task.name == "outcompare":
        dataset = outcompare.EvalDataset.load(
            f"data/{xp.model.replace('google/', '').replace('meta-llama/', '').lower()}.pt"
        )
        results.update(
            outcompare.evaluate(
                model, dataset, xp.execution.batch_size,
                limit=xp.task.n_samples, metrics=xp.task.metrics,
            )
        )
    else:
        raise ValueError("Task not supported")
    results["duration"] = time.time() - t0

    if xp.execution.wandb:
        wandb.summary.update(results)
        assert wandb.run is not None
        results["wandb"] = dict(
            id=wandb.run.id, name=wandb.run.name, url=wandb.run.get_url()
        )
        wandb.finish()

    return dict(**config, **results)
