import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import torch
import transformers
import wandb

import quantisation as Q
from eval import outcompare, vqa

WANDB_PROJECT = "llama-mobile"
logger = logging.getLogger(__name__)


@dataclass
class Task:
    name: str
    n_examples: Optional[int]
    metrics: list[str]

    @classmethod
    def outcompare(
        cls, n_examples: Optional[int] = None, metrics=outcompare.METRICS
    ) -> "Task":
        return cls(name="outcompare", n_examples=n_examples, metrics=metrics)

    @classmethod
    def vqa(cls, n_examples: int = 1000) -> "Task":
        return cls(name="vqa", n_examples=n_examples, metrics=["accuracy"])


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

    # TODO: Move this to utility.py as its also done in eval.outcompare
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
    processor = transformers.AutoProcessor.from_pretrained(xp.model)

    # Quantise the model
    n_bytes = Q.quantise_model(model, xp.quantisation)

    out = {}
    out["n_params"] = sum(p.nelement() for p in model.parameters())
    out["n_bytes"] = n_bytes

    t0 = time.time()
    if xp.task.name == "outcompare":
        dataset = outcompare.EvalDataset.load(
            f"data/{xp.model.replace('google/', '').replace('meta-llama/', '').lower()}.pt"
        )
        out.update(
            outcompare.evaluate(
                model,
                dataset,
                xp.execution.batch_size,
                limit=xp.task.n_examples,
                metrics=xp.task.metrics,
            )
        )
    elif xp.task.name == "vqa":
        results = list(
            vqa.evaluate(
                model=model,
                processor=processor,
                examples=vqa.VQA.get_examples(),
                batch_size=xp.execution.batch_size,
                n_examples=xp.task.n_examples,
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
