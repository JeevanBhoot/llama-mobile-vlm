"""Utilities for interacting with quantised models (designed for use with Jupyter)"""

import base64
import copy
import html
import io
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import PIL.Image
import safetensors.torch
import torch
import tqdm
import transformers
import weight_formats.quantisation as Q
import weight_formats.quantisation_training as QT
from torch import Tensor, nn

import train
from utility import LLAMA_PROMPT_TEMPLATES

# Components


@dataclass
class MultiParameterSetModel:
    """Wraps transformers.PreTrainedModel, storing multiple sets parameters on CPU,
    so that they can be restored when needed.
    """

    model: transformers.PreTrainedModel
    params: dict[str, Tensor]

    @classmethod
    def create(cls, model: transformers.PreTrainedModel) -> "MultiParameterSetModel":
        base_params = {
            k: v.detach().to("cpu", copy=True) for k, v in model.named_parameters()
        }
        return cls(model=model, params=dict(base=base_params))

    def select(self, name: str) -> None:
        for key, p in self.model.named_parameters():
            p.data[...] = self.params[name][key].to(p.device)

    @property
    def device(self) -> torch.device:
        (device,) = set(p.device for p in self.model.parameters())
        return device


def load_parameters_from_file(
    model: transformers.PreTrainedModel, checkpoint: Path
) -> dict[str, Tensor]:
    qmodel = copy.deepcopy(model)
    QT.load_convert(qmodel, safetensors.torch.load_file(checkpoint))

    # Convert to plain parameters (for the original/unconverted model)
    params = {}
    with torch.no_grad():
        for name, module in qmodel.named_modules():
            if isinstance(module, QT.Weight):
                params[name] = module().cpu()
            else:
                for k, p in module._parameters.items():
                    params[".".join(name.split(".") + [k])] = None if p is None else p.cpu()
    return params


@dataclass
class Completions:
    image: PIL.Image.Image
    prompt: str
    completions: dict[str, str]

    def _repr_html_(self) -> str:
        f = io.BytesIO()
        thumb = self.image.copy()
        thumb.thumbnail((384, 768))
        thumb.save(f, format="PNG")
        image_b64 = base64.b64encode(f.getvalue()).decode()

        def escape(s: str) -> str:
            return html.escape(s).replace("\n", "\\n")

        completions_html = " ".join(
            [
                f'<p><span style="padding-right: 0.5em; min-width: 7em; display: inline-block; font-weight: bold;">{key}</span>'
                f" {escape(completion)}...</p>"
                for key, completion in self.completions.items()
            ]
        )
        return f"""
        <div style="display: flex; align-items: top;">
            <img src="data:image/png;base64,{image_b64}" style="width: 25em; margin-right: 2em; align-self: flex-start;">
            <div style="font-size: 16px; line-height: 1.8; max-width: 50em;">
                <p style="margin-top: 0em;">&gt; {escape(self.prompt)}</p>
                {completions_html}
            </div>
        </div>
        """


# Top-level


class LlamaVQA:
    def __init__(self):
        model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raw_model = transformers.MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="bfloat16" if self.device.type == "cuda" else "float32",
        ).to(self.device)
        self.model = MultiParameterSetModel.create(raw_model)
        self.processor = transformers.AutoProcessor.from_pretrained(model_name)
        self.prompt_templates = LLAMA_PROMPT_TEMPLATES
        self.images = {}

    def __repr__(self) -> str:
        return (
            f"LlamaVQA(models={list(self.model.params)},"
            f" images={list(self.images)}, prompt_templates={list(self.prompt_templates)})"
        )

    def load_model(self, name: str, checkpoint_name: str) -> None:
        path = Path(__file__).parent / "checkpoints" / f"{checkpoint_name}.safetensors"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            s3_path = train.CHECKPOINT_PATH.format(name=checkpoint_name)
            try:
                subprocess.check_call(["aws", "s3", "cp", s3_path, str(path)])
            except subprocess.CalledProcessError as e:
                raise ValueError(
                    f"checkpoint not found: {checkpoint_name!r} at {path} or {s3_path}"
                ) from e
        self.model.params[name] = load_parameters_from_file(self.model.model, path)

    def load_image(self, name: str, url: str) -> None:
        image = PIL.Image.open(urllib.request.urlopen(url))
        image.load()
        self.images[name] = image

    def __call__(
        self,
        image: str,
        prompt: str,
        *models_and_templates: tuple[str, str],
        n_tokens: int = 40,
        progress: bool = True,
        do_sample: bool = False,
        **args: Any,
    ) -> Completions:
        image_data = self.images[image]
        completions = {}
        if not do_sample:
            args.update(temperature=None, top_p=None)  # avoid warnings
        for model, template in tqdm.tqdm(models_and_templates, disable=not progress):
            self.model.select(model)
            inputs = self.processor(
                image_data,
                self.prompt_templates[template].format(prompt=prompt),
                return_tensors="pt",
                add_special_tokens=False,
            )
            out = self.model.model.generate(
                **inputs.to(self.device),
                max_new_tokens=n_tokens,
                do_sample=do_sample,
                **args,
            )[0, inputs.input_ids.shape[1] :]
            completions[f"{model}[{template}]"] = self.processor.decode(out)
        return Completions(image_data, prompt, completions)
