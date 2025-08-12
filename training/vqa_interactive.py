"""Utilities for interacting with quantised models (designed for use with Jupyter)"""

import base64
import html
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import PIL.Image
import safetensors.torch
import torch
import tqdm
import transformers
import weight_formats.quantisation as Q
import weight_formats.quantisation_training as QT
from torch import Tensor, nn

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


def load_parameter(
    name: str, fmt: Q.TensorFormat, checkpoint: dict[str, Tensor], device: torch.device
) -> Tensor:
    if name in checkpoint:
        return checkpoint[name].to(device)
    master = checkpoint[f"{name}.master"]
    q_weight = QT.Weight(
        master,
        fmt,
        scaling_mode="parameter" if f"{name}.scale" in checkpoint else "dynamic",
        clip_gradient=False,
    )
    q_weight.to(device)
    for key in ["master", "scale", "sparse_idx", "sparse_weight"]:
        full_key = f"{name}.{key}"
        assert (full_key in checkpoint) == hasattr(q_weight, key), full_key
        if full_key in checkpoint:
            setattr(q_weight, key, nn.Parameter(checkpoint[full_key].to(device)))
    with torch.no_grad():
        return q_weight()


def load_parameters_from_file(
    model: transformers.PreTrainedModel, fmt: Q.TensorFormat, checkpoint: Path
) -> dict[str, Tensor]:
    (device,) = set(p.device for p in model.parameters())
    f = safetensors.torch.load_file(checkpoint)
    return {
        key: load_parameter(key, fmt, f, device).cpu()
        for key, _ in model.named_parameters()
    }


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
        self.prompt_templates = dict(
            instruct="<|start_header_id|>user<|end_header_id|>"
            "\n\n<|image|>{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>",
            simple="<|image|>{prompt}",
        )
        self.images = {}

    def __repr__(self) -> str:
        return (
            f"LlamaVQA(models={list(self.model.params)},"
            f" images={list(self.images)}, prompt_templates={list(self.prompt_templates)})"
        )

    def load_model(self, name: str, checkpoint_name: str, fmt: Q.TensorFormat) -> None:
        path = Path(__file__).parent / "checkpoints" / f"{checkpoint_name}.safetensors"
        if not path.exists():
            raise ValueError(f"checkpoint not found: {checkpoint_name} at {path}")
        self.model.params[name] = load_parameters_from_file(self.model.model, fmt, path)

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
    ) -> Completions:
        image_data = self.images[image]
        completions = {}
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
                do_sample=False,
                temperature=None,
                top_p=None,
            )[0, inputs.input_ids.shape[1] :]
            completions[f"{model}[{template}]"] = self.processor.decode(out)
        return Completions(image_data, prompt, completions)
