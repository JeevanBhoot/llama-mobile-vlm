import base64
import pickle
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import regex as re


def sh(cmd: list[str], input: bytes | None = None) -> None:
    try:
        print(f"$ {' '.join(cmd)}", file=sys.stderr)
        subprocess.run(cmd, check=True, input=input)
    except subprocess.CalledProcessError:
        print(f"ERROR RUNNING $ {cmd}", file=sys.stderr)
        raise


@dataclass
class Job:
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def run(self) -> None:
        self.fn(*self.args, **self.kwargs)

    def save(self) -> bytes:
        return pickle.dumps(self)


def _default_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


@dataclass
class Submission:
    user: str
    project: str
    env: dict[str, str]
    job: Job
    commit: str = field(default_factory=_default_commit)
    gpu_clique: str | None = None
    priority: Literal["", "low", "high"] = ""
    n_gpus: int = 4
    n_cpus_per_gpu: int = 20
    mem_per_cpu: int = 8


def _template_replace(src: Path, dest: Path, replacements: dict[str, str]) -> None:
    script = src.read_text()
    for k, v in replacements.items():
        if k not in script:
            raise ValueError(f"{k} not in template {src}")
        script = script.replace(k, repr(v).replace("'", ""))

    if not_replaced := re.findall(r"__TEMPLATE_.+__", script):
        raise ValueError(f"Un-matched template placeholders: {not_replaced} in {src}")

    dest.write_text(script)


def _generate_yaml(sub: Submission) -> Path:
    template_path = Path(__file__).resolve().parent / "_volt_template.yaml"
    yaml_path = template_path.parent / "job.yaml"

    n_cpus = sub.n_gpus * sub.n_cpus_per_gpu
    replacements = dict(
        __TEMPLATE_USER__=sub.user,
        __TEMPLATE_PROJECT__=sub.project,
        __TEMPLATE_PRIORITY_CLASS__=sub.priority,
        __TEMPLATE_ENV__=[dict(name=k, value=v) for k, v in sub.env.items()]
        + [
            dict(name="GIT_COMMIT", value=sub.commit),
            # Install PyTorch + CUDA
            dict(
                name="PIP_EXTRA_INDEX_URL",
                value="https://download.pytorch.org/whl/cu130",
            ),
        ],
        __TEMPLATE_JOB__=base64.b64encode(sub.job.save()).decode("ascii"),
        __TEMPLATE_NODE_SELECTOR__=(
            {"nvidia.com/gpu.clique": sub.gpu_clique} if sub.gpu_clique else {}
        ),
        __TEMPLATE_GPUS__=sub.n_gpus,
        __TEMPLATE_MEMORY__=f"{n_cpus * sub.mem_per_cpu}Gi",
        __TEMPLATE_CPUS__=n_cpus,
    )
    _template_replace(template_path, yaml_path, replacements)

    return yaml_path


def submit(sub: Submission) -> None:
    path = _generate_yaml(sub)
    try:
        sh(["volt", "kubectl", "create", "-f", str(path)])
    finally:
        path.unlink()
