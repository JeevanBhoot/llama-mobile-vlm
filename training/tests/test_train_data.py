import unittest.mock as um
from pathlib import Path

import torch

import train_data


def test_dataset(tmp_path: Path) -> None:
    # To avoid running multiprocessing
    class InlineProc:
        def __init__(self, target, args):
            self._target, self._args = target, args
            self.exitcode = 0

        def start(self):
            self._target(*self._args)  # run synchronously in-process

        def join(self):  # nothing to join
            pass

    class InlineCtx:
        def Process(self, target, args):
            return InlineProc(target, args)

    config = train_data.GenerationConfig.default(
        split="validation", data_range=(0, 3), n_generated_tokens=1
    )

    # Mock loading the model to speed it up
    dummy = um.Mock()
    dummy.device = "cpu"
    dummy.name_or_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    dummy.generate.return_value = torch.tensor([[0], [0]], device=dummy.device)

    with um.patch("train_data.mp.get_context", return_value=InlineCtx()), um.patch(
        "train_data.transformers.MllamaForConditionalGeneration.from_pretrained",
        return_value=dummy,
    ):
        out_dir = train_data.generate_data(
            config,
            batch_size=2,
            world_size=1,
            data_path=str(tmp_path),
            sync_to_s3=False,
        )

    config_read = train_data.load_config(out_dir)
    assert config == config_read

    # Last batch is dropped as n_examples % batch_size != 0
    assert config.data_range == (0, 2)

    # # Check if joining works
    ds = train_data.Dataset([out_dir] * 2, n_examples=[None] * 2)

    assert len(ds.data) == 4
    for x in ds.get_datums():
        assert isinstance(x, train_data.Datum)
        assert x.out == "!"  # token_idx = 0
