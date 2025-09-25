import json
from datetime import datetime
from pathlib import Path

import train_data


def test_dataset(tmp_path: Path) -> None:
    # NOTE: Slow test as it calls Llama-11B-Vision
    config = train_data.GenerationConfig.default(
        data_range=(0, 3), n_generated_tokens=1
    )

    out_dir = train_data.generate_data(
        config, batch_size=2, world_size=1, data_path=str(tmp_path)
    )

    with open(f"{out_dir}/config.json") as f:
        config_from_file = json.loads(f.read())
    assert datetime.fromisoformat(config_from_file.pop("created at"))
    orig_config = config.to_dict()
    orig_config.pop("data_range")

    # Last batch is dropped as n_examples % batch_size != 0
    assert config_from_file.pop("data_range") == [0, 2]
    assert orig_config == config_from_file

    # Check if joining works
    ds = train_data.Dataset([out_dir] * 2, n_examples=[None] * 2)

    assert len(ds.data) == 4
    for x in ds.get_datums():
        assert isinstance(x, train_data.Datum)
