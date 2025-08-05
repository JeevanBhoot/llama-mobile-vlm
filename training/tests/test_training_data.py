import training_data


def test_generate_dataset() -> None:
    # NOTE: Slow test as it loads Llama-3.2-11B-Vision!
    ds = training_data.generate_dataset(
        limit=2, n_generated_tokens=1, batch_size=2, world_size=1
    )

    assert ds.model_name == "meta-llama/Llama-3.2-11B-Vision-Instruct"
    assert ds.dataset_name == "imagenet"
    assert ds.limit == 2
    assert isinstance(ds._out, dict)
    assert len(ds._out) == 2
    for k, v in ds._out.items():
        assert isinstance(k, int)
        assert isinstance(v, str)

    datums = list(ds.get_datums())
    assert len(datums) == 2
    for datum in datums:
        assert isinstance(datum.index, int)
        assert isinstance(datum.image, training_data.ImageFile)
        assert datum.prompt == f"<|image|>{training_data.ImageNet.PROMPT}"
        assert datum.out == ds._out[datum.index]
