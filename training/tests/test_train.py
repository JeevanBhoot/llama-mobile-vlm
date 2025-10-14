import train
import PIL
from train_data import Datum
import transformers
import torch


def test_tokenise_and_add_mask() -> None:

    # Dummy image
    img = PIL.Image.new("RGB", (32, 32))

    tok = "<|finetune_right_pad_id|>"
    x1 = Datum(
        index=0,
        image=img,
        prompt="<|image|>" + tok * 2,
        out="<|image|>" + tok * 5,
    )
    x2 = Datum(
        index=0,
        image=img,
        prompt="<|image|>" + tok * 3,
        out="<|image|>" + tok * 7,
    )

    processor = transformers.AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct"
    )

    inps = train._tokenise_and_add_mask([x1, x2], processor)
    mask = inps.pop("prompt_mask")

    expected_inps = processor(
        [[img] for _ in range(2)], [x1.out, x2.out], return_tensors="pt", padding=True
    )
    expected_mask = torch.ones_like(expected_inps["attention_mask"])
    # Length prompt + 2 bot tokens
    expected_mask[0, :5] = 0
    expected_mask[1, :6] = 0

    assert inps.keys() == expected_inps.keys()
    for k in inps:
        torch.testing.assert_close(inps[k], expected_inps[k])
    torch.testing.assert_close(mask, expected_mask)
