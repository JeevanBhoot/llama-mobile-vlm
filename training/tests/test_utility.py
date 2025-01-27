from utility import batches


def test_batches() -> None:
    assert list(batches("abcdefg", 3)) == [list("abc"), list("def"), list("g")]
