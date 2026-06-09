# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import prompt_sampling


def test_get_prompts_returns_expected_count_and_template_wrapping() -> None:
    prompts = prompt_sampling.get_prompts(
        n_prompts=7,
        templates=["<BEGIN>{prompt}<END>"],
        template_weights=[1.0],
        instructions=["INST"],
        prompts=["PROMPT"],
        instruction_prob=0.0,
        seed=123,
    )

    assert len(prompts) == 7
    assert all(x.startswith("<BEGIN>") and x.endswith("<END>") for x in prompts)


def test_get_prompts_is_deterministic_for_fixed_seed() -> None:
    kwargs = dict(
        n_prompts=20,
        templates=["{prompt}"],
        template_weights=[1.0],
        instructions=["INST A", "INST B"],
        prompts=["PROMPT A", "PROMPT B", "PROMPT C"],
        instruction_prob=0.8,
        seed=777,
    )

    first = prompt_sampling.get_prompts(**kwargs)
    second = prompt_sampling.get_prompts(**kwargs)

    assert first == second


def test_instruction_prob_controls_inclusion() -> None:
    no_instr = prompt_sampling.get_prompts(
        n_prompts=40,
        templates=["{prompt}"],
        template_weights=[1.0],
        instructions=["UNIQUE_INSTRUCTION_TOKEN"],
        prompts=["UNIQUE_PROMPT_TOKEN"],
        instruction_prob=0.0,
        seed=1,
    )
    assert all("UNIQUE_INSTRUCTION_TOKEN" not in x for x in no_instr)
    assert all("UNIQUE_PROMPT_TOKEN" in x for x in no_instr)

    with_instr = prompt_sampling.get_prompts(
        n_prompts=40,
        templates=["{prompt}"],
        template_weights=[1.0],
        instructions=["UNIQUE_INSTRUCTION_TOKEN"],
        prompts=["UNIQUE_PROMPT_TOKEN"],
        instruction_prob=1.0,
        seed=1,
    )
    assert all("UNIQUE_INSTRUCTION_TOKEN" in x for x in with_instr)
    assert all("UNIQUE_PROMPT_TOKEN" in x for x in with_instr)


def test_instruction_block_can_include_length_format_and_adherence() -> None:
    settings = prompt_sampling.PromptSamplingSettings(
        format_instructions=["FORMAT_TOKEN"],
        adherence_probes=["ADHERENCE_TOKEN"],
        length_instruction_buckets=[(["LENGTH_TOKEN"], 1.0)],
        format_instruction_prob=1.0,
        adherence_probe_prob=1.0,
        length_instruction_prob=1.0,
    )

    outputs = prompt_sampling.get_prompts(
        n_prompts=1,
        templates=["{prompt}"],
        template_weights=[1.0],
        instructions=["BASE_INST_TOKEN"],
        prompts=["PROMPT_TOKEN"],
        instruction_prob=1.0,
        seed=0,
        sampling_settings=settings,
    )

    out = outputs[0]
    assert "BASE_INST_TOKEN" in out
    assert "PROMPT_TOKEN" in out
    assert "LENGTH_TOKEN" in out
    assert "FORMAT_TOKEN" in out
    assert "ADHERENCE_TOKEN" in out


def test_invalid_template_weights_raise_assertion() -> None:
    try:
        prompt_sampling.get_prompts(
            n_prompts=1,
            templates=["{prompt}", "{prompt}"],
            template_weights=[1.0],
            instructions=["INST"],
            prompts=["PROMPT"],
            instruction_prob=0.5,
            seed=0,
        )
    except AssertionError:
        return

    assert False, "Expected AssertionError for mismatched templates/template_weights"


def test_can_disable_optional_instruction_components() -> None:
    settings = prompt_sampling.PromptSamplingSettings(
        format_instructions=["FORMAT_TOKEN"],
        adherence_probes=["ADHERENCE_TOKEN"],
        length_instruction_buckets=[(["LENGTH_TOKEN"], 1.0)],
        format_instruction_prob=0.0,
        adherence_probe_prob=0.0,
        length_instruction_prob=0.0,
    )

    outputs = prompt_sampling.get_prompts(
        n_prompts=10,
        templates=["{prompt}"],
        template_weights=[1.0],
        instructions=["BASE_INST_TOKEN"],
        prompts=["PROMPT_TOKEN"],
        instruction_prob=1.0,
        seed=4,
        sampling_settings=settings,
    )
    assert all("BASE_INST_TOKEN" in x for x in outputs)
    assert all("PROMPT_TOKEN" in x for x in outputs)
    assert all("FORMAT_TOKEN" not in x for x in outputs)
    assert all("ADHERENCE_TOKEN" not in x for x in outputs)
    assert all("LENGTH_TOKEN" not in x for x in outputs)
