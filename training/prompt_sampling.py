from dataclasses import dataclass, field

import numpy as np

# General response guidance and image-reasoning instructions.
BASE_INSTRUCTIONS: list[str] = [
    "Answer directly and stay faithful to visible evidence.",
    "Use simple language and avoid jargon.",
    "If details are uncertain, say what is unclear.",
    "Give a concise answer first, then optionally one short explanation.",
    "Prefer concrete visual details over speculation.",
    "If multiple interpretations are plausible, list the top two.",
    "Ground your answer in colors, shapes, or objects you can see.",
    "When helpful, mention the most salient object first.",
    "State uncertainty explicitly instead of guessing.",
    "If text is visible and legible, transcribe it exactly.",
    "If people or animals are present, include an estimated count.",
    "Mention one detail in the foreground and one in the background.",
    "Separate observed facts from uncertain inferences.",
    "Use neutral tone suitable for a dataset annotation.",
    "Keep the answer image-centric and avoid outside world knowledge.",
    "If asked to classify, provide the best class and one runner-up.",
]

# Explicitly probes whether the model follows formatting/control constraints.
ADHERENCE_PROBES: list[str] = [
    "Follow every instruction exactly.",
    "The formatting rule is mandatory.",
    "Do not add extra headers or preambles.",
    "If uncertain, keep the requested format and write 'unclear'.",
    "Do not repeat the question.",
    "Do not include chain-of-thought; only output the requested answer.",
    "Prioritize instruction compliance over verbosity.",
    "Do not output anything after the final line.",
    "Do not include disclaimers.",
]

# Covers short / medium / long outputs to diversify target generation length.
LENGTH_INSTRUCTIONS_SHORT: list[str] = [
    "Reply in exactly one word.",
    "Reply in exactly two words.",
    "Keep the answer under 8 words.",
    "Use a single short sentence.",
]

LENGTH_INSTRUCTIONS_MEDIUM: list[str] = [
    "Use 2-3 sentences.",
    "Write a short paragraph (about 40-80 words).",
    "Provide a concise answer with one brief justification.",
]

LENGTH_INSTRUCTIONS_LONG: list[str] = [
    "Write a detailed paragraph (about 120-180 words).",
    "Provide a longer analysis with multiple visual cues.",
    "Give a structured explanation with 2-3 supporting observations.",
]

FORMAT_INSTRUCTIONS: list[str] = [
    "Begin your answer with 'FINAL ANSWER:'.",
    "Prefix the first line with 'ANSWER:'.",
    "Output a single line only.",
    "Write the answer in ALL CAPS.",
    "Write the answer in lowercase.",
    "Do not use punctuation.",
    "End your answer with '<END>'.",
    "Use a bullet list of exactly three items.",
    "Return valid JSON with keys 'answer' and 'evidence'.",
    "Return YAML with keys 'answer' and 'evidence'.",
    "Wrap the final answer in double quotes.",
]

INSTRUCTIONS: list[str] = BASE_INSTRUCTIONS

# Core prompts that make sense across most natural images.
BASE_PROMPTS: list[str] = [
    "Describe the image.",
    "Write a caption for this image.",
    "Summarize the scene.",
    "Write alt text for accessibility.",
    "Identify the most prominent object.",
    "Name up to five notable objects.",
    "What action, if any, is happening?",
    "What setting does this appear to be in?",
    "What are the dominant colors in the image?",
    "What seems to be the main focus of attention?",
    "What is unclear or ambiguous in this image?",
    "Transcribe any legible text.",
    "Estimate how many people or animals are visible.",
    "What visual detail is most diagnostic of the scene type?",
    "Does anything look unusual or out of place?",
]

PROMPT_ACTIONS: list[str] = [
    "Describe",
    "Summarize",
    "Characterize",
    "Explain",
    "Caption",
]

PROMPT_TARGETS: list[str] = [
    "the main subject",
    "the full scene",
    "the foreground and background",
    "the key objects and their relationships",
    "the likely setting and context",
    "the most informative visual cues",
]

PROMPT_AUDIENCES: list[str] = [
    "for a dataset annotation",
    "for a visually impaired user",
    "for a quick human review",
    "for an image retrieval index",
]

PROMPT_STYLES: list[str] = [
    "using plain language.",
    "without making unsupported assumptions.",
    "with emphasis on observable details.",
    "while explicitly noting uncertainty.",
]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _build_prompt_pool() -> list[str]:
    generated = [
        f"{action} {target} {audience} {style}"
        for action in PROMPT_ACTIONS
        for target in PROMPT_TARGETS
        for audience in PROMPT_AUDIENCES
        for style in PROMPT_STYLES
    ]
    return _dedupe_keep_order(BASE_PROMPTS + generated)


PROMPTS: list[str] = _build_prompt_pool()

PROMPT_WRAPPERS_WITH_INSTRUCTION: list[str] = [
    "{instruction}\n{prompt}\n",
    "Instruction: {instruction}\nTask: {prompt}\n",
    "Follow these instructions exactly.\n{instruction}\n\nTask: {prompt}\n",
    "Task: {prompt}\n\nOutput constraints:\n{instruction}\n",
]

PROMPT_WRAPPERS_PLAIN: list[str] = [
    "{prompt}\n",
    "Task: {prompt}\n",
    "Question: {prompt}\n",
]


def _default_length_instruction_buckets() -> list[tuple[list[str], float]]:
    return [
        (LENGTH_INSTRUCTIONS_SHORT, 0.40),
        (LENGTH_INSTRUCTIONS_MEDIUM, 0.38),
        (LENGTH_INSTRUCTIONS_LONG, 0.22),
    ]


@dataclass
class PromptSamplingSettings:
    format_instructions: list[str] = field(
        default_factory=lambda: FORMAT_INSTRUCTIONS.copy()
    )
    adherence_probes: list[str] = field(default_factory=lambda: ADHERENCE_PROBES.copy())
    length_instruction_buckets: list[tuple[list[str], float]] = field(
        default_factory=_default_length_instruction_buckets
    )
    format_instruction_prob: float = 0.45
    adherence_probe_prob: float = 0.40
    length_instruction_prob: float = 0.75
    instruction_wrapper_prob: float = 0.7
    prompt_wrappers_with_instruction: list[str] = field(
        default_factory=lambda: PROMPT_WRAPPERS_WITH_INSTRUCTION.copy()
    )
    prompt_wrappers_plain: list[str] = field(
        default_factory=lambda: PROMPT_WRAPPERS_PLAIN.copy()
    )

    def validate(self) -> None:
        assert 0 <= self.format_instruction_prob <= 1
        assert 0 <= self.adherence_probe_prob <= 1
        assert 0 <= self.length_instruction_prob <= 1
        assert 0 <= self.instruction_wrapper_prob <= 1
        assert len(self.prompt_wrappers_with_instruction) > 0
        assert len(self.prompt_wrappers_plain) > 0

        if self.length_instruction_buckets:
            assert all(len(bucket) > 0 for bucket, _ in self.length_instruction_buckets)
            assert all(weight >= 0 for _, weight in self.length_instruction_buckets)
            assert sum(weight for _, weight in self.length_instruction_buckets) > 0


def _sample_length_instruction(
    rng: np.random.Generator,
    length_instruction_buckets: list[tuple[list[str], float]],
) -> str:
    weights = np.array([x[1] for x in length_instruction_buckets], dtype=float)
    weights = weights / weights.sum()
    bucket_idx = rng.choice(len(length_instruction_buckets), p=weights)
    bucket = length_instruction_buckets[bucket_idx][0]
    return str(rng.choice(bucket))


def _sample_instruction_block(
    rng: np.random.Generator,
    instructions: list[str],
    sampling_settings: PromptSamplingSettings,
) -> str:
    lines: list[str] = [str(rng.choice(instructions))]

    if (
        sampling_settings.length_instruction_buckets
        and rng.random() < sampling_settings.length_instruction_prob
    ):
        lines.append(
            _sample_length_instruction(
                rng, sampling_settings.length_instruction_buckets
            )
        )

    if (
        sampling_settings.format_instructions
        and rng.random() < sampling_settings.format_instruction_prob
    ):
        lines.append(str(rng.choice(sampling_settings.format_instructions)))

    if (
        sampling_settings.adherence_probes
        and rng.random() < sampling_settings.adherence_probe_prob
    ):
        lines.append(str(rng.choice(sampling_settings.adherence_probes)))

    return "\n".join(_dedupe_keep_order(lines))


def get_prompts(
    n_prompts: int,
    templates: list[str],
    template_weights: list[float],
    instructions: list[str],
    prompts: list[str],
    instruction_prob: float,
    seed: int | None,
    sampling_settings: PromptSamplingSettings | None = None,
) -> list[str]:
    assert len(templates) == len(template_weights)
    assert len(instructions) > 0
    assert len(prompts) > 0
    assert 0 <= instruction_prob <= 1
    if sampling_settings is None:
        sampling_settings = PromptSamplingSettings()
    sampling_settings.validate()

    rng = np.random.default_rng(seed)

    t_idxs = rng.choice(len(templates), size=n_prompts, p=template_weights)
    p_idxs = rng.integers(len(prompts), size=n_prompts)
    use_instruction = rng.random(size=n_prompts) < instruction_prob
    wrap_with_instruction = (
        rng.random(size=n_prompts) < sampling_settings.instruction_wrapper_prob
    )

    outputs: list[str] = []
    for t_idx, p_idx, instr_flag, wrap_flag in zip(
        t_idxs,
        p_idxs,
        use_instruction,
        wrap_with_instruction,
    ):
        prompt_text = prompts[p_idx]

        if instr_flag:
            instruction_block = _sample_instruction_block(
                rng=rng,
                instructions=instructions,
                sampling_settings=sampling_settings,
            )
            if wrap_flag:
                inner_prompt = str(
                    rng.choice(sampling_settings.prompt_wrappers_with_instruction)
                ).format(instruction=instruction_block, prompt=prompt_text)
            else:
                inner_prompt = f"{instruction_block}\n{prompt_text}\n"
        else:
            inner_prompt = str(
                rng.choice(sampling_settings.prompt_wrappers_plain)
            ).format(prompt=prompt_text)

        outputs.append(templates[t_idx].format(prompt=inner_prompt))

    return outputs
