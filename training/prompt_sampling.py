import numpy as np

INSTRUCTIONS: list[str] = [
    "Answer in as few words as possible.",
    "Answer in as many words as possible.",
    "Keep the response under 10 words.",
    "Keep the response under 100 words.",
    "Keep the response under 200 words.",
    "Write a single concise sentence.",
    "Write a few-sentence summary only.",
    "Give a short bulleted list.",
    "Give the answer only, no explanation.",
    "Provide a short answer first, then an explanation.",
    "List the two most important visual cues used.",
    "Answer, then ask a follow-up question.",
    "If text appears in the image, transcribe exactly.",
    "If people or animals appear in the image, count them.",
    "If multiple valid answers exist, rank them.",
    "If the answer depends on unreadable details, say 'unclear'.",
    "When answering with a list, sort items alphabetically.",
    "Use simple language when answering.",
]


PROMPTS: list[str] = [
    # --- Descriptive prompts ---
    "Describe the image.",
    "Write an appropriate caption.",
    "Summarize the scene in the image.",
    "Write a news article about the image.",
    "Write a book chapter based on what you see.",
    "Describe the layout from left to right.",
    "Transcribe all legible text.",
    # --- Questions ---
    "What is the main object visible?",
    "What are the main colors appearing in the image?",
    "How many persons or animals do you see?",
    "Is motion implied? How?",
    "What time of day does this appear to be?",
    "Does anything look out of place?",
    "Which features suggest the location or region?",
    "What material or texture stands out most?",
    "Which part of the image draws attention first and why?",
    "What is unclear or ambiguous in this image?",
]


def get_prompts(
    n_prompts: int,
    templates: list[str],
    template_weights: list[float],
    instructions: list[str],
    prompts: list[str],
    instruction_prob: float,
    seed: int | None,
) -> list[str]:
    assert len(templates) == len(template_weights)
    assert 0 <= instruction_prob <= 1

    rng = np.random.default_rng(seed)

    t_idxs = rng.choice(len(templates), size=n_prompts, p=template_weights)
    i_idxs = rng.integers(len(instructions), size=n_prompts)
    p_idxs = rng.integers(len(prompts), size=n_prompts)
    use_instruction = rng.random(size=n_prompts) > 1 - instruction_prob

    return [
        templates[t_idx].format(
            prompt=(
                f"{instructions[i_idx]}\n{prompts[p_idx]}\n"
                if instr_flag
                else f"{prompts[p_idx]}\n"
            )
        )
        for t_idx, i_idx, p_idx, instr_flag in zip(
            t_idxs, i_idxs, p_idxs, use_instruction
        )
    ]
