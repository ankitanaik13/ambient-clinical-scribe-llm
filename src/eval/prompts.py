"""The single prompt template used to elicit SOAP-style clinical notes from the model.

This template is used UNCHANGED in three places:
  - src/eval/baseline_eval.py   (Phase 2, zero-shot)
  - src/train/finetune.py      (Phase 3, SFT targets are formatted with this same template)
  - src/eval/finetuned_eval.py (Phase 4, apples-to-apples comparison against baseline)

Keeping it in one place is what makes the baseline-vs-fine-tuned comparison meaningful: any
difference in ROUGE/LLM-judge scores between Phase 2 and Phase 4 reflects what the model
learned, not a prompt change.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a clinical scribe assistant. Given a transcript of a doctor-patient "
    "conversation, write a concise, structured clinical note documenting the encounter. "
    "Use standard clinical documentation style. Only include information that is stated "
    "or clearly implied in the transcript -- do not invent findings, medications, or "
    "history that were not discussed."
)

USER_PROMPT_TEMPLATE = (
    "Below is a transcript of a doctor-patient conversation. Write the clinical note "
    "section that documents this encounter.\n\n"
    "Transcript:\n"
    "{dialogue}\n\n"
    "Clinical note:"
)


def build_messages(dialogue: str) -> list[dict[str, str]]:
    """Build the chat-format messages for a single dialogue.

    Args:
        dialogue: Raw doctor-patient conversation transcript.

    Returns:
        A list of {"role", "content"} messages suitable for a tokenizer's chat template
        (Llama 3 Instruct format), with the system prompt first and the user turn second.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(dialogue=dialogue.strip())},
    ]
