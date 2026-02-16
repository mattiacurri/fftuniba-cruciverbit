"""
Gemini API client for structured multi-label classification
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from models import BATCH_CLASSIFICATION_SCHEMA

# ---------- Configuration ----------

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)


MODEL_ID = "gemini-3-flash-preview"

SYSTEM_PROMPT = """\
You are an expert linguist specialised in Italian crossword puzzles (cruciverba).

Your task is to classify each crossword clue into one or more of the following categories:

1. **definition** – The clue is a straightforward semantic definition, description, or synonym of the answer word. Examples: "Capitale d'Italia" → ROMA, "Mammifero marino" → FOCA.

2. **wordplay** – The clue involves a linguistic trick that manipulates the *form* of words or letters rather than (or in addition to) their meaning. This includes:
   - Abbreviations / initials hints ("Le iniziali di Rossi" → RR, "Sigla dell'ONU" → ONU)
   - Letter extraction ("I confini dell'Arkansas" → AS, taking first+last letters)
   - Anagrams, charades (building words from parts), reversals
   - Phonetic puns or homophones
   - Syllabic wordplay, portmanteau clues
   - Clues referencing word parts ("Metà di sole" → SO)

3. **cryptic** – The clue has a deceptive surface reading; the literal sentence seems to refer to something else entirely, and the solver must see through the misdirection. The true meaning is hidden behind an innocent-sounding sentence. A cryptic clue may *also* contain wordplay, but the key feature is the misleading surface.

4. **filling_the_blank** – The clue explicitly asks the solver to complete a famous phrase, title, expression, proverb, or proper name. Typical markers: "___", "...", "…", or formulations like "Completare: ..." or quoting a partial phrase.

**Rules**:
- A clue can have MULTIPLE labels (e.g. both "definition" and "wordplay").
- Every clue must have AT LEAST ONE label.
- Be precise: only assign a label when you are confident it applies.
- Respond ONLY with valid JSON matching the provided schema. Do NOT include reasoning.
"""


def _build_client() -> genai.Client:
    """Instantiate the Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. "
            "Create synth_generator/.env with GEMINI_API_KEY=..."
        )
    return genai.Client(api_key=api_key)


def classify_batch(
    clues: list[dict[str, Any]],
    *,
    client: genai.Client | None = None,
    max_retries: int = 6,
    retry_delay: float = 4.0,
) -> list[dict[str, Any]]:
    """
    Classify a batch of clues via Gemini structured output.

    Parameters
    ----------
    clues : list[dict]
        Each dict has keys "clue", "answer", "answer_length".
    client : genai.Client | None
        Reusable client instance. Created on-the-fly if None.
    max_retries : int
        Number of retries on transient errors.
    retry_delay : float
        Base delay between retries (exponential backoff).

    Returns
    -------
    list[dict]
        One dict per clue with keys: index, labels, reasoning.
    """
    if client is None:
        client = _build_client()

    # Build user message with numbered clues
    lines = []
    for i, c in enumerate(clues):
        answer_info = f" (answer: {c['answer']})" if c.get("answer") else ""
        lines.append(
            f"[{i}] Clue: \"{c['clue']}\" | "
            f"answer_length={c['answer_length']}{answer_info}"
        )
    user_msg = (
        "Classify each of the following Italian crossword clues.\n\n"
        + "\n".join(lines)
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=BATCH_CLASSIFICATION_SCHEMA,
                    temperature=0,
                    max_output_tokens=16384,
                ),
            )

            # Check for truncation
            if hasattr(response, 'candidates') and response.candidates:
                finish = response.candidates[0].finish_reason
                if finish and str(finish) not in ("STOP", "1", "FinishReason.STOP"):
                    raise RuntimeError(
                        f"Response truncated (finish_reason={finish}), "
                        f"got {len(response.text)} chars"
                    )

            result = json.loads(response.text)
            return result.get("classifications", [])

        except Exception as e:
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                print(f"  Attempt {attempt+1} failed: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"  All {max_retries} attempts failed: {e}")
                raise


def create_client() -> genai.Client:
    """Create and return a reusable Gemini client."""
    return _build_client()
