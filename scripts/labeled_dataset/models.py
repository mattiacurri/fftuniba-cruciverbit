"""
Pydantic models for structured output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ClueLabel(str, Enum):
    """Possible multi-labels for an Italian crossword clue."""
    DEFINITION = "definition"
    WORDPLAY = "wordplay"
    CRYPTIC = "cryptic"
    FILLING_THE_BLANK = "filling_the_blank"


# ---------- Gemini Structured Output JSON Schema ----------

BATCH_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "0-based index of the clue in the batch.",
                    },
                    "labels": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [l.value for l in ClueLabel],
                        },
                        "description": "One or more labels that apply to this clue.",
                    },
                },
                "required": ["index", "labels"],
            },
        },
    },
    "required": ["classifications"],
}


# ---------- Internal dataclasses ----------

@dataclass
class ClueClassification:
    """Result of classifying a single clue."""
    clue: str
    answer: str
    answer_length: int
    labels: list[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Full result of a classification run."""
    total: int = 0
    classified: int = 0
    failed: int = 0
    items: list[ClueClassification] = field(default_factory=list)
