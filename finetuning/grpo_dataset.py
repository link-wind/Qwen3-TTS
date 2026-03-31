# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset


@dataclass
class GRPOSample:
    text: str
    language: str = "Auto"
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    emotion: Optional[str] = None
    intensity: Optional[str] = None
    emphasis: Optional[str] = None
    control_tags: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class GRPODataset(Dataset):
    """Minimal dataset for GRPO prompts.

    Expected JSONL fields:
      - text (required)
      - language / speaker / instruct (optional)
      - emotion / intensity / emphasis / control_tags (optional)
      - any other fields are preserved in `meta`
    """

    def __init__(self, jsonl_path: str):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line.strip()) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> GRPOSample:
        row = self.rows[idx]
        text = row.get("text", None)
        if not isinstance(text, str) or text.strip() == "":
            raise ValueError(f"Invalid or missing text at index {idx}")

        known = {
            "text", "language", "speaker", "instruct",
            "emotion", "intensity", "emphasis", "control_tags",
        }
        meta = {k: v for k, v in row.items() if k not in known}

        return GRPOSample(
            text=text,
            language=row.get("language", "Auto"),
            speaker=row.get("speaker", None),
            instruct=row.get("instruct", None),
            emotion=row.get("emotion", None),
            intensity=row.get("intensity", None),
            emphasis=row.get("emphasis", None),
            control_tags=row.get("control_tags", None),
            meta=meta,
        )


def grpo_collate_fn(batch: List[GRPOSample]) -> List[GRPOSample]:
    return batch
