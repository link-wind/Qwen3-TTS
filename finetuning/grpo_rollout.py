# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

from grpo_dataset import GRPOSample


@dataclass
class RolloutItem:
    sample_index: int
    group_index: int
    wav: np.ndarray
    sample_rate: int
    prompt: GRPOSample
    aux: Dict[str, Any]


class GRPORolloutEngine:
    """Minimal rollout engine.

    Notes:
      - This scaffold currently targets `custom_voice` checkpoints first.
      - For `base` / `voice_design`, callers can extend `_generate_once`.
    """

    def __init__(
        self,
        tts: Qwen3TTSModel,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        self.tts = tts
        self.gen_kwargs: Dict[str, Any] = {}
        if max_new_tokens is not None:
            self.gen_kwargs["max_new_tokens"] = int(max_new_tokens)
        if temperature is not None:
            self.gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            self.gen_kwargs["top_p"] = float(top_p)

    def _audio_stats(self, wav: np.ndarray) -> Dict[str, float]:
        x = np.asarray(wav, dtype=np.float32)
        if x.size == 0:
            return {"rms": 0.0, "peak": 0.0, "duration_sec": 0.0}
        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        peak = float(np.max(np.abs(x)))
        return {"rms": rms, "peak": peak}

    def _generate_once(self, prompt: GRPOSample) -> tuple[np.ndarray, int]:
        model_type = self.tts.model.tts_model_type

        if model_type == "custom_voice":
            if prompt.speaker is None:
                raise ValueError("custom_voice rollout requires `speaker` in GRPO sample")
            wavs, sr = self.tts.generate_custom_voice(
                text=prompt.text,
                language=prompt.language,
                speaker=prompt.speaker,
                instruct=prompt.instruct,
                emotion=prompt.emotion,
                intensity=prompt.intensity,
                emphasis=prompt.emphasis,
                control_tags=prompt.control_tags,
                **self.gen_kwargs,
            )
            return wavs[0], sr

        if model_type == "voice_design":
            wavs, sr = self.tts.generate_voice_design(
                text=prompt.text,
                language=prompt.language,
                instruct=prompt.instruct or "",
                emotion=prompt.emotion,
                intensity=prompt.intensity,
                emphasis=prompt.emphasis,
                control_tags=prompt.control_tags,
                **self.gen_kwargs,
            )
            return wavs[0], sr

        raise NotImplementedError("GRPO rollout scaffold currently supports custom_voice/voice_design")

    def sample_group(self, prompts: List[GRPOSample], group_size: int) -> List[RolloutItem]:
        outputs: List[RolloutItem] = []
        for sample_index, prompt in enumerate(prompts):
            for group_index in range(group_size):
                wav, sr = self._generate_once(prompt)
                aux = self._audio_stats(wav)
                outputs.append(
                    RolloutItem(
                        sample_index=sample_index,
                        group_index=group_index,
                        wav=wav,
                        sample_rate=sr,
                        prompt=prompt,
                        aux=aux,
                    )
                )
        return outputs
