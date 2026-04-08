# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

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
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        self.tts = tts
        self.gen_kwargs: Dict[str, Any] = {}
        if max_new_tokens is not None:
            self.gen_kwargs["max_new_tokens"] = int(max_new_tokens)
        if do_sample is not None:
            self.gen_kwargs["do_sample"] = bool(do_sample)
        if temperature is not None:
            self.gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            self.gen_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            self.gen_kwargs["top_k"] = int(top_k)

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
            payload = self.tts._build_controlled_payload_texts(
                [prompt.text],
                emotion=prompt.emotion,
                intensity=prompt.intensity,
                emphasis=prompt.emphasis,
                control_tags=prompt.control_tags,
            )[0]
            input_ids = self.tts._tokenize_texts([self.tts._build_assistant_text(payload)])

            instruct_ids = None
            if prompt.instruct is not None and str(prompt.instruct).strip() != "":
                instruct_ids = [self.tts._tokenize_texts([self.tts._build_instruct_text(str(prompt.instruct))])[0]]

            talker_codes_list, talker_hidden_states_list = self.tts.model.generate(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                languages=[prompt.language or "Auto"],
                speakers=[prompt.speaker],
                non_streaming_mode=True,
                **self.gen_kwargs,
            )
            codes = talker_codes_list[0]
            hidden = talker_hidden_states_list[0]
            wavs, sr = self.tts.model.speech_tokenizer.decode([{"audio_codes": codes}])
            return wavs[0], sr, codes.detach(), hidden.detach()

        if model_type == "voice_design":
            payload = self.tts._build_controlled_payload_texts(
                [prompt.text],
                emotion=prompt.emotion,
                intensity=prompt.intensity,
                emphasis=prompt.emphasis,
                control_tags=prompt.control_tags,
            )[0]
            input_ids = self.tts._tokenize_texts([self.tts._build_assistant_text(payload)])
            instruct_ids = [
                self.tts._tokenize_texts([self.tts._build_instruct_text(prompt.instruct or "")])[0]
            ]

            talker_codes_list, talker_hidden_states_list = self.tts.model.generate(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                languages=[prompt.language or "Auto"],
                non_streaming_mode=True,
                **self.gen_kwargs,
            )
            codes = talker_codes_list[0]
            hidden = talker_hidden_states_list[0]
            wavs, sr = self.tts.model.speech_tokenizer.decode([{"audio_codes": codes}])
            return wavs[0], sr, codes.detach(), hidden.detach()

        raise NotImplementedError("GRPO rollout scaffold currently supports custom_voice/voice_design")

    def sample_group(self, prompts: List[GRPOSample], group_size: int) -> List[RolloutItem]:
        outputs: List[RolloutItem] = []
        for sample_index, prompt in enumerate(prompts):
            for group_index in range(group_size):
                wav, sr, codes, hidden = self._generate_once(prompt)
                aux = self._audio_stats(wav)
                aux["codes"] = codes
                aux["hidden"] = hidden
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
