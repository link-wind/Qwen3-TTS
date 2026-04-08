# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from grpo_rollout import RolloutItem


@dataclass
class RewardWeights:
    ser: float = 1.0
    emo_intensity: float = 1.0
    emphasis: float = 1.0


def reward_ser_accuracy(item: RolloutItem) -> float:
    """Placeholder SER reward.

    If upstream pipeline writes `meta.ser_correct` (0/1), we use it.
    Otherwise returns 0.0.
    """
    meta = item.prompt.meta or {}
    if "ser_correct" in meta:
        return float(meta["ser_correct"])
    return 0.0


def reward_emotion_intensity_fidelity(item: RolloutItem) -> float:
    """Proxy reward from metadata or audio energy.

    Priority:
      1) use `meta.emo_intensity_score` if provided
      2) energy proxy by target intensity bucket
    """
    meta = item.prompt.meta or {}
    if "emo_intensity_score" in meta:
        return float(meta["emo_intensity_score"])

    target = (item.prompt.intensity or "").lower().strip()
    rms = float(item.aux.get("rms", 0.0))

    # weak/medium/strong proxy bins in normalized waveform space
    if target == "weak":
        return 1.0 - min(abs(rms - 0.04) / 0.04, 1.0)
    if target == "medium":
        return 1.0 - min(abs(rms - 0.08) / 0.08, 1.0)
    if target == "strong":
        return 1.0 - min(abs(rms - 0.12) / 0.12, 1.0)

    return 0.0


def reward_emphasis_controllability(item: RolloutItem) -> float:
    """Placeholder emphasis reward.

    If upstream pipeline writes `meta.emphasis_score`, use it.
    Else give weak proxy reward when emphasis exists and energy is non-trivial.
    """
    meta = item.prompt.meta or {}
    if "emphasis_score" in meta:
        return float(meta["emphasis_score"])

    has_emphasis = bool((item.prompt.emphasis or "").strip()) or bool((item.prompt.control_tags or "").find("[emphasis") >= 0)
    if not has_emphasis:
        return 0.0

    rms = float(item.aux.get("rms", 0.0))
    return float(np.clip((rms - 0.03) / 0.09, 0.0, 1.0))


def compose_reward(item: RolloutItem, weights: RewardWeights) -> Dict[str, float]:
    ser = reward_ser_accuracy(item)
    emo_intensity = reward_emotion_intensity_fidelity(item)
    emphasis = reward_emphasis_controllability(item)
    total = (
        weights.ser * ser
        + weights.emo_intensity * emo_intensity
        + weights.emphasis * emphasis
    )
    return {
        "ser": float(ser),
        "emo_intensity": float(emo_intensity),
        "emphasis": float(emphasis),
        "total": float(total),
    }


def group_relative_advantage(group_rewards: List[float]) -> List[float]:
    x = np.asarray(group_rewards, dtype=np.float32)
    if x.size == 0:
        return []
    mu = float(x.mean())
    std = float(x.std() + 1e-6)
    return ((x - mu) / std).tolist()
