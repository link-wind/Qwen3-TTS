# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

EMOTION_ALIASES = {
    "happiness": "happy",
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "surprise": "surprised",
    "neutrality": "neutral",
}

EMOTION_KEYS = [
    "emotion",
    "emotion_category",
    "emo",
    "style",
    "label",
]

EMPHASIS_SPAN_KEYS = [
    "emphasis_spans",
    "emphasis_span",
    "emphasis_ranges",
]

VAD_SCORE_KEYS = [
    "vad_score",
    "arousal",
    "intensity_score",
]

VAD_VECTOR_KEYS = [
    "vad_vector",
    "vad_embedding",
    "emotion_embedding",
]


def normalize_emotion(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower()
    return EMOTION_ALIASES.get(key, key)


def first_existing_value(line: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in line and line[k] is not None:
            return line[k]
    return None


def parse_emotion(line: Dict[str, Any]) -> Optional[str]:
    value = first_existing_value(line, EMOTION_KEYS)
    if isinstance(value, dict):
        value = value.get("name") or value.get("label")
    return normalize_emotion(value)


def parse_emphasis_spans(line: Dict[str, Any]) -> List[List[int]]:
    spans = first_existing_value(line, EMPHASIS_SPAN_KEYS)
    if isinstance(spans, list):
        normalized = []
        for item in spans:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                s, e = int(item[0]), int(item[1])
                if e > s:
                    normalized.append([s, e])
        if normalized:
            return normalized

    text = str(line.get("text", ""))
    marker_spans = []
    for match in re.finditer(r"\*[^*]+\*", text):
        start = match.start() + 1
        end = match.end() - 1
        if end > start:
            marker_spans.append([start, end])
    return marker_spans


def parse_vad_value(line: Dict[str, Any]) -> Tuple[Optional[float], Optional[List[float]]]:
    vector_val = first_existing_value(line, VAD_VECTOR_KEYS)
    if isinstance(vector_val, list) and len(vector_val) > 0:
        try:
            vec = [float(x) for x in vector_val]
            return None, vec
        except Exception:
            pass

    scalar_val = first_existing_value(line, VAD_SCORE_KEYS)
    if scalar_val is not None:
        try:
            return float(scalar_val), None
        except Exception:
            pass

    vad = line.get("vad")
    if isinstance(vad, dict):
        if "arousal" in vad:
            try:
                return float(vad["arousal"]), None
            except Exception:
                pass
        if "vector" in vad and isinstance(vad["vector"], list):
            try:
                return None, [float(x) for x in vad["vector"]]
            except Exception:
                pass

    return None, None


def load_thresholds(path: Optional[str]) -> Dict[str, List[float]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out = {}
    for k, v in payload.items():
        if isinstance(v, list) and len(v) == 2:
            t1, t2 = float(v[0]), float(v[1])
            if t2 > t1:
                out[normalize_emotion(k) or str(k)] = [t1, t2]
    return out


def load_neutral_centroid(path: Optional[str]) -> Tuple[Optional[float], Optional[List[float]]]:
    if not path:
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, (int, float)):
        return float(payload), None
    if isinstance(payload, list):
        return None, [float(x) for x in payload]

    if isinstance(payload, dict):
        if "neutral" in payload:
            neutral = payload["neutral"]
            if isinstance(neutral, (int, float)):
                return float(neutral), None
            if isinstance(neutral, list):
                return None, [float(x) for x in neutral]
        if "neutral_scalar" in payload:
            return float(payload["neutral_scalar"]), None
        if "neutral_vector" in payload and isinstance(payload["neutral_vector"], list):
            return None, [float(x) for x in payload["neutral_vector"]]

    return None, None


def euclidean(v1: List[float], v2: List[float]) -> float:
    n = min(len(v1), len(v2))
    if n == 0:
        return 0.0
    return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(n)))


def compute_dist_to_neutral(
    vad_score: Optional[float],
    vad_vector: Optional[List[float]],
    neutral_scalar: Optional[float],
    neutral_vector: Optional[List[float]],
) -> Optional[float]:
    if vad_vector is not None:
        if neutral_vector is not None:
            return euclidean(vad_vector, neutral_vector)
        return math.sqrt(sum(x * x for x in vad_vector))

    if vad_score is not None:
        if neutral_scalar is not None:
            return abs(vad_score - neutral_scalar)
        return abs(vad_score)

    return None


def discretize_intensity(dist: Optional[float], thresholds: List[float]) -> Optional[str]:
    if dist is None:
        return None
    t1, t2 = thresholds
    if dist < t1:
        return "weak"
    if dist < t2:
        return "medium"
    return "strong"
