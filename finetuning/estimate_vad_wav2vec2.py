# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""Estimate VAD vectors using wav2vec2 / HuBERT regression-style models.

Expected: model outputs a continuous vector (logits) for each utterance.
The script takes the first `vad_dims` from logits (with optional offset).

Example usage (plain text):
  python estimate_vad_wav2vec2.py \
    --model_id <YOUR_WAV2VEC2_OR_HUBERT_REGRESSION_MODEL> \
    --input_jsonl data/ESD/train_raw.jsonl \
    --output_jsonl data/ESD/train_with_vad.jsonl \
    --device cuda:0 \
    --neutral_centroid_out data/ESD/neutral_centroid.json
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from transformers import AutoModelForAudioClassification, AutoProcessor


def _load_audio(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    return wav.astype(np.float32), int(sr)


def _compute_neutral_centroid(rows: List[Dict], neutral_label: str) -> Optional[List[float]]:
    neutral = [r.get("vad_vector") for r in rows if r.get("emotion") == neutral_label]
    neutral = [v for v in neutral if isinstance(v, list) and len(v) > 0]
    if not neutral:
        return None
    arr = np.asarray(neutral, dtype=np.float32)
    return arr.mean(axis=0).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--vad_dims", type=int, default=3)
    parser.add_argument("--vad_offset", type=int, default=0)
    parser.add_argument("--neutral_label", type=str, default="neutral")
    parser.add_argument("--neutral_centroid_out", type=str, default=None)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForAudioClassification.from_pretrained(args.model_id).to(args.device)
    model.eval()

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line.strip()) for line in f if line.strip()]

    for row in rows:
        wav_path = row.get("audio", None)
        if not wav_path:
            raise ValueError("Missing audio path in JSONL")
        wav, sr = _load_audio(wav_path, target_sr=args.target_sr)

        inputs = processor(wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits.squeeze(0)

        if logits.numel() < args.vad_offset + args.vad_dims:
            raise ValueError(
                f"Logits length {logits.numel()} is smaller than required {args.vad_offset + args.vad_dims}"
            )

        vad = logits[args.vad_offset : args.vad_offset + args.vad_dims].detach().cpu().tolist()
        row["vad_vector"] = [float(x) for x in vad]

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.neutral_centroid_out:
        centroid = _compute_neutral_centroid(rows, args.neutral_label)
        if centroid is None:
            raise ValueError("Neutral centroid could not be computed (no neutral samples).")
        os.makedirs(os.path.dirname(args.neutral_centroid_out) or ".", exist_ok=True)
        with open(args.neutral_centroid_out, "w", encoding="utf-8") as f:
            json.dump({"neutral_vector": centroid}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
