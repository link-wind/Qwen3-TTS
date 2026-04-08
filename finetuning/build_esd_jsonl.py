# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""Build control-conditioned JSONL from ESD dataset.

Example usage (plain text, not executed here):
  python build_esd_jsonl.py \
    --esd_root "data/ESD/raw/ESD/Emotional Speech Dataset (ESD)" \
    --output_jsonl data/ESD/train_raw.jsonl \
    --speakers 0001,0002 \
    --splits train \
    --use_absolute_paths
"""

import argparse
import json
import os
from typing import Dict, List, Optional


EMOTION_DIR_TO_LABEL = {
    "Neutral": "neutral",
    "Happy": "happy",
    "Angry": "angry",
    "Sad": "sad",
    "Surprise": "surprised",
}


def _read_text_file(path: str) -> List[str]:
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read().splitlines()
        except Exception:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().splitlines()


def _parse_transcripts(txt_path: str) -> Dict[str, str]:
    mapping = {}
    for line in _read_text_file(txt_path):
        parts = [p for p in line.strip().split("\t") if p]
        if len(parts) >= 2:
            utt_id = parts[0].strip()
            text = parts[1].strip()
            if utt_id and text:
                mapping[utt_id] = text
    return mapping


def _pick_ref_audio(speaker_dir: str, splits: List[str]) -> Optional[str]:
    for split in splits:
        neutral_dir = os.path.join(speaker_dir, "Neutral", split)
        if os.path.isdir(neutral_dir):
            wavs = [f for f in os.listdir(neutral_dir) if f.lower().endswith(".wav")]
            if wavs:
                wavs.sort()
                return os.path.join(neutral_dir, wavs[0])
    return None


def _iter_speaker_dirs(esd_root: str, speakers: Optional[List[str]]) -> List[str]:
    entries = [d for d in os.listdir(esd_root) if os.path.isdir(os.path.join(esd_root, d))]
    entries.sort()
    if speakers:
        sset = set(speakers)
        entries = [d for d in entries if d in sset]
    return [os.path.join(esd_root, d) for d in entries]


def build_jsonl(esd_root: str, output_jsonl: str, speakers: Optional[List[str]], splits: List[str], use_absolute_paths: bool):
    rows = []
    for speaker_dir in _iter_speaker_dirs(esd_root, speakers):
        speaker_id = os.path.basename(speaker_dir)
        transcript_path = os.path.join(speaker_dir, f"{speaker_id}.txt")
        transcript_map = _parse_transcripts(transcript_path)
        ref_audio = _pick_ref_audio(speaker_dir, splits)
        if ref_audio is None:
            raise ValueError(f"Cannot find ref_audio for speaker {speaker_id} under splits={splits}")

        for emo_dir, emo_label in EMOTION_DIR_TO_LABEL.items():
            for split in splits:
                wav_dir = os.path.join(speaker_dir, emo_dir, split)
                if not os.path.isdir(wav_dir):
                    continue
                for wav_name in sorted(os.listdir(wav_dir)):
                    if not wav_name.lower().endswith(".wav"):
                        continue
                    utt_id = os.path.splitext(wav_name)[0]
                    text = transcript_map.get(utt_id, None)
                    if not text:
                        continue
                    wav_path = os.path.join(wav_dir, wav_name)
                    item = {
                        "audio": os.path.abspath(wav_path) if use_absolute_paths else wav_path,
                        "text": text,
                        "ref_audio": os.path.abspath(ref_audio) if use_absolute_paths else ref_audio,
                        "emotion": emo_label,
                        "speaker": speaker_id,
                        "language": "Chinese" if speaker_id <= "0010" else "English",
                    }
                    rows.append(item)

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esd_root", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--speakers", type=str, default=None, help="Comma-separated speaker ids, e.g. 0001,0002")
    parser.add_argument("--splits", type=str, default="train", help="Comma-separated splits, e.g. train,test")
    parser.add_argument("--use_absolute_paths", action="store_true", default=False)
    args = parser.parse_args()

    speakers = [s.strip() for s in args.speakers.split(",")] if args.speakers else None
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    build_jsonl(
        esd_root=args.esd_root,
        output_jsonl=args.output_jsonl,
        speakers=speakers,
        splits=splits,
        use_absolute_paths=args.use_absolute_paths,
    )


if __name__ == "__main__":
    main()
