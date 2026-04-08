# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""Quick A/B validation script for instruct control (with vs without instruct)."""

import argparse
import csv
import os
import random
import zipfile
from typing import Dict

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_cases():
    return [
        {
            "name": "happy_weak",
            "text": "今天我真的很开心，见到你太好了。",
            "instruct": "请用开心、轻微的语气说这句话，并强调‘真的’。",
        },
        {
            "name": "happy_strong",
            "text": "今天我真的很开心，见到你太好了。",
            "instruct": "请用开心、强烈的语气说这句话，并强调‘真的’。",
        },
        {
            "name": "sad_weak",
            "text": "我很难过，但还是会努力。",
            "instruct": "请用难过、轻微的语气说这句话，并强调‘难过’。",
        },
        {
            "name": "sad_strong",
            "text": "我很难过，但还是会努力。",
            "instruct": "请用难过、强烈的语气说这句话，并强调‘难过’。",
        },
    ]


def _build_gen_kwargs(args) -> Dict:
    """Only override generation kwargs when user explicitly sets them."""
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": (not args.greedy),
    }
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        kwargs["repetition_penalty"] = args.repetition_penalty
    return kwargs


def _audio_health(wav: np.ndarray) -> str:
    if wav is None or len(wav) == 0:
        return "empty"
    if not np.isfinite(wav).all():
        return "nan_or_inf"
    rms = float(np.sqrt(np.mean(np.square(wav))))
    peak = float(np.max(np.abs(wav)))
    if rms < 1e-4:
        return "near_silent"
    if peak > 2.0:
        return f"clipping_peak={peak:.3f}"
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--speaker", type=str, required=True)
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output_dir", type=str, default="ab_instruct_validate")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (not recommended for this model).")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--pack_zip", action="store_true", help="Pack all wav/csv into a zip file.")
    parser.add_argument("--no_flash_attn", action="store_true")
    args = parser.parse_args()

    _set_seed(args.seed)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    attn_impl = None if args.no_flash_attn else "flash_attention_2"
    tts = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=args.device,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation=attn_impl,
    )

    cases = _default_cases()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.csv")

    rows = []
    gen_kwargs = _build_gen_kwargs(args)
    for idx, item in enumerate(cases, start=1):
        case_id = f"case{idx:02d}_{item['name']}"
        text = item["text"]
        instruct = item["instruct"]

        # A: no instruct
        _set_seed(args.seed + idx)
        wavs_a, sr = tts.generate_custom_voice(
            text=text,
            speaker=args.speaker,
            language=args.language,
            instruct="",
            **gen_kwargs,
        )
        out_a = os.path.join(args.output_dir, f"{case_id}__no_instruct.wav")
        sf.write(out_a, wavs_a[0], sr)

        # B: with instruct
        _set_seed(args.seed + idx)
        wavs_b, sr = tts.generate_custom_voice(
            text=text,
            speaker=args.speaker,
            language=args.language,
            instruct=instruct,
            **gen_kwargs,
        )
        out_b = os.path.join(args.output_dir, f"{case_id}__with_instruct.wav")
        sf.write(out_b, wavs_b[0], sr)

        rows.append({
            "case_id": case_id,
            "text": text,
            "instruct": instruct,
            "no_instruct_wav": os.path.basename(out_a),
            "with_instruct_wav": os.path.basename(out_b),
            "no_instruct_health": _audio_health(wavs_a[0]),
            "with_instruct_health": _audio_health(wavs_b[0]),
            "sample_rate": sr,
        })

        print(f"saved: {out_a}")
        print(f"saved: {out_b}")

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "text",
                "instruct",
                "no_instruct_wav",
                "with_instruct_wav",
                "no_instruct_health",
                "with_instruct_health",
                "sample_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {manifest_path}")

    if args.pack_zip:
        zip_path = os.path.join(args.output_dir, "ab_bundle.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name in os.listdir(args.output_dir):
                if name.endswith(".wav") or name.endswith(".csv"):
                    abs_path = os.path.join(args.output_dir, name)
                    zf.write(abs_path, arcname=name)
        print(f"saved: {zip_path}")


if __name__ == "__main__":
    main()
