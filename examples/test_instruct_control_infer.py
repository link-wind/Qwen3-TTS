# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""Quick inference test for natural-language instruct control."""

import argparse
import os

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--speaker", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output_dir", type=str, default="instruct_control_tests")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--no_flash_attn", action="store_true")
    args = parser.parse_args()

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

    tests = [
        {
            "name": "happy_weak",
            "text": "今天我真的很开心，见到你太好了。",
            "instruct": "请用开心、轻微的语气说这句话，并强调“真的”。",
        },
        {
            "name": "happy_strong",
            "text": "今天我真的很开心，见到你太好了。",
            "instruct": "请用开心、强烈的语气说这句话，并强调“真的”。",
        },
        {
            "name": "sad_weak",
            "text": "我很难过，但还是会努力。",
            "instruct": "请用难过、轻微的语气说这句话，并强调“难过”。",
        },
        {
            "name": "sad_strong",
            "text": "我很难过，但还是会努力。",
            "instruct": "请用难过、强烈的语气说这句话，并强调“难过”。",
        },
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    for item in tests:
        wavs, sr = tts.generate_custom_voice(
            text=item["text"],
            speaker=args.speaker,
            language="Chinese",
            instruct=item["instruct"],
            max_new_tokens=args.max_new_tokens,
        )
        out_path = os.path.join(args.output_dir, f"{item['name']}.wav")
        sf.write(out_path, wavs[0], sr)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
