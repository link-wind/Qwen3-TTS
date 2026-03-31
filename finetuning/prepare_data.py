# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json

from qwen_tts import Qwen3TTSTokenizer

import control_labeling as cl

BATCH_INFER_NUM = 32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument(
        "--emotion_thresholds_json",
        type=str,
        default=None,
        help="JSON file mapping emotion -> [t1, t2] for intensity discretization.",
    )
    parser.add_argument(
        "--global_thresholds",
        type=str,
        default="0.33,0.66",
        help="Fallback thresholds formatted as 't1,t2' when emotion-specific thresholds are missing.",
    )
    parser.add_argument(
        "--neutral_centroid_json",
        type=str,
        default=None,
        help="JSON file of neutral centroid. Supports scalar or vector (see README).",
    )
    parser.add_argument(
        "--keep_existing_intensity",
        action="store_true",
        help="If line already has intensity, keep it instead of recomputing.",
    )
    args = parser.parse_args()

    global_thresholds = [float(x.strip()) for x in args.global_thresholds.split(",")]
    if len(global_thresholds) != 2 or global_thresholds[1] <= global_thresholds[0]:
        raise ValueError("--global_thresholds must be 't1,t2' with t2 > t1")

    emotion_thresholds = cl.load_thresholds(args.emotion_thresholds_json)
    neutral_scalar, neutral_vector = cl.load_neutral_centroid(args.neutral_centroid_json)

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        total_lines = [json.loads(line.strip()) for line in f if line.strip()]

    final_lines = []
    batch_lines = []
    batch_audios = []
    for line in total_lines:
        emotion = cl.parse_emotion(line)
        if emotion is not None:
            line["emotion"] = emotion

        line["emphasis_spans"] = cl.parse_emphasis_spans(line)

        vad_score, vad_vector = cl.parse_vad_value(line)
        if vad_score is not None:
            line["vad_score"] = vad_score

        dist_to_neutral = cl.compute_dist_to_neutral(
            vad_score=vad_score,
            vad_vector=vad_vector,
            neutral_scalar=neutral_scalar,
            neutral_vector=neutral_vector,
        )
        if dist_to_neutral is not None:
            line["dist_to_neutral"] = dist_to_neutral

        if not (args.keep_existing_intensity and line.get("intensity") is not None):
            thresholds = emotion_thresholds.get(emotion, global_thresholds)
            intensity = cl.discretize_intensity(dist_to_neutral, thresholds)
            if intensity is not None:
                line["intensity"] = intensity

        batch_lines.append(line)
        batch_audios.append(line['audio'])

        if len(batch_lines) >= BATCH_INFER_NUM:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, line in zip(enc_res.audio_codes, batch_lines):
                line['audio_codes'] = code.cpu().tolist()
                final_lines.append(line)
            batch_lines.clear()
            batch_audios.clear()

    if len(batch_audios) > 0:
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, line in zip(enc_res.audio_codes, batch_lines):
            line['audio_codes'] = code.cpu().tolist()
            final_lines.append(line)
        batch_lines.clear()
        batch_audios.clear()

    final_lines = [json.dumps(line, ensure_ascii=False) for line in final_lines]

    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for line in final_lines:
            f.writelines(line + '\n')

if __name__ == "__main__":
    main()
