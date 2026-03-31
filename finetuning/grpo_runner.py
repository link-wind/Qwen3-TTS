# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""Minimal GRPO scaffold runner.

This file intentionally focuses on:
  1) group rollout
  2) three-task reward computation
  3) group-relative advantage computation
  4) trajectory logging for later policy-update integration

Policy update and KL-regularized objective are left as TODO hooks.
"""

import argparse
import json
import os
from typing import Dict, List

from torch.utils.data import DataLoader

from grpo_dataset import GRPODataset, grpo_collate_fn
from grpo_rollout import GRPORolloutEngine
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from reward_fns import RewardWeights, compose_reward, group_relative_advantage


def _save_jsonl(path: str, rows: List[Dict]):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_grpo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, required=True)
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="grpo_output")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--reward_w_ser", type=float, default=1.0)
    parser.add_argument("--reward_w_emo_intensity", type=float, default=1.0)
    parser.add_argument("--reward_w_emphasis", type=float, default=1.0)

    parser.add_argument(
        "--allow_no_update",
        action="store_true",
        default=True,
        help="Scaffold mode: rollout + rewards + advantages only (no policy update).",
    )
    parser.add_argument(
        "--no_allow_no_update",
        action="store_false",
        dest="allow_no_update",
        help="Require a policy update implementation (will raise NotImplementedError).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    traj_path = os.path.join(args.output_dir, "grpo_trajectories.jsonl")

    dtype_map = {
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float16": "float16",
        "fp16": "float16",
        "float32": "float32",
        "fp32": "float32",
    }
    torch_dtype = dtype_map.get(args.dtype.lower(), "bfloat16")

    tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        device_map=args.device,
        dtype=getattr(__import__("torch"), torch_dtype),
        attn_implementation="flash_attention_2",
    )

    dataset = GRPODataset(args.train_jsonl)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=grpo_collate_fn)

    rollout = GRPORolloutEngine(
        tts=tts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    weights = RewardWeights(
        ser=args.reward_w_ser,
        emo_intensity=args.reward_w_emo_intensity,
        emphasis=args.reward_w_emphasis,
    )

    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            rollouts = rollout.sample_group(batch, group_size=args.group_size)

            # each prompt has `group_size` candidates in fixed order
            rows = []
            mean_total_rewards = []
            for sample_index in range(len(batch)):
                items = [r for r in rollouts if r.sample_index == sample_index]
                reward_parts = [compose_reward(it, weights) for it in items]
                group_totals = [p["total"] for p in reward_parts]
                adv = group_relative_advantage(group_totals)
                mean_total_rewards.append(sum(group_totals) / max(len(group_totals), 1))

                for it, part, a in zip(items, reward_parts, adv):
                    rows.append(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "sample_index": sample_index,
                            "group_index": it.group_index,
                            "text": it.prompt.text,
                            "language": it.prompt.language,
                            "speaker": it.prompt.speaker,
                            "emotion": it.prompt.emotion,
                            "intensity": it.prompt.intensity,
                            "emphasis": it.prompt.emphasis,
                            "control_tags": it.prompt.control_tags,
                            "rms": it.aux.get("rms", 0.0),
                            "ser_reward": part["ser"],
                            "emo_intensity_reward": part["emo_intensity"],
                            "emphasis_reward": part["emphasis"],
                            "total_reward": part["total"],
                            "advantage": a,
                        }
                    )

            _save_jsonl(traj_path, rows)

            mean_batch_reward = sum(mean_total_rewards) / max(len(mean_total_rewards), 1)
            print(f"[GRPO] epoch={epoch} step={global_step} mean_group_reward={mean_batch_reward:.4f}")

            if not args.allow_no_update:
                raise NotImplementedError(
                    "Policy update with KL regularization is not implemented in scaffold yet. "
                    "Please integrate logprob extraction and optimizer step here."
                )

            global_step += 1

    print(f"[GRPO] finished. trajectories saved at: {traj_path}")


if __name__ == "__main__":
    train_grpo()
