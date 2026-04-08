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
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from grpo_dataset import GRPODataset, grpo_collate_fn
from grpo_rollout import GRPORolloutEngine
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from reward_fns import RewardWeights, compose_reward, group_relative_advantage


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features
        self.lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base.parameters():
            p.requires_grad = False

        # Keep LoRA params on the same device/dtype as the replaced base layer.
        self.lora_A.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        self.lora_B.to(device=self.base.weight.device, dtype=self.base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


def _get_parent_module(root: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _inject_lora_modules(
    root: nn.Module,
    target_module_names: List[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    replaced = []
    named = dict(root.named_modules())
    for name in target_module_names:
        mod = named.get(name, None)
        if mod is None:
            continue
        if not isinstance(mod, nn.Linear):
            continue
        parent, child = _get_parent_module(root, name)
        setattr(parent, child, LoRALinear(mod, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)
    return replaced


def _collect_lora_params(model: nn.Module) -> List[nn.Parameter]:
    params = []
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True
            params.append(p)
        else:
            p.requires_grad = False
    return params


def _extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    out = {}
    for n, p in model.state_dict().items():
        if ".lora_A." in n or ".lora_B." in n:
            out[n] = p.detach().cpu()
    return out


def _module_device(module: nn.Module) -> torch.device:
    for p in module.parameters():
        return p.device
    return torch.device("cpu")


def _save_jsonl(path: str, rows: List[Dict]):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_reward_curve(
    output_dir: str,
    rewards: List[float],
    reward_stds: List[float],
    ser_means: List[float],
    emo_means: List[float],
    emph_means: List[float],
):
    if len(rewards) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    metric_path = os.path.join(output_dir, "reward_curve.json")
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "step": list(range(len(rewards))),
                "mean_group_reward": rewards,
                "reward_std": reward_stds,
                "ser_mean": ser_means,
                "emo_intensity_mean": emo_means,
                "emphasis_mean": emph_means,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(rewards), dtype=np.float32)
        y = np.asarray(rewards, dtype=np.float32)
        s = np.asarray(reward_stds, dtype=np.float32)

        plt.figure(figsize=(8, 4.5), dpi=150)
        plt.plot(x, y, label="mean_group_reward", linewidth=2)
        plt.fill_between(x, y - s, y + s, alpha=0.2, label="±1 std")

        if len(y) >= 5:
            k = min(20, len(y))
            ma = np.convolve(y, np.ones(k, dtype=np.float32) / float(k), mode="valid")
            x_ma = np.arange(k - 1, len(y), dtype=np.float32)
            plt.plot(x_ma, ma, linestyle="--", linewidth=1.8, label=f"moving_avg(k={k})")

        plt.xlabel("step")
        plt.ylabel("reward")
        plt.title("GRPO Reward Curve")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(output_dir, "reward_curve.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"[GRPO] saved reward curve: {fig_path}")
    except Exception as e:
        print(f"[GRPO] reward curve png not generated ({e}). metrics saved at: {metric_path}")


def _prepare_update_models_and_optimizer(args):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype.lower(), torch.bfloat16)

    policy_tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        device_map=args.device,
        dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )

    # reference model for KL regularization
    ref_tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        device_map=args.device,
        dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    ref_tts.model.eval()
    for p in ref_tts.model.parameters():
        p.requires_grad = False

    # default trainable scope
    for p in policy_tts.model.parameters():
        p.requires_grad = False

    if args.update_scope == "codec_head":
        for p in policy_tts.model.talker.codec_head.parameters():
            p.requires_grad = True
        trainable_params = [p for p in policy_tts.model.parameters() if p.requires_grad]

    elif args.update_scope == "lora":
        target_names = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        replaced = _inject_lora_modules(
            policy_tts.model,
            target_module_names=target_names,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        if len(replaced) == 0:
            raise ValueError(
                "No target module was replaced by LoRA. "
                "Please check --lora_target_modules names against model.named_modules()."
            )
        print(f"[GRPO] LoRA injected into modules: {replaced}")
        trainable_params = _collect_lora_params(policy_tts.model)

    else:
        raise ValueError(f"Unsupported --update_scope: {args.update_scope}")

    if len(trainable_params) == 0:
        raise ValueError("No trainable params for GRPO update.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.update_lr, weight_decay=args.weight_decay)

    return policy_tts, ref_tts, optimizer


def _compute_policy_update_loss(
    policy_model,
    ref_model,
    batch_rollouts,
    batch_advantages,
    kl_coef: float,
):
    # action = first codebook token at each generated frame
    logp_terms = []
    kl_terms = []
    head_device = _module_device(policy_model.talker.codec_head)
    for it, adv in zip(batch_rollouts, batch_advantages):
        hidden = it.aux["hidden"].to(head_device)
        codes = it.aux["codes"].to(hidden.device)

        # Normalize shapes from different generate backends.
        # hidden: (T, D) or (1, T, D) -> (T, D)
        if hidden.ndim == 3:
            if hidden.shape[0] == 1:
                hidden = hidden[0]
            else:
                hidden = hidden.reshape(-1, hidden.shape[-1])

        # codes: (T, Q) / (1, T, Q) / (T,) -> (T, Q) or (T, 1)
        if codes.ndim == 1:
            codes = codes.unsqueeze(-1)
        elif codes.ndim == 3:
            if codes.shape[0] == 1:
                codes = codes[0]
            else:
                codes = codes.reshape(-1, codes.shape[-1])

        if hidden.ndim != 2 or codes.ndim != 2 or hidden.shape[0] == 0 or codes.shape[0] == 0:
            continue

        # Align length to avoid shape mismatch in rare backend differences.
        t = min(hidden.shape[0], codes.shape[0])
        hidden = hidden[:t]
        codes = codes[:t]

        action = codes[:, 0].long()  # first codebook token per frame
        logits_pi = policy_model.talker.codec_head(hidden)  # (T, V)
        logp_pi_all = torch.log_softmax(logits_pi, dim=-1)
        logp_pi = logp_pi_all.gather(1, action.unsqueeze(-1)).squeeze(-1).mean()

        with torch.no_grad():
            logits_ref = ref_model.talker.codec_head(hidden)
            logp_ref_all = torch.log_softmax(logits_ref, dim=-1)

        adv_t = torch.tensor(float(adv), device=hidden.device, dtype=logp_pi.dtype)
        logp_terms.append(-adv_t * logp_pi)
        # Distribution KL: KL(pi || ref) = E_pi[log pi - log ref], non-negative in expectation.
        prob_pi = torch.softmax(logits_pi, dim=-1)
        kl_token = (prob_pi * (logp_pi_all - logp_ref_all)).sum(dim=-1)
        kl_terms.append(kl_token.mean())

    if len(logp_terms) == 0:
        zero = torch.tensor(0.0, device=head_device)
        return zero, 0.0, 0.0

    policy_loss = torch.stack(logp_terms).mean()
    kl_loss = torch.stack(kl_terms).mean()
    total_loss = policy_loss + kl_coef * kl_loss
    return total_loss, float(policy_loss.detach().cpu()), float(kl_loss.detach().cpu())


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
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--no_do_sample", action="store_false", dest="do_sample")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument(
        "--save_curve_every",
        type=int,
        default=20,
        help="Save reward_curve.json/png every N steps (and once at the end).",
    )

    parser.add_argument("--reward_w_ser", type=float, default=1.0)
    parser.add_argument("--reward_w_emo_intensity", type=float, default=1.0)
    parser.add_argument("--reward_w_emphasis", type=float, default=1.0)
    parser.add_argument("--update_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--reward_clip_abs",
        type=float,
        default=0.0,
        help="If > 0, clip per-sample total reward to [-x, x] before computing advantage.",
    )
    parser.add_argument(
        "--adv_clip_abs",
        type=float,
        default=2.0,
        help="If > 0, clip normalized group-relative advantage to [-x, x].",
    )
    parser.add_argument(
        "--update_scope",
        type=str,
        default="lora",
        choices=["codec_head", "lora"],
        help="Trainable parameter scope for GRPO update.",
    )
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="talker.codec_head",
        help="Comma-separated full module names for LoRA injection. Default is stable head-only LoRA.",
    )

    parser.add_argument(
        "--allow_no_update",
        action="store_true",
        default=False,
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

    tts, ref_tts, optimizer = _prepare_update_models_and_optimizer(args)

    dataset = GRPODataset(args.train_jsonl)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=grpo_collate_fn)

    rollout = GRPORolloutEngine(
        tts=tts,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    weights = RewardWeights(
        ser=args.reward_w_ser,
        emo_intensity=args.reward_w_emo_intensity,
        emphasis=args.reward_w_emphasis,
    )

    step_reward_means: List[float] = []
    step_reward_stds: List[float] = []
    step_ser_means: List[float] = []
    step_emo_means: List[float] = []
    step_emph_means: List[float] = []

    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            rollouts = rollout.sample_group(batch, group_size=args.group_size)

            # each prompt has `group_size` candidates in fixed order
            rows = []
            mean_total_rewards = []
            all_ser = []
            all_emo = []
            all_emph = []
            update_rollouts = []
            update_advantages = []
            for sample_index in range(len(batch)):
                items = [r for r in rollouts if r.sample_index == sample_index]
                reward_parts = [compose_reward(it, weights) for it in items]
                raw_totals = [p["total"] for p in reward_parts]
                group_totals = list(raw_totals)
                if args.reward_clip_abs and args.reward_clip_abs > 0:
                    c = float(args.reward_clip_abs)
                    group_totals = [float(np.clip(x, -c, c)) for x in group_totals]

                adv = group_relative_advantage(group_totals)
                if args.adv_clip_abs and args.adv_clip_abs > 0:
                    a = float(args.adv_clip_abs)
                    adv = [float(np.clip(x, -a, a)) for x in adv]
                mean_total_rewards.append(sum(group_totals) / max(len(group_totals), 1))

                for it, part, used_total, a in zip(items, reward_parts, group_totals, adv):
                    update_rollouts.append(it)
                    update_advantages.append(a)
                    all_ser.append(float(part["ser"]))
                    all_emo.append(float(part["emo_intensity"]))
                    all_emph.append(float(part["emphasis"]))
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
                            "total_reward": used_total,
                            "total_reward_raw": part["total"],
                            "advantage": a,
                        }
                    )

            _save_jsonl(traj_path, rows)

            mean_batch_reward = sum(mean_total_rewards) / max(len(mean_total_rewards), 1)
            reward_std = float(np.std(mean_total_rewards)) if len(mean_total_rewards) > 0 else 0.0
            ser_mean = float(np.mean(all_ser)) if len(all_ser) > 0 else 0.0
            emo_mean = float(np.mean(all_emo)) if len(all_emo) > 0 else 0.0
            emph_mean = float(np.mean(all_emph)) if len(all_emph) > 0 else 0.0
            step_reward_means.append(float(mean_batch_reward))
            step_reward_stds.append(float(reward_std))
            step_ser_means.append(ser_mean)
            step_emo_means.append(emo_mean)
            step_emph_means.append(emph_mean)
            msg = (
                f"[GRPO] epoch={epoch} step={global_step} "
                f"mean_group_reward={mean_batch_reward:.4f} reward_std={reward_std:.4f} "
                f"ser={ser_mean:.4f} emo={emo_mean:.4f} emph={emph_mean:.4f}"
            )

            if not args.allow_no_update:
                tts.model.train()
                optimizer.zero_grad(set_to_none=True)
                total_loss, policy_loss_scalar, kl_loss_scalar = _compute_policy_update_loss(
                    policy_model=tts.model,
                    ref_model=ref_tts.model,
                    batch_rollouts=update_rollouts,
                    batch_advantages=update_advantages,
                    kl_coef=args.kl_coef,
                )
                total_loss.backward()
                eff_n = int(len(update_rollouts))
                torch.nn.utils.clip_grad_norm_(
                    [p for p in tts.model.parameters() if p.requires_grad],
                    max_norm=args.max_grad_norm,
                )
                optimizer.step()
                msg += f" policy_loss={policy_loss_scalar:.6f} kl={kl_loss_scalar:.6e} n={eff_n}"

            print(msg)

            if args.save_curve_every > 0 and (global_step % args.save_curve_every == 0):
                _save_reward_curve(
                    args.output_dir,
                    step_reward_means,
                    step_reward_stds,
                    step_ser_means,
                    step_emo_means,
                    step_emph_means,
                )

            global_step += 1

    if not args.allow_no_update:
        if args.update_scope == "codec_head":
            save_path = os.path.join(args.output_dir, "grpo_codec_head.pt")
            torch.save(tts.model.talker.codec_head.state_dict(), save_path)
            print(f"[GRPO] saved codec_head checkpoint: {save_path}")
        else:
            save_path = os.path.join(args.output_dir, "grpo_lora_adapter.pt")
            torch.save(_extract_lora_state_dict(tts.model), save_path)
            print(f"[GRPO] saved LoRA adapter: {save_path}")

            _save_reward_curve(
                args.output_dir,
                step_reward_means,
                step_reward_stds,
                step_ser_means,
                step_emo_means,
                step_emph_means,
            )

    print(f"[GRPO] finished. trajectories saved at: {traj_path}")


if __name__ == "__main__":
    train_grpo()
