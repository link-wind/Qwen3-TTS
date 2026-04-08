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
import os
import shutil
from typing import List

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from huggingface_hub import snapshot_download
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None


def _set_requires_grad(module, enabled: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = enabled


def _configure_trainable_scope(model, freeze_codec_embedding: bool = True):
    # 1) freeze all first
    _set_requires_grad(model, False)

    # 2) unfreeze LLM/talker trunk
    if hasattr(model, "talker"):
        _set_requires_grad(model.talker, True)

    # 3) keep speaker encoder frozen
    if hasattr(model, "speaker_encoder"):
        _set_requires_grad(model.speaker_encoder, False)

    # 4) explicitly freeze codec/tokenizer-style embeddings if requested
    if freeze_codec_embedding and hasattr(model, "talker"):
        talker_model = getattr(model.talker, "model", None)
        if talker_model is not None:
            if hasattr(talker_model, "codec_embedding"):
                _set_requires_grad(talker_model.codec_embedding, False)
            if hasattr(talker_model, "text_embedding"):
                _set_requires_grad(talker_model.text_embedding, False)

        code_predictor = getattr(model.talker, "code_predictor", None)
        if code_predictor is not None and hasattr(code_predictor, "get_input_embeddings"):
            for emb in code_predictor.get_input_embeddings():
                _set_requires_grad(emb, False)


def _collect_trainable_params(model) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _print_trainable_stats(model, accelerator: Accelerator):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    ratio = (trainable / total * 100.0) if total > 0 else 0.0
    accelerator.print(
        f"Trainable params: {trainable:,} / {total:,} ({ratio:.2f}%)"
    )


def _resolve_model_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    return snapshot_download(model_path, allow_patterns=["*"], local_files_only=False)


def _build_instruct_prefix_embeds(processor, model, instruct_texts):
    if instruct_texts is None:
        return None, None

    wrapped = []
    non_empty = False
    for txt in instruct_texts:
        t = (txt or "").strip()
        if t:
            non_empty = True
            wrapped.append(f"<|im_start|>user\n{t}<|im_end|>\n")
        else:
            wrapped.append("")

    if not non_empty:
        return None, None

    emb_list = []
    len_list = []
    for text in wrapped:
        if text == "":
            emb = torch.zeros((0, model.talker.config.hidden_size), device=model.device, dtype=model.dtype)
        else:
            tok = processor(text=text, return_tensors="pt", padding=True)
            ids = tok["input_ids"].to(model.device)
            txt_emb = model.talker.get_text_embeddings()(ids)
            emb = model.talker.text_projection(txt_emb).squeeze(0)
        emb_list.append(emb)
        len_list.append(emb.shape[0])

    max_len = max(len_list)
    if max_len == 0:
        return None, None

    bsz = len(emb_list)
    hidden = model.talker.config.hidden_size
    prefix = torch.zeros((bsz, max_len, hidden), device=model.device, dtype=model.dtype)
    prefix_mask = torch.zeros((bsz, max_len), device=model.device, dtype=torch.long)
    for i, emb in enumerate(emb_list):
        if emb.shape[0] > 0:
            prefix[i, :emb.shape[0], :] = emb
            prefix_mask[i, :emb.shape[0]] = 1

    return prefix, prefix_mask


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument(
        "--control_mode",
        type=str,
        default="instruct",
        choices=["prompt", "instruct"],
        help="Control interface mode in dataset text construction.",
    )
    parser.add_argument(
        "--freeze_codec_embedding",
        action="store_true",
        default=True,
        help="Freeze codec/text input embeddings and code-predictor input embeddings.",
    )
    parser.add_argument(
        "--no_freeze_codec_embedding",
        action="store_false",
        dest="freeze_codec_embedding",
        help="Disable freezing codec/text input embeddings.",
    )
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    MODEL_PATH = _resolve_model_path(args.init_model_path)

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config, control_mode=args.control_mode)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    _configure_trainable_scope(
        qwen3tts.model,
        freeze_codec_embedding=args.freeze_codec_embedding,
    )
    _print_trainable_stats(qwen3tts.model, accelerator)
    trainable_params = _collect_trainable_params(qwen3tts.model)
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters left after scope configuration.")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']
                instruct_texts = batch.get('instruct_texts', None)

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                instruct_prefix_embeds, instruct_prefix_mask = _build_instruct_prefix_embeds(
                    processor=qwen3tts.processor,
                    model=model,
                    instruct_texts=instruct_texts,
                )

                if instruct_prefix_embeds is not None:
                    input_embeddings = torch.cat([instruct_prefix_embeds, input_embeddings], dim=1)
                    attention_mask = torch.cat([instruct_prefix_mask, attention_mask], dim=1)

                    bsz = codec_0_labels.shape[0]
                    prefix_len = instruct_prefix_embeds.shape[1]
                    prefix_labels = torch.full(
                        (bsz, prefix_len),
                        -100,
                        dtype=codec_0_labels.dtype,
                        device=codec_0_labels.device,
                    )
                    codec_0_labels = torch.cat([prefix_labels, codec_0_labels], dim=1)

                    prefix_codec_mask = torch.zeros(
                        (codec_mask.shape[0], prefix_len),
                        dtype=codec_mask.dtype,
                        device=codec_mask.device,
                    )
                    codec_mask = torch.cat([prefix_codec_mask, codec_mask], dim=1)

                    prefix_codec_ids = torch.zeros(
                        (codec_ids.shape[0], prefix_len, codec_ids.shape[2]),
                        dtype=codec_ids.dtype,
                        device=codec_ids.device,
                    )
                    codec_ids = torch.cat([prefix_codec_ids, codec_ids], dim=1)

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

if __name__ == "__main__":
    train()
