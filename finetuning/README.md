## Fine Tuning Qwen3-TTS-12Hz-1.7B/0.6B-Base

The Qwen3-TTS-12Hz-1.7B/0.6B-Base model series currently supports single-speaker fine-tuning. Please run `pip install qwen-tts` first, then run the command below:

```
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/finetuning
```

Then follow the steps below to complete the entire fine-tuning workflow. Multi-speaker fine-tuning and other advanced fine-tuning features will be supported in future releases.

> Note: this document now includes a **control-conditioned SFT migration spec** (emotion/intensity/emphasis) for ESD + Expresso style training. It defines data contracts and pipeline invariants used by upcoming commits.

### 1) Input JSONL format

Prepare your training file as a JSONL (one JSON object per line). Each line must contain:

- `audio`: path to the target training audio (wav)
- `text`: transcript corresponding to `audio`
- `ref_audio`: path to the reference speaker audio (wav)

Example:
```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav"}
```

`ref_audio` recommendation:
- Strongly recommended: use the same `ref_audio` for all samples.
- Keeping `ref_audio` identical across the dataset usually improves speaker consistency and stability during generation.


### 1.1) Control-conditioned JSONL schema (ESD + Expresso migration)

For emotion/intensity/emphasis controllable SFT, each JSONL line should additionally include:

- `emotion`: emotion category string, e.g. `happy|sad|angry|neutral|...`
- `intensity`: discretized level, one of `weak|medium|strong`
- `emphasis_spans` (optional): emphasis spans as char-level ranges, e.g. `[[2, 4], [10, 12]]`
- `vad_score` (optional): scalar VAD/arousal score used for debugging
- `dist_to_neutral` (optional): euclidean distance to neutral centroid used for debugging

Example (control-conditioned):
```jsonl
{"audio":"./data/esd_0001.wav","text":"今天真的很开心见到你。","ref_audio":"./data/ref.wav","emotion":"happy","intensity":"strong","emphasis_spans":[[2,4]]}
{"audio":"./data/expresso_0002.wav","text":"我可以再说一遍。","ref_audio":"./data/ref.wav","emotion":"neutral","intensity":"weak","emphasis_spans":[]}
```

Compatibility rule:
- If control fields are missing, pipeline must fall back to the original base SFT behavior.

Control text canonicalization (train/infer must be identical):
- Canonical prepend form: `[emotion][intensity] + text`
- Emphasis rendering: preserve `*强调词*` style or convert from `emphasis_spans` into equivalent text markers before tokenization.


### 2) Prepare data (extract `audio_codes`)

Convert `train_raw.jsonl` into a training JSONL that includes `audio_codes`:

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

For control-conditioned migration data, `prepare_data.py` is expected to additionally:

1. Map source labels from ESD/Expresso/EmotionTalk to unified fields (`emotion`, `emphasis_spans`).
2. Compute VAD-based `dist_to_neutral`.
3. Discretize intensity into `weak|medium|strong` with **emotion-specific thresholds**.
4. Export `audio_codes` while preserving all control fields.

Recommended threshold policy:
- Primary: emotion-specific thresholds (best alignment with arousal differences across emotions)
- Fallback: global thresholds for quick smoke tests


### 3) Fine-tune

Run SFT using the prepared JSONL:

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name speaker_test
```

Checkpoints will be written to:
- `output/checkpoint-epoch-0`
- `output/checkpoint-epoch-1`
- `output/checkpoint-epoch-2`
- ...

Control-conditioned SFT invariants:

1. Keep train/infer prompt construction isomorphic (`[emotion][intensity] + text`).
2. Mask control-token positions in `labels` (set to `-100`) if the objective is codec-token CE only.
3. Preserve `ref_audio` path and 3-second cloning behavior.
4. Prefer low LR (`2e-5 ~ 5e-5`) and LLM-only trainable scope to reduce speech-rate drift.


### 4) Quick inference test

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="She said she would be here by noon.",
    speaker="speaker_test",
)
sf.write("output.wav", wavs[0], sr)
```

Control-conditioned inference target interface (migration spec):

```python
wavs, sr = tts.generate_custom_voice(
  text="[happy][strong] 今天*特别*开心。",
  speaker="speaker_test",
  # or explicit controls in API params once exposed:
  # emotion="happy", intensity="strong", emphasis="特别"
)
```

Parameter precedence rule (once enabled):
- `control_tags` > `emotion/intensity` > tags parsed from raw text

### One-click shell script example

```bash
#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"

RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="output"

BATCH_SIZE=2
LR=2e-5
EPOCHS=3
SPEAKER_NAME="speaker_1"

python prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

python sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME}
```

---

## Commit 1 acceptance checklist (docs/schema+pipeline)

- [x] Document control-conditioned JSONL schema (`emotion`, `intensity`, `emphasis_spans`)
- [x] Define canonical prepend format (`[emotion][intensity] + text`)
- [x] Define train/infer isomorphism constraints
- [x] Define compatibility fallback for non-control datasets
- [x] Define data-prep expectations for VAD distance and intensity discretization