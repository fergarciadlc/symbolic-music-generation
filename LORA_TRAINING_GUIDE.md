# LoRA Training Guide (midi-gpt2)

This guide covers LoRA finetuning for `xingjianll/midi-gpt2` using your artist MIDI files.

## Dataset

- Place MIDI files in a folder (default: `~/datasets/lux_midi`).
- Files can be `.mid` or `.midi`.
- The training script splits by files, then tokenizes and chunks sequences to `--max-seq-len`.

## Artist-Style Only vs Style Flavor

- **Artist-style only**: train only on the album data.
  - Pros: very consistent style.
  - Cons: high risk of overfitting and verbatim memorization.
  - Use fewer epochs, smaller LoRA rank, and stronger sampling noise.
- **Artist-style flavor**: mix album data with base data or apply light augmentation.
  - Pros: more variety and better generalization.
  - Cons: style can be less strict.
  - Use slightly more epochs, add augmentation, or blend datasets.

## Augmentation (Optional)

Not strictly required, but recommended with only ~15 files.
Common choices:
- Transpose by +/- 2-3 semitones.
- Slight tempo scaling (e.g., 0.9x to 1.1x).

If you want to preserve key-specific traits, skip transposition.

## Training

Example (M1 Pro / MPS):

```bash
python src/augment_midi.py \
  --input-dir ~/datasets/lux_midi \
  --output-dir ~/datasets/lux_midi_aug \
  --transpose -2 -1 1 2 \
  --tempo 0.9 1.0 1.1

python src/train_lora.py \
  --data-dir ~/datasets/lux_midi_aug \
  --output-dir outputs/lora-midi-gpt2-lux \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --epochs 6 \
  --num-workers 0
```

Example (M1 Pro / MPS, no augmentation):

```bash
python src/train_lora.py \
  --data-dir ~/datasets/lux_midi \
  --output-dir outputs/lora-midi-gpt2-lux \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --epochs 6 \
  --num-workers 0
```

Example (RTX 4090):

```bash
python src/augment_midi.py \
  --input-dir ~/datasets/lux_midi \
  --output-dir ~/datasets/lux_midi_aug \
  --transpose -2 -1 1 2 \
  --tempo 0.9 1.0 1.1

python src/train_lora.py \
  --data-dir ~/datasets/lux_midi_aug \
  --output-dir outputs/lora-midi-gpt2-lux \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --epochs 8
```

Example (RTX 4090, no augmentation):

```bash
python src/train_lora.py \
  --data-dir ~/datasets/lux_midi \
  --output-dir outputs/lora-midi-gpt2-lux \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --epochs 8
```

Key knobs:
- `--max-seq-len`: 1024 (base model default).
- `--lora-r`: 8-16 for small datasets (higher = more capacity, more overfit).
- `--epochs`: 4-10 for small datasets; stop early if samples start copying.

Outputs:
- `outputs/lora-midi-gpt2-lux/adapter-best`
- `outputs/lora-midi-gpt2-lux/adapter-last`

## Generation

```bash
python src/generate_lora.py \
  --adapter-dir outputs/lora-midi-gpt2-lux/adapter-best \
  --output lora_output.mid \
  --max-new-tokens 1024 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 50
```

## Training Time (Rough)

- **RTX 4090**: tens of minutes for a few epochs.
- **M1 Pro (CPU/MPS)**: several hours depending on batch size and sequence length.

If you want faster iteration, use fewer epochs and a smaller `--max-seq-len`.




--- 
Lora parameters
Those LoRA settings in train_lora.py are the standard, low‑risk defaults for GPT‑2‑style models and small-ish datasets:

target_modules=["c_attn", "c_proj", "c_fc"] matches GPT‑2’s core linear layers: c_attn is the QKV projection, c_proj is the attention output (and also MLP output in GPT‑2), and c_fc is the MLP input. Hitting these gives you most of the model’s expressive capacity without touching embeddings or layer norms.
r, alpha, dropout are the usual LoRA knobs: r controls adapter rank (capacity), alpha scales the LoRA update, and dropout regularizes to avoid overfitting—important with 15 files.
bias="none" keeps the adapter lightweight and avoids training biases, which helps stability and reduces overfitting.
task_type=TaskType.CAUSAL_LM tells PEFT to wire the adapter in the expected causal LM path.
If you want to tune for your dataset size:

More conservative (less overfit): r=8, lora_dropout=0.1
More capacity (if outputs feel too generic): r=32, lora_alpha=64