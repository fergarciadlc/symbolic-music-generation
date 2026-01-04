import argparse
import json
import math
import random
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from miditok import PerTok, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, GenerationConfig, get_linear_schedule_with_warmup


def build_tokenizer(tokenizer_name: str):
    config = TokenizerConfig(
        num_velocities=8,
        use_velocities=True,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=False,
        use_sustain_pedals=False,
        use_pitch_bends=False,
        use_pitch_intervals=False,
        use_programs=False,
        use_pitchdrum_tokens=False,
        ticks_per_quarter=320,
        use_microtiming=False,
        max_microtiming_shift=0.125,
    )
    tokenizer = PerTok(config)
    tokenizer.from_pretrained(tokenizer_name)
    return tokenizer


def list_midi_files(data_dir: Path):
    files = []
    for ext in ("*.mid", "*.midi"):
        files.extend(data_dir.glob(f"**/{ext}"))
    return sorted(set(files))


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    steps = 0
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for xingjianll/midi-gpt2.")
    parser.add_argument("--data-dir", default="~/datasets/lux_midi", help="Folder with .mid/.midi files.")
    parser.add_argument("--output-dir", default="outputs/lora-midi-gpt2-lux", help="Where to save adapters.")
    parser.add_argument("--model-name", default="xingjianll/midi-gpt2")
    parser.add_argument("--tokenizer-name", default="xingjianll/midi-tokenizer")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    midi_files = list_midi_files(data_dir)
    if len(midi_files) < 2:
        raise ValueError("Need at least 2 MIDI files to split train/val.")

    train_files, val_files = train_test_split(
        midi_files, test_size=args.val_split, random_state=args.seed
    )
    if len(val_files) == 0:
        val_files = train_files[:1]
        train_files = train_files[1:]

    tokenizer = build_tokenizer(args.tokenizer_name)

    train_dataset = DatasetMIDI(
        files_paths=train_files,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    val_dataset = DatasetMIDI(
        files_paths=val_files,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )

    collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            running_loss += loss.item()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, device)

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"steps={global_step}/{total_steps}"
        )

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(output_dir / "adapter-best")

    model.save_pretrained(output_dir / "adapter-last")

    metadata = {
        "base_model": args.model_name,
        "tokenizer": args.tokenizer_name,
        "max_seq_len": args.max_seq_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "val_split": args.val_split,
        "best_val_loss": best_val,
    }
    (output_dir / "training_meta.json").write_text(json.dumps(metadata, indent=2))

    gen_config = GenerationConfig.from_pretrained(args.model_name)
    gen_config.save_pretrained(output_dir / "generation_config")

    print(f"Done. Adapters saved to: {output_dir}")


if __name__ == "__main__":
    main()
