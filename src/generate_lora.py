import argparse
from pathlib import Path

import torch
from miditok import PerTok, TokenizerConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, GenerationConfig


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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MIDI from a LoRA-adapted midi-gpt2.")
    parser.add_argument("--adapter-dir", required=True, help="Path to adapter-best or adapter-last.")
    parser.add_argument("--output", default="lora_output.mid")
    parser.add_argument("--model-name", default="xingjianll/midi-gpt2")
    parser.add_argument("--tokenizer-name", default="xingjianll/midi-tokenizer")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    tokenizer = build_tokenizer(args.tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.to(device)
    model.eval()

    input_ids = torch.tensor([[tokenizer["BOS_None"]]], dtype=torch.long, device=device)
    gen_config = GenerationConfig.from_pretrained(args.model_name)
    gen_config.max_new_tokens = args.max_new_tokens
    gen_config.temperature = args.temperature
    gen_config.top_p = args.top_p
    gen_config.top_k = args.top_k

    with torch.no_grad():
        output = model.generate(input_ids=input_ids, generation_config=gen_config)

    generated_ids = output[0].tolist()
    midi_bytes = tokenizer.decode([generated_ids]).dumps_midi()
    Path(args.output).write_bytes(midi_bytes)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
