import argparse
import random
from pathlib import Path

import symusic
import torch
from miditok import PerTok, TokenizerConfig
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
    parser = argparse.ArgumentParser(description="Generate MIDI from base midi-gpt2 model.")
    parser.add_argument("--prompt-midi", default=None, help="Optional MIDI file to seed generation.")
    parser.add_argument("--output", default="output.mid", help="Output MIDI file path.")
    parser.add_argument("--model-name", default="xingjianll/midi-gpt2", help="Model name or path.")
    parser.add_argument("--tokenizer-name", default="xingjianll/midi-tokenizer", help="Tokenizer name or path.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling top-p.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (omit for random).")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randrange(1 << 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    tokenizer = build_tokenizer(args.tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    if args.prompt_midi:
        prompt_path = Path(args.prompt_midi).expanduser()
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt MIDI not found: {prompt_path}")
        score = symusic.Score.from_file(str(prompt_path))
        prompt_ids = tokenizer.encode(score)[0]
        if not prompt_ids:
            prompt_ids = [tokenizer["BOS_None"]]
    else:
        prompt_ids = [tokenizer["BOS_None"]]

    max_positions = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", None
    )
    if max_positions is not None and len(prompt_ids) > max_positions:
        prompt_ids = prompt_ids[-max_positions:]
    
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    gen_config = GenerationConfig.from_pretrained(args.model_name)
    if max_positions is not None:
        remaining = max_positions - input_ids.shape[1]
        gen_config.max_new_tokens = max(0, min(args.max_new_tokens, remaining))
    else:
        gen_config.max_new_tokens = args.max_new_tokens
    gen_config.temperature = args.temperature
    gen_config.top_p = args.top_p
    gen_config.top_k = args.top_k

    print(f"Generating with base model: {args.model_name} on device: {device}")
    print(f"Seed: {seed}")
    print("Parameters:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 40)

    with torch.no_grad():
        output = model.generate(input_ids=input_ids, generation_config=gen_config)

    generated_ids = output[0].tolist()
    midi_bytes = tokenizer.decode([generated_ids]).dumps_midi()
    Path(args.output).write_bytes(midi_bytes)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
