import random
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import gradio as gr
import symusic
import torch
from miditok import PerTok, TokenizerConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, GenerationConfig

# ============================================================
# Configuration
# ============================================================
DEFAULT_MODEL = "xingjianll/midi-gpt2"
DEFAULT_TOKENIZER = "xingjianll/midi-tokenizer"
DEFAULT_ADAPTER = "outputs/lora-midi-gpt2-lux-segmented-bz4-gas16-e3/adapter-best"

SOUNDFONT_PATH = Path(__file__).parent / "soundfont.sf2"
SOUNDFONT_URLS = [
    # "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2",
    "https://github.com/urish/cinto/raw/master/media/FluidR3%20GM.sf2",
]

# Temp files to clean up on exit
_temp_files: list[Path] = []

# ============================================================
# Global state (initialized at startup)
# ============================================================
BASE_MODEL = None
LORA_MODEL = None
TOKENIZER = None
DEVICE = None


# ============================================================
# Setup utilities
# ============================================================
def setup_soundfont() -> None:
    """Download soundfont if not present."""
    if SOUNDFONT_PATH.exists():
        print(f"Soundfont already exists at {SOUNDFONT_PATH}")
        return

    for url in SOUNDFONT_URLS:
        try:
            print(f"Downloading soundfont from {url}...")
            urllib.request.urlretrieve(url, SOUNDFONT_PATH)
            print("Soundfont ready!")
            return
        except Exception as e:
            print(f"Failed to download from {url}: {e}")

    print("WARNING: Could not download soundfont. Audio preview disabled.")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all devices."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_temp_file(suffix: str) -> Path:
    """Create a temp file and track it for cleanup."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = Path(f.name)
    f.close()
    _temp_files.append(path)
    return path


def cleanup_temp_files() -> None:
    """Remove all tracked temp files."""
    for path in _temp_files:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
    _temp_files.clear()


# ============================================================
# Audio conversion
# ============================================================
def midi_to_audio(midi_path: str) -> str | None:
    """Convert MIDI to WAV using fluidsynth."""
    if not SOUNDFONT_PATH.exists():
        return None

    output_path = create_temp_file(".wav")

    try:
        subprocess.run(
            [
                "fluidsynth",
                "-ni",
                "-a", "file",
                "-T", "wav",
                "-F", str(output_path),
                "-r", "44100",
                str(SOUNDFONT_PATH),
                midi_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)

        print(f"Output file empty or missing: {output_path}")
        return None

    except subprocess.CalledProcessError as e:
        print(f"Fluidsynth error: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Fluidsynth not found. Install with: brew install fluidsynth")
        return None


# ============================================================
# Model loading
# ============================================================
def build_tokenizer(tokenizer_name: str) -> PerTok:
    """Build and load the MIDI tokenizer."""
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


def load_models() -> None:
    """Load both base and LoRA models."""
    global BASE_MODEL, LORA_MODEL, TOKENIZER, DEVICE

    DEVICE = get_device()
    print(f"Using device: {DEVICE}")

    print("Loading tokenizer...")
    TOKENIZER = build_tokenizer(DEFAULT_TOKENIZER)

    print("Loading base model...")
    BASE_MODEL = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
    BASE_MODEL.to(DEVICE)
    BASE_MODEL.eval()

    print("Loading LoRA model...")
    base_for_lora = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
    LORA_MODEL = PeftModel.from_pretrained(base_for_lora, DEFAULT_ADAPTER)
    LORA_MODEL.to(DEVICE)
    LORA_MODEL.eval()

    print("Models ready!")


def get_model(model_choice: str):
    """Get the selected model."""
    if model_choice == "LoRA Fine-tuned":
        return LORA_MODEL
    return BASE_MODEL


# ============================================================
# Generation logic
# ============================================================
def preview_prompt(midi_file: str | None) -> str | None:
    """Convert uploaded MIDI to audio for preview."""
    if midi_file is None:
        return None
    return midi_to_audio(midi_file)


def generate_midi(
    model_choice: str,
    prompt_midi_file: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str | None, str]:
    """Generate MIDI from prompt using selected model."""
    if TOKENIZER is None:
        raise gr.Error("Models not loaded!")

    model = get_model(model_choice)

    # Seed
    actual_seed = seed if seed > 0 else random.randrange(1 << 32)
    set_seed(actual_seed)

    progress(0.1, desc="Encoding prompt...")

    # Build prompt tokens
    if prompt_midi_file is not None:
        score = symusic.Score.from_file(prompt_midi_file)
        prompt_ids = TOKENIZER.encode(score)[0] or [TOKENIZER["BOS_None"]]
    else:
        prompt_ids = [TOKENIZER["BOS_None"]]

    # Truncate if exceeds context length
    max_pos = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", None
    )
    if max_pos and len(prompt_ids) > max_pos:
        prompt_ids = prompt_ids[-max_pos:]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
    )

    progress(0.2, desc=f"Generating with {model_choice}...")

    with torch.inference_mode():
        output = model.generate(input_ids=input_ids, generation_config=gen_config)

    progress(0.8, desc="Decoding MIDI...")

    generated_ids = output[0].tolist()
    midi_bytes = TOKENIZER.decode([generated_ids]).dumps_midi()

    midi_path = create_temp_file(".mid")
    midi_path.write_bytes(midi_bytes)

    progress(0.9, desc="Rendering audio...")

    audio_path = midi_to_audio(str(midi_path))

    info = (
        f"Model: {model_choice} | Seed: {actual_seed} | "
        f"Prompt: {len(prompt_ids)} tokens | Generated: {len(generated_ids)} tokens"
    )

    return str(midi_path), audio_path, info


def generate_both(
    prompt_midi_file: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str | None, str, str, str | None, str]:
    """Generate with both models using the same seed for comparison."""
    if TOKENIZER is None:
        raise gr.Error("Models not loaded!")

    actual_seed = seed if seed > 0 else random.randrange(1 << 32)

    progress(0, desc="Generating with Base model...")
    base_midi, base_audio, base_info = generate_midi(
        "Base Model",
        prompt_midi_file,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        actual_seed,
        progress,
    )

    progress(0.5, desc="Generating with LoRA model...")
    lora_midi, lora_audio, lora_info = generate_midi(
        "LoRA Fine-tuned",
        prompt_midi_file,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        actual_seed,
        progress,
    )

    return base_midi, base_audio, base_info, lora_midi, lora_audio, lora_info


# ============================================================
# Startup
# ============================================================
print("=" * 50)
print("Setting up MIDI Generator...")
print("=" * 50)
setup_soundfont()
load_models()
print("=" * 50)


# ============================================================
# Gradio UI
# ============================================================
with gr.Blocks(title="MIDI Generator") as demo:
    gr.Markdown("# 🎹 Symbolic MIDI Generator")
    gr.Markdown(f"**Base Model:** `{DEFAULT_MODEL}` | **LoRA Adapter:** `{DEFAULT_ADAPTER}`")

    with gr.Tabs():
        # ============================================================
        # Tab 1: Single Generation
        # ============================================================
        with gr.TabItem("Single Generation"):
            with gr.Row():
                with gr.Column():
                    model_choice = gr.Radio(
                        choices=["Base Model", "LoRA Fine-tuned"],
                        value="LoRA Fine-tuned",
                        label="Model",
                    )
                    prompt_midi = gr.File(
                        label="Prompt MIDI (optional)",
                        file_types=[".mid", ".midi"],
                    )
                    prompt_audio = gr.Audio(label="Prompt Preview", type="filepath")

                    max_tokens = gr.Slider(
                        64, 2048, value=512, step=64, label="Max New Tokens"
                    )
                    temperature = gr.Slider(
                        0.1, 2.0, value=1.0, step=0.1, label="Temperature"
                    )
                    top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-p")
                    top_k = gr.Slider(1, 100, value=50, step=1, label="Top-k")
                    seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                    generate_btn = gr.Button("Generate", variant="primary")

                with gr.Column():
                    output_file = gr.File(label="Generated MIDI (download)")
                    output_audio = gr.Audio(label="Generated Audio", type="filepath")
                    output_info = gr.Textbox(label="Info", interactive=False)

            prompt_midi.change(preview_prompt, [prompt_midi], [prompt_audio])
            generate_btn.click(
                generate_midi,
                [model_choice, prompt_midi, max_tokens, temperature, top_p, top_k, seed],
                [output_file, output_audio, output_info],
            )

        # ============================================================
        # Tab 2: Side-by-Side Comparison
        # ============================================================
        with gr.TabItem("Compare Models"):
            with gr.Row():
                with gr.Column():
                    compare_prompt_midi = gr.File(
                        label="Prompt MIDI (optional)",
                        file_types=[".mid", ".midi"],
                    )
                    compare_prompt_audio = gr.Audio(
                        label="Prompt Preview", type="filepath"
                    )

                    compare_max_tokens = gr.Slider(
                        64, 2048, value=512, step=64, label="Max New Tokens"
                    )
                    compare_temperature = gr.Slider(
                        0.1, 2.0, value=1.0, step=0.1, label="Temperature"
                    )
                    compare_top_p = gr.Slider(
                        0.1, 1.0, value=1.0, step=0.05, label="Top-p"
                    )
                    compare_top_k = gr.Slider(1, 100, value=50, step=1, label="Top-k")
                    compare_seed = gr.Number(
                        value=0, label="Seed (0 = random)", precision=0
                    )
                    compare_btn = gr.Button(
                        "Generate Both (same seed)", variant="primary"
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Base Model")
                    base_output_file = gr.File(label="MIDI")
                    base_output_audio = gr.Audio(label="Audio", type="filepath")
                    base_output_info = gr.Textbox(label="Info", interactive=False)

                with gr.Column():
                    gr.Markdown("### LoRA Fine-tuned")
                    lora_output_file = gr.File(label="MIDI")
                    lora_output_audio = gr.Audio(label="Audio", type="filepath")
                    lora_output_info = gr.Textbox(label="Info", interactive=False)

            compare_prompt_midi.change(
                preview_prompt, [compare_prompt_midi], [compare_prompt_audio]
            )
            compare_btn.click(
                generate_both,
                [
                    compare_prompt_midi,
                    compare_max_tokens,
                    compare_temperature,
                    compare_top_p,
                    compare_top_k,
                    compare_seed,
                ],
                [
                    base_output_file,
                    base_output_audio,
                    base_output_info,
                    lora_output_file,
                    lora_output_audio,
                    lora_output_info,
                ],
            )

if __name__ == "__main__":
    try:
        demo.launch()
    finally:
        cleanup_temp_files()