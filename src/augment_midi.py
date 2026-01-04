import argparse
from pathlib import Path

import symusic
from symusic import Score


def list_midi_files(data_dir: Path):
    files = []
    for ext in ("*.mid", "*.midi"):
        files.extend(data_dir.glob(f"**/{ext}"))
    return sorted(set(files))


def parse_args():
    parser = argparse.ArgumentParser(description="Augment MIDI files with transposition and tempo scaling.")
    parser.add_argument("--input-dir", default="~/datasets/lux_midi", help="Folder with .mid/.midi files.")
    parser.add_argument("--output-dir", default="~/datasets/lux_midi_aug", help="Folder to write augmented MIDI.")
    parser.add_argument("--transpose", nargs="*", type=int, default=[-2, -1, 1, 2])
    parser.add_argument("--tempo", nargs="*", type=float, default=[0.9, 1.0, 1.1])
    parser.add_argument("--include-original", action="store_true", help="Include transpose=0 and tempo=1.0.")
    return parser.parse_args()


def apply_tempo_scale(score: Score, factor: float) -> None:
    if factor == 1.0:
        return
    for tempo in score.tempos:
        tempo.qpm *= factor


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = list_midi_files(input_dir)
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {input_dir}")

    transposes = list(args.transpose)
    tempos = list(args.tempo)

    if args.include_original:
        if 0 not in transposes:
            transposes.append(0)
        if 1.0 not in tempos:
            tempos.append(1.0)

    total = 0
    for file_path in midi_files:
        try:
            base_score = Score.from_file(str(file_path))
        except Exception as exc:
            print(f"Skipping {file_path.name}: {exc}")
            continue

        for semitones in transposes:
            for tempo_factor in tempos:
                if not args.include_original and semitones == 0 and tempo_factor == 1.0:
                    continue
                try:
                    score = base_score if semitones == 0 else base_score.shift_pitch(semitones)
                    score = score.copy(deep=True) if semitones == 0 else score
                    apply_tempo_scale(score, tempo_factor)
                    stem = file_path.stem
                    out_name = f"{stem}__tp{semitones:+d}__tempo{tempo_factor:.2f}.mid"
                    out_path = output_dir / out_name
                    score.dump_midi(out_path)
                    total += 1
                except Exception as exc:
                    print(f"Failed {file_path.name} tp={semitones} tempo={tempo_factor}: {exc}")
                    continue

    print(f"Wrote {total} files to {output_dir}")


if __name__ == "__main__":
    main()
