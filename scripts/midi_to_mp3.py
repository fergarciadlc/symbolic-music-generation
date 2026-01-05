import argparse
from pathlib import Path
import subprocess
import sys

CMD = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"

def midi_to_mp3(input_file: Path, output_file: Path, musescore_cmd: str = CMD) -> bool:
    """Convert a MIDI file to MP3 using MuseScore CLI."""
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {input_file.name} to {output_file.name}")
    
    try:
        # MuseScore CLI: convert MIDI directly to MP3
        cmd = [
            musescore_cmd,
            str(input_file),
            "-o", str(output_file)
        ]
        
        print("Running command:", " ".join(cmd))
        subprocess.run(cmd, check=False, capture_output=True)
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert MIDI file(s) to MP3 audio using MuseScore.")
    parser.add_argument("input", help="Path to: a MIDI file, comma-separated list of files, or a folder containing MIDI files")
    parser.add_argument("--output-dir", default="audio", help="Output directory for MP3 files (default: audio)")
    parser.add_argument("--separate-folders", action="store_true", help="Create separate subfolder for each input file")
    
    args = parser.parse_args()
    
    # Determine input type and collect files
    midi_files = []
    input_path = Path(args.input)
    
    if "," in args.input:
        # Comma-separated list of files
        file_paths = [p.strip() for p in args.input.split(",")]
        for file_path in file_paths:
            path = Path(file_path)
            if path.exists() and path.is_file():
                midi_files.append(path)
            else:
                print(f"Warning: File not found: {path}")
    elif input_path.is_dir():
        # Directory: find all MIDI files
        midi_files = list(input_path.glob("**/*.mid")) + list(input_path.glob("**/*.midi"))
        if not midi_files:
            print(f"No MIDI files found in {input_path}")
            return
        print(f"Found {len(midi_files)} MIDI file(s) in {input_path}\n")
    elif input_path.is_file():
        # Single file
        midi_files = [input_path]
    else:
        print(f"Error: Invalid input: {args.input}")
        return
    
    # Process each file
    successful = 0
    failed = 0
    
    for midi_file in midi_files:
        if args.separate_folders:
            # Create a subfolder for this file's audio
            output_dir = Path(args.output_dir) / midi_file.stem
        else:
            output_dir = Path(args.output_dir)
        
        output_file = output_dir / f"{midi_file.stem}.mp3"
        
        success = midi_to_mp3(midi_file, output_file)
        if success:
            successful += 1
        else:
            failed += 1
        
        print()  # Empty line between files
    
    print(f"Conversion complete: {successful} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
