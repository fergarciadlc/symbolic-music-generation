import argparse
from pathlib import Path
import symusic


def segment_midi(input_file: Path, output_dir: Path):
    """Segment a MIDI file by markers and save each segment separately."""
    
    # Load the MIDI file
    score = symusic.Score(str(input_file))
    
    # Check if there are markers
    if len(score.markers) == 0:
        print(f"No markers found in {input_file.name}")
        return
    
    # Sort markers by time
    markers = sorted(score.markers, key=lambda m: m.time)
    
    # Get the original filename without extension
    original_name = input_file.stem
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {input_file.name}")
    print(f"Found {len(markers)} segments:")
    
    # Process each segment
    for i, marker in enumerate(markers):
        # Determine segment boundaries
        start_time = marker.time
        end_time = markers[i + 1].time if i + 1 < len(markers) else None
        
        # Clean segment name (remove spaces, special chars)
        segment_name = marker.text.strip().replace(" ", "_").replace("/", "-")
        
        # Create output filename: {00}_{segment_name}_originalname.mid
        output_filename = f"{i:02d}_{segment_name}_{original_name}.mid"
        output_path = output_dir / output_filename
        
        # Create a copy of the score
        segment_score = score.copy()
        
        # Filter tracks to only include notes within the segment time range
        for track in segment_score.tracks:
            # Filter notes
            if end_time is not None:
                track.notes = [n for n in track.notes if start_time <= n.time < end_time]
            else:
                track.notes = [n for n in track.notes if start_time <= n.time]
            
            # Shift notes to start at time 0
            for note in track.notes:
                note.time -= start_time
            
            # Filter and shift controls
            if end_time is not None:
                track.controls = [c for c in track.controls if start_time <= c.time < end_time]
            else:
                track.controls = [c for c in track.controls if start_time <= c.time]
            for control in track.controls:
                control.time -= start_time
            
            # Filter and shift pedals
            if end_time is not None:
                track.pedals = [p for p in track.pedals if start_time <= p.time < end_time]
            else:
                track.pedals = [p for p in track.pedals if start_time <= p.time]
            for pedal in track.pedals:
                pedal.time -= start_time
            
            # Filter and shift pitch bends
            if end_time is not None:
                track.pitch_bends = [pb for pb in track.pitch_bends if start_time <= pb.time < end_time]
            else:
                track.pitch_bends = [pb for pb in track.pitch_bends if start_time <= pb.time]
            for pb in track.pitch_bends:
                pb.time -= start_time
        
        # Filter and shift tempo changes
        if end_time is not None:
            segment_score.tempos = [t for t in segment_score.tempos if start_time <= t.time < end_time]
        else:
            segment_score.tempos = [t for t in segment_score.tempos if start_time <= t.time]
        for tempo in segment_score.tempos:
            tempo.time -= start_time
        
        # Ensure there's at least one tempo at the start
        if len(segment_score.tempos) == 0 or segment_score.tempos[0].time > 0:
            # Get the tempo at the start time from original score
            default_tempo = symusic.Tempo(time=0, qpm=120)
            for tempo in score.tempos:
                if tempo.time <= start_time:
                    default_tempo = symusic.Tempo(time=0, qpm=tempo.qpm)
            segment_score.tempos.insert(0, default_tempo)
        
        # Filter and shift time signatures
        if end_time is not None:
            segment_score.time_signatures = [ts for ts in segment_score.time_signatures if start_time <= ts.time < end_time]
        else:
            segment_score.time_signatures = [ts for ts in segment_score.time_signatures if start_time <= ts.time]
        for ts in segment_score.time_signatures:
            ts.time -= start_time
        
        # Filter and shift key signatures
        if end_time is not None:
            segment_score.key_signatures = [ks for ks in segment_score.key_signatures if start_time <= ks.time < end_time]
        else:
            segment_score.key_signatures = [ks for ks in segment_score.key_signatures if start_time <= ks.time]
        for ks in segment_score.key_signatures:
            ks.time -= start_time
        
        # Clear markers from the segment
        segment_score.markers = []
        
        # Save the segment
        segment_score.dump_midi(str(output_path))
        
        duration_info = f"to tick {end_time}" if end_time else "to end"
        print(f"  [{i:02d}] {marker.text} (tick {start_time} {duration_info}) -> {output_filename}")
    
    print(f"\nSaved {len(markers)} segments to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Segment a MIDI file by markers into separate files.")
    parser.add_argument("input_file", help="Path to the input MIDI file")
    parser.add_argument("--output-dir", default="segments", help="Output directory for segments (default: segments)")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return
    
    output_dir = Path(args.output_dir)
    segment_midi(input_file, output_dir)


if __name__ == "__main__":
    main()
