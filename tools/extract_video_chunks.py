#!/usr/bin/env python3
"""
Extract video chunks as separate video files for VDA processing.
"""

import argparse
import json
import subprocess
from pathlib import Path


def extract_chunk(video_path: str, start_frame: int, end_frame: int, output_path: str, fps: float):
    """Extract a chunk of video using ffmpeg."""
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        output_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    with open(args.metadata) as f:
        metadata = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = metadata["video_info"]["input_path"]
    fps = metadata["video_info"]["fps"]
    
    print(f"Extracting {len(metadata['chunks'])} video chunks...")
    
    for chunk_info in metadata["chunks"]:
        chunk_id = chunk_info["chunk_id"]
        start = chunk_info["global_start_frame"]
        end = chunk_info["global_end_frame"]
        
        output_path = output_dir / f"chunk_{chunk_id:02d}.mp4"
        print(f"  Chunk {chunk_id}: frames {start}-{end} -> {output_path.name}")
        
        extract_chunk(video_path, start, end, str(output_path), fps)
    
    print("Done!")


if __name__ == "__main__":
    main()
