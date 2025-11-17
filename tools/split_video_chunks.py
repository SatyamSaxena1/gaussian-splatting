#!/usr/bin/env python3
"""
Split video into overlapping chunks for parallel multi-GPU depth estimation.

This script prepares video chunks with temporal overlap to enable processing
on multiple GPUs while maintaining temporal consistency for Video Depth Anything.

Usage:
    python split_video_chunks.py --input video.mp4 --output_dir chunks/ --num_gpus 5
"""

import argparse
import json
import os
import cv2
from pathlib import Path
from typing import List, Tuple
import numpy as np


def get_video_info(video_path: str) -> Tuple[int, float, int, int]:
    """Extract video metadata."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return total_frames, fps, width, height


def calculate_chunk_boundaries(total_frames: int, num_chunks: int, overlap: int = 32) -> List[Tuple[int, int]]:
    """
    Calculate frame boundaries for each chunk with overlap.
    
    Args:
        total_frames: Total number of frames in video
        num_chunks: Number of chunks to split into (typically num_gpus)
        overlap: Number of frames to overlap between chunks (default 32 for VDA temporal window)
    
    Returns:
        List of (start_frame, end_frame) tuples for each chunk
    """
    # Calculate base chunk size
    base_chunk_size = total_frames // num_chunks
    
    chunks = []
    for i in range(num_chunks):
        # Calculate start frame (with overlap from previous chunk)
        if i == 0:
            start_frame = 0
        else:
            # Start 'overlap' frames before the boundary
            start_frame = max(0, i * base_chunk_size - overlap)
        
        # Calculate end frame (with overlap into next chunk)
        if i == num_chunks - 1:
            end_frame = total_frames
        else:
            # End 'overlap' frames after the boundary
            end_frame = min(total_frames, (i + 1) * base_chunk_size + overlap)
        
        chunks.append((start_frame, end_frame))
    
    return chunks


def extract_chunk_frames(video_path: str, start_frame: int, end_frame: int, output_dir: Path) -> str:
    """
    Extract frames from video for a specific chunk.
    
    Args:
        video_path: Path to input video
        start_frame: Starting frame index
        end_frame: Ending frame index (exclusive)
        output_dir: Directory to save extracted frames
    
    Returns:
        Path to the chunk directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_idx = start_frame
    local_idx = 0
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame with zero-padded index
        frame_path = output_dir / f"frame_{local_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        frame_idx += 1
        local_idx += 1
    
    cap.release()
    return str(output_dir)


def create_chunk_metadata(chunks: List[Tuple[int, int]], output_dir: Path, 
                          video_info: dict, overlap: int = 32) -> str:
    """
    Create metadata file describing chunk boundaries and merge instructions.
    
    Args:
        chunks: List of (start_frame, end_frame) tuples
        output_dir: Output directory for metadata
        video_info: Dictionary with video metadata
        overlap: Overlap size in frames
    
    Returns:
        Path to metadata JSON file
    """
    metadata = {
        "video_info": video_info,
        "num_chunks": len(chunks),
        "overlap_frames": overlap,
        "chunks": []
    }
    
    for i, (start, end) in enumerate(chunks):
        # Determine which frames to keep after processing (discard overlap regions)
        if i == 0:
            # First chunk: keep all frames
            keep_start = 0
            keep_end = end - start
        elif i == len(chunks) - 1:
            # Last chunk: discard overlap at start
            keep_start = overlap
            keep_end = end - start
        else:
            # Middle chunks: discard overlap at start, keep overlap at end
            keep_start = overlap
            keep_end = end - start
        
        chunk_info = {
            "chunk_id": i,
            "global_start_frame": start,
            "global_end_frame": end,
            "num_frames": end - start,
            "keep_start_local": keep_start,
            "keep_end_local": keep_end,
            "gpu_assignment": i % len(chunks)  # Default: one chunk per GPU
        }
        metadata["chunks"].append(chunk_info)
    
    metadata_path = output_dir / "chunks_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return str(metadata_path)


def main():
    parser = argparse.ArgumentParser(description="Split video into chunks for multi-GPU depth estimation")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output_dir", required=True, help="Output directory for chunks")
    parser.add_argument("--num_gpus", type=int, default=5, help="Number of GPUs/chunks (default: 5)")
    parser.add_argument("--overlap", type=int, default=32, help="Overlap frames between chunks (default: 32)")
    parser.add_argument("--extract_frames", action="store_true", help="Extract frames to disk (memory intensive)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing video: {args.input}")
    total_frames, fps, width, height = get_video_info(args.input)
    
    video_info = {
        "input_path": args.input,
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": total_frames / fps
    }
    
    print(f"Video info: {total_frames} frames @ {fps:.2f} FPS ({video_info['duration_seconds']:.1f}s)")
    print(f"Resolution: {width}x{height}")
    
    # Calculate chunk boundaries
    print(f"\nCalculating chunk boundaries for {args.num_gpus} GPUs with {args.overlap}-frame overlap...")
    chunks = calculate_chunk_boundaries(total_frames, args.num_gpus, args.overlap)
    
    print(f"\nChunk boundaries:")
    for i, (start, end) in enumerate(chunks):
        duration = (end - start) / fps
        print(f"  Chunk {i}: frames {start:5d}-{end:5d} ({end-start:4d} frames, {duration:.1f}s)")
    
    # Create metadata
    metadata_path = create_chunk_metadata(chunks, output_dir, video_info, args.overlap)
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Optionally extract frames
    if args.extract_frames:
        print("\nExtracting frames to disk...")
        for i, (start, end) in enumerate(chunks):
            chunk_dir = output_dir / f"chunk_{i:02d}"
            print(f"  Extracting chunk {i} to {chunk_dir}...")
            extract_chunk_frames(args.input, start, end, chunk_dir)
        print("Frame extraction complete!")
    else:
        print("\nSkipping frame extraction (use --extract_frames to extract)")
        print("Parallel processing script will read frames directly from video.")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total frames: {total_frames}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Avg chunk size: {total_frames // len(chunks)} frames")
    print(f"  Overlap: {args.overlap} frames")
    print(f"  Expected speedup: ~{args.num_gpus:.1f}x")
    print(f"{'='*60}")
    
    print(f"\nNext steps:")
    print(f"  1. Run: python tools/process_chunks_parallel.py --metadata {metadata_path}")
    print(f"  2. Run: python tools/merge_depth_chunks.py --metadata {metadata_path}")


if __name__ == "__main__":
    main()
