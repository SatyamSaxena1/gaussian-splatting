#!/usr/bin/env python3
"""
Extract depth frames from VDA visualization videos.
Since VDA processed the videos but only saved visualization MP4s,
this extracts individual frames as PNG files.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from tqdm import tqdm

def extract_depth_frames(video_path, output_dir, start_frame=0):
    """
    Extract depth frames from VDA visualization video.
    
    Args:
        video_path: Path to VDA visualization video (_vis.mp4)
        output_dir: Directory to save extracted depth frames
        start_frame: Starting frame number for naming (to handle chunks)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Frames: {total_frames}")
    print(f"  Output: {output_dir}")
    
    frame_idx = 0
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save as grayscale PNG (depth visualization is already grayscale/colored)
            # Frame numbering continues from start_frame
            output_path = output_dir / f"depth_{start_frame + frame_idx:06d}.png"
            cv2.imwrite(str(output_path), frame)
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    print(f"✓ Extracted {frame_idx} frames to {output_dir}")
    return frame_idx

def main():
    parser = argparse.ArgumentParser(description='Extract depth frames from VDA visualization videos')
    parser.add_argument('--vda_output_dir', type=str, required=True,
                        help='Directory containing chunk_XX subdirs with _vis.mp4 files')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to chunks_metadata.json')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for extracted depth frames (per chunk)')
    
    args = parser.parse_args()
    
    vda_output_dir = Path(args.vda_output_dir)
    output_base = Path(args.output_dir)
    
    # Load metadata to get frame ranges
    import json
    with open(args.metadata) as f:
        metadata = json.load(f)
    
    print(f"VDA Output Dir: {vda_output_dir}")
    print(f"Metadata: {args.metadata}")
    print(f"Output Base: {output_base}")
    print(f"Total chunks: {len(metadata['chunks'])}\n")
    
    # Process each chunk
    total_frames_extracted = 0
    for chunk in metadata['chunks']:
        chunk_id = chunk['chunk_id']
        start_frame = chunk['start_frame']
        end_frame = chunk['end_frame']
        expected_frames = end_frame - start_frame
        
        # Find the visualization video
        chunk_dir = vda_output_dir / f"chunk_{chunk_id:02d}"
        vis_video = chunk_dir / f"chunk_{chunk_id:02d}_vis.mp4"
        
        if not vis_video.exists():
            print(f"⚠ Warning: {vis_video} not found, skipping")
            continue
        
        # Create output directory for this chunk
        chunk_output_dir = output_base / f"chunk_{chunk_id:02d}"
        
        # Extract frames (starting numbering from chunk's start_frame)
        print(f"\n{'='*60}")
        print(f"Chunk {chunk_id}: Frames {start_frame}-{end_frame}")
        print(f"{'='*60}")
        
        frames_extracted = extract_depth_frames(vis_video, chunk_output_dir, start_frame)
        total_frames_extracted += frames_extracted
        
        if frames_extracted != expected_frames:
            print(f"⚠ Warning: Expected {expected_frames} frames, got {frames_extracted}")
    
    print(f"\n{'='*60}")
    print(f"✓ COMPLETE: Extracted {total_frames_extracted} total frames")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
