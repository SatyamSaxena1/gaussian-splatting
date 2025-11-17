#!/usr/bin/env python3
"""
Merge depth map chunks from parallel GPU processing.

This script merges overlapping depth chunks by discarding overlap regions
and reassembling into a single continuous depth map sequence.

Usage:
    python merge_depth_chunks.py --metadata chunks/chunks_metadata.json --depth_dir depth_chunks/ --output_dir merged_depth/
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List
import numpy as np
import cv2
from tqdm import tqdm


def load_metadata(metadata_path: str) -> Dict:
    """Load chunk metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def find_depth_files(chunk_dir: Path) -> List[Path]:
    """Find all depth map files in a chunk directory."""
    # Check if there's a depth_frames subdirectory (VDA format)
    depth_frames_dir = chunk_dir / "depth_frames"
    if depth_frames_dir.exists():
        search_dir = depth_frames_dir
    else:
        search_dir = chunk_dir
    
    # Look for common depth map file patterns
    patterns = ["*.png", "*.npy", "*.npz", "*.jpg", "*.tif"]
    
    files = []
    for pattern in patterns:
        files.extend(sorted(search_dir.glob(pattern)))
    
    # Filter out visualization files (usually contain 'vis' or 'color' in name)
    depth_files = [f for f in files if 'vis' not in f.stem.lower() and 'color' not in f.stem.lower()]
    
    return sorted(depth_files)


def load_depth_map(file_path: Path) -> np.ndarray:
    """Load depth map from various file formats."""
    ext = file_path.suffix.lower()
    
    if ext == '.npy':
        return np.load(file_path)
    elif ext == '.npz':
        data = np.load(file_path)
        # Try common keys (including VDA's 'depths' key)
        for key in ['depths', 'depth', 'depth_map', 'arr_0']:
            if key in data:
                # If data is 3D (frames, height, width), return all frames
                return data[key]
        raise ValueError(f"Could not find depth data in NPZ file: {file_path}")
    elif ext in ['.png', '.jpg', '.tif', '.tiff']:
        # Load as grayscale
        depth = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise ValueError(f"Failed to load image: {file_path}")
        return depth
    else:
        raise ValueError(f"Unsupported depth map format: {ext}")


def save_depth_map(depth: np.ndarray, file_path: Path, format: str = "png16"):
    """
    Save depth map in specified format.
    
    Args:
        depth: Depth map array
        file_path: Output file path
        format: Output format (png16, png8, npy, npz)
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "npy":
        np.save(file_path.with_suffix('.npy'), depth)
    elif format == "npz":
        np.savez_compressed(file_path.with_suffix('.npz'), depth=depth)
    elif format == "png16":
        # Save as 16-bit PNG for maximum precision
        if depth.dtype != np.uint16:
            # Normalize to 16-bit range
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 65535).astype(np.uint16)
        else:
            depth_norm = depth
        cv2.imwrite(str(file_path.with_suffix('.png')), depth_norm)
    elif format == "png8":
        # Save as 8-bit PNG (more compressed, less precision)
        if depth.dtype != np.uint8:
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        else:
            depth_norm = depth
        cv2.imwrite(str(file_path.with_suffix('.png')), depth_norm)
    else:
        raise ValueError(f"Unsupported output format: {format}")


def merge_chunks(metadata: Dict, depth_dir: Path, output_dir: Path, 
                 output_format: str = "png16", create_video: bool = False):
    """
    Merge depth chunks by discarding overlap regions.
    
    Args:
        metadata: Chunk metadata dictionary
        depth_dir: Directory containing chunk depth maps
        output_dir: Output directory for merged depth maps
        output_format: Output format (png16, png8, npy, npz)
        create_video: Create visualization video from merged depth
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunks = metadata["chunks"]
    total_frames = metadata["video_info"]["total_frames"]
    
    print(f"{'='*60}")
    print(f"Merging Depth Chunks")
    print(f"{'='*60}")
    print(f"Total frames: {total_frames}")
    print(f"Chunks: {len(chunks)}")
    print(f"Output format: {output_format}")
    print(f"{'='*60}\n")
    
    # Track global frame index
    global_frame_idx = 0
    
    # Process each chunk
    for chunk_id, chunk_info in enumerate(chunks):
        chunk_dir = depth_dir / f"chunk_{chunk_id:02d}"
        
        if not chunk_dir.exists():
            print(f"✗ Chunk {chunk_id} directory not found: {chunk_dir}")
            continue
        
        print(f"Processing chunk {chunk_id}...")
        
        # Find depth files in chunk
        depth_files = find_depth_files(chunk_dir)
        
        if not depth_files:
            print(f"  ✗ No depth files found in {chunk_dir}")
            continue
        
        # Determine which frames to keep (discard overlap)
        keep_start = chunk_info["keep_start_local"]
        keep_end = chunk_info["keep_end_local"]
        
        print(f"  Chunk has {len(depth_files)} depth file(s)")
        
        # Check if we have a single NPZ file with all frames
        if len(depth_files) == 1 and depth_files[0].suffix.lower() == '.npz':
            print(f"  Loading all frames from NPZ: {depth_files[0].name}")
            try:
                depths_all = load_depth_map(depth_files[0])
                num_frames = depths_all.shape[0] if len(depths_all.shape) == 3 else 1
                print(f"  NPZ contains {num_frames} frames")
                print(f"  Keeping local frames {keep_start} to {keep_end}")
                
                # Extract and save kept frames
                kept_count = 0
                for local_idx in range(keep_start, min(keep_end, num_frames)):
                    depth = depths_all[local_idx] if len(depths_all.shape) == 3 else depths_all
                    
                    # Save with global frame index
                    output_file = output_dir / f"depth_{global_frame_idx:06d}"
                    save_depth_map(depth, output_file, output_format)
                    
                    global_frame_idx += 1
                    kept_count += 1
                
                print(f"  ✓ Kept {kept_count} frames (global frames {global_frame_idx - kept_count} to {global_frame_idx})")
            except Exception as e:
                print(f"  ✗ Error processing NPZ: {e}")
        else:
            # Handle individual frame files
            print(f"  Keeping local frames {keep_start} to {keep_end}")
            
            # Copy/convert kept frames to output
            kept_count = 0
            for local_idx in range(keep_start, keep_end):
                if local_idx >= len(depth_files):
                    print(f"  ✗ Warning: Local index {local_idx} exceeds available files ({len(depth_files)})")
                    break
                
                # Load depth map
                try:
                    depth = load_depth_map(depth_files[local_idx])
                except Exception as e:
                    print(f"  ✗ Error loading {depth_files[local_idx]}: {e}")
                    continue
                
                # Save with global frame index
                output_file = output_dir / f"depth_{global_frame_idx:06d}"
                save_depth_map(depth, output_file, output_format)
                
                global_frame_idx += 1
                kept_count += 1
            
            print(f"  ✓ Kept {kept_count} frames (global frames {global_frame_idx - kept_count} to {global_frame_idx})")
    
    print(f"\n{'='*60}")
    print(f"Merge Complete!")
    print(f"{'='*60}")
    print(f"Total merged frames: {global_frame_idx}")
    print(f"Expected frames: {total_frames}")
    
    if global_frame_idx != total_frames:
        print(f"⚠️  Warning: Frame count mismatch!")
        print(f"   Expected {total_frames}, got {global_frame_idx}")
    else:
        print(f"✓ Frame count matches expected")
    
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    
    # Create visualization video if requested
    if create_video:
        print(f"\nCreating visualization video...")
        create_depth_video(output_dir, metadata["video_info"]["fps"], output_format)


def create_depth_video(depth_dir: Path, fps: float, depth_format: str = "png16"):
    """
    Create colorized visualization video from depth maps.
    
    Args:
        depth_dir: Directory containing merged depth maps
        fps: Video frame rate
        depth_format: Format of depth maps
    """
    # Find all depth files
    if depth_format.startswith("png"):
        depth_files = sorted(depth_dir.glob("depth_*.png"))
    elif depth_format == "npy":
        depth_files = sorted(depth_dir.glob("depth_*.npy"))
    elif depth_format == "npz":
        depth_files = sorted(depth_dir.glob("depth_*.npz"))
    else:
        print(f"Unsupported format for video creation: {depth_format}")
        return
    
    if not depth_files:
        print(f"No depth files found for video creation")
        return
    
    print(f"Creating video from {len(depth_files)} depth maps...")
    
    # Load first frame to get dimensions
    first_depth = load_depth_map(depth_files[0])
    height, width = first_depth.shape
    
    # Create video writer
    output_video = depth_dir / "depth_visualization.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    # Process frames with progress bar
    for depth_file in tqdm(depth_files, desc="Rendering video"):
        try:
            # Load depth
            depth = load_depth_map(depth_file)
            
            # Normalize and colorize
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            
            # Write frame
            writer.write(depth_color)
        
        except Exception as e:
            print(f"Error processing {depth_file}: {e}")
    
    writer.release()
    print(f"✓ Video saved: {output_video}")


def main():
    parser = argparse.ArgumentParser(description="Merge depth map chunks from parallel processing")
    parser.add_argument("--metadata", required=True, help="Path to chunks metadata JSON")
    parser.add_argument("--depth_dir", required=True, help="Directory containing chunk depth maps")
    parser.add_argument("--output_dir", required=True, help="Output directory for merged depth maps")
    parser.add_argument("--format", choices=["png16", "png8", "npy", "npz"], default="png16",
                       help="Output format (default: png16 for precision)")
    parser.add_argument("--create_video", action="store_true", help="Create visualization video")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.metadata):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")
    
    if not os.path.exists(args.depth_dir):
        raise FileNotFoundError(f"Depth directory not found: {args.depth_dir}")
    
    # Load metadata
    metadata = load_metadata(args.metadata)
    
    # Merge chunks
    merge_chunks(
        metadata=metadata,
        depth_dir=Path(args.depth_dir),
        output_dir=Path(args.output_dir),
        output_format=args.format,
        create_video=args.create_video
    )
    
    print(f"\nNext steps:")
    print(f"  1. Verify frame count: ls {args.output_dir}/depth_*.{args.format.replace('png16', 'png').replace('png8', 'png')} | wc -l")
    print(f"  2. Apply depth to trajectories: python tools/apply_vda_to_tracking.py")


if __name__ == "__main__":
    main()
