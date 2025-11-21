#!/usr/bin/env python3
import numpy as np
import cv2
import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {args.npz_file}...")
    data = np.load(args.npz_file)
    
    # Check keys
    print(f"Keys: {list(data.keys())}")
    if 'depth' in data:
        depths = data['depth']
    elif 'depths' in data:
        depths = data['depths']
    else:
        # Try first key
        key = list(data.keys())[0]
        depths = data[key]
        
    print(f"Depth shape: {depths.shape}")
    
    # Normalize and save
    for i, depth in enumerate(depths):
        # Normalize to 0-65535
        # VDA output is relative depth (inverse depth?)
        # We want to map min-max to 0-65535 for max precision in PNG16
        
        d_min = depth.min()
        d_max = depth.max()
        
        if d_max - d_min > 1e-6:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth)
            
        depth_uint16 = (depth_norm * 65535).astype(np.uint16)
        
        # Frame numbering usually starts at 1 in this pipeline
        # process_video.sh produces frame_00001.jpg
        frame_num = i + 1
        output_path = args.output_dir / f"depth_{frame_num:05d}.png"
        
        cv2.imwrite(str(output_path), depth_uint16)
        
        if i % 50 == 0:
            print(f"Saved frame {i}/{len(depths)}")
            
    print("Conversion complete.")

if __name__ == "__main__":
    main()
