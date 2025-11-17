#!/usr/bin/env python3
"""
Process extracted video chunks in parallel using Video Depth Anything.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import multiprocessing as mp


def process_chunk_simple(args):
    """Process a single video chunk."""
    chunk_id, video_path, output_dir, vda_path, gpu_id, model_size = args
    
    log_dir = output_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"chunk_{chunk_id:02d}_gpu_{gpu_id}.log"
    
    chunk_output = output_dir / f"chunk_{chunk_id:02d}"
    chunk_output.mkdir(parents=True, exist_ok=True)
    
    # Use absolute paths - VDA will work from its own directory
    abs_video = video_path.resolve()
    abs_output = chunk_output.resolve()
    
    cmd = [
        sys.executable,
        "run_streaming.py",
        "--input_video", str(abs_video),
        "--output_dir", str(abs_output),
        "--encoder", model_size,
        "--grayscale"
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[Chunk {chunk_id}] Starting on GPU {gpu_id}")
    start_time = time.time()
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(vda_path))
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[Chunk {chunk_id}] ✓ Complete in {elapsed:.1f}s (GPU {gpu_id})")
            return {"chunk_id": chunk_id, "success": True, "elapsed": elapsed}
        else:
            print(f"[Chunk {chunk_id}] ✗ Failed with code {result.returncode} (GPU {gpu_id})")
            return {"chunk_id": chunk_id, "success": False, "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Chunk {chunk_id}] ✗ Exception: {e} (GPU {gpu_id})")
        return {"chunk_id": chunk_id, "success": False, "elapsed": elapsed}


def main():
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_chunks_dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--vda_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--model_size", default="vits")
    args = parser.parse_args()
    
    video_chunks_dir = Path(args.video_chunks_dir)
    vda_path = Path(args.vda_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.metadata) as f:
        metadata = json.load(f)
    
    print(f"{'='*60}")
    print(f"Video Depth Anything - Multi-GPU Processing")
    print(f"{'='*60}")
    print(f"Chunks: {len(metadata['chunks'])}")
    print(f"GPUs: {len(args.gpu_ids)} ({args.gpu_ids})")
    print(f"Model: {args.model_size}")
    print(f"{'='*60}\n")
    
    # Prepare tasks
    tasks = []
    for chunk_info in metadata["chunks"]:
        chunk_id = chunk_info["chunk_id"]
        video_file = video_chunks_dir / f"chunk_{chunk_id:02d}.mp4"
        gpu_id = args.gpu_ids[chunk_id % len(args.gpu_ids)]
        
        tasks.append((chunk_id, video_file, output_dir, vda_path, gpu_id, args.model_size))
    
    # Process in parallel
    start_time = time.time()
    with mp.Pool(processes=len(args.gpu_ids)) as pool:
        results = pool.map(process_chunk_simple, tasks)
    
    total_time = time.time() - start_time
    
    successful = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"Complete! {successful}/{len(results)} successful")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
