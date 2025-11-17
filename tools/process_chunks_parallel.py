#!/usr/bin/env python3
"""
Process video chunks in parallel across multiple GPUs using Video Depth Anything.

This script orchestrates parallel depth estimation on multiple GPUs by reading
the chunk metadata and launching separate processes with CUDA_VISIBLE_DEVICES.

Usage:
    python process_chunks_parallel.py --metadata chunks/chunks_metadata.json --vda_path /path/to/Video-Depth-Anything
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict
import multiprocessing as mp


def load_metadata(metadata_path: str) -> Dict:
    """Load chunk metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def create_vda_command(chunk_id: int, chunk_info: Dict, metadata: Dict, 
                       vda_path: Path, gpu_id: int, output_dir: Path,
                       model_size: str = "vits", streaming: bool = True) -> List[str]:
    """
    Create Video Depth Anything command for processing a chunk.
    
    Args:
        chunk_id: Chunk identifier
        chunk_info: Chunk metadata from JSON
        metadata: Full metadata dictionary
        vda_path: Path to Video-Depth-Anything repository
        gpu_id: GPU device ID to use
        output_dir: Output directory for depth maps
        model_size: Model size (vits/vitb/vitl)
        streaming: Use streaming mode (lower memory)
    
    Returns:
        Command as list of strings
    """
    video_path = metadata["video_info"]["input_path"]
    start_frame = chunk_info["global_start_frame"]
    end_frame = chunk_info["global_end_frame"]
    num_frames = chunk_info["num_frames"]
    
    chunk_output_dir = output_dir / f"chunk_{chunk_id:02d}"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use streaming mode for lower memory usage (important for 1660 Ti with 6GB)
    script_name = "run_streaming.py" if streaming else "run.py"
    script_path = vda_path / script_name
    
    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        str(script_path),
        "--encoder", model_size,  # vits, vitb, or vitl
        "--video-path", video_path,
        "--output-dir", str(chunk_output_dir),
        "--start-frame", str(start_frame),
        "--end-frame", str(end_frame),
    ]
    
    # Add metric depth if available
    if model_size in ["vits", "vitb", "vitl"]:
        cmd.append("--pred-only")  # Skip visualization, only save depth
        cmd.append("--grayscale")  # Save as grayscale depth maps
    
    return cmd


def process_chunk_worker(args_tuple):
    """
    Worker function to process a single chunk on a specific GPU.
    
    Args:
        args_tuple: Tuple of (chunk_id, chunk_info, metadata, vda_path, gpu_id, output_dir, model_size, streaming)
    """
    chunk_id, chunk_info, metadata, vda_path, gpu_id, output_dir, model_size, streaming = args_tuple
    
    print(f"[Chunk {chunk_id}] Starting on GPU {gpu_id}")
    start_time = time.time()
    
    # Create command
    cmd = create_vda_command(chunk_id, chunk_info, metadata, vda_path, gpu_id, 
                             output_dir, model_size, streaming)
    
    # Set environment to use specific GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Log command to file
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"chunk_{chunk_id:02d}_gpu_{gpu_id}.log"
    
    try:
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"GPU: {gpu_id}\n")
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            f.flush()
            
            # Run process
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait for completion
            return_code = process.wait()
            
            elapsed = time.time() - start_time
            f.write(f"\n\n{'='*60}\n")
            f.write(f"Return code: {return_code}\n")
            f.write(f"Elapsed time: {elapsed:.1f}s\n")
            f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if return_code == 0:
            print(f"[Chunk {chunk_id}] ✓ Complete in {elapsed:.1f}s (GPU {gpu_id})")
            return {"chunk_id": chunk_id, "success": True, "elapsed": elapsed, "gpu": gpu_id}
        else:
            print(f"[Chunk {chunk_id}] ✗ Failed with code {return_code} (GPU {gpu_id})")
            return {"chunk_id": chunk_id, "success": False, "elapsed": elapsed, "gpu": gpu_id, "error": f"Exit code {return_code}"}
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Chunk {chunk_id}] ✗ Exception: {e} (GPU {gpu_id})")
        return {"chunk_id": chunk_id, "success": False, "elapsed": elapsed, "gpu": gpu_id, "error": str(e)}


def process_chunks_parallel(metadata_path: str, vda_path: str, output_dir: str,
                            gpu_ids: List[int], model_size: str = "vits", 
                            streaming: bool = True):
    """
    Process all chunks in parallel across available GPUs.
    
    Args:
        metadata_path: Path to chunks metadata JSON
        vda_path: Path to Video-Depth-Anything repository
        output_dir: Output directory for depth maps
        gpu_ids: List of GPU device IDs to use
        model_size: Model size (vits/vitb/vitl)
        streaming: Use streaming mode
    """
    # Load metadata
    metadata = load_metadata(metadata_path)
    chunks = metadata["chunks"]
    
    print(f"{'='*60}")
    print(f"Video Depth Anything - Multi-GPU Parallel Processing")
    print(f"{'='*60}")
    print(f"Video: {metadata['video_info']['input_path']}")
    print(f"Total frames: {metadata['video_info']['total_frames']}")
    print(f"Chunks: {len(chunks)}")
    print(f"GPUs: {len(gpu_ids)} ({gpu_ids})")
    print(f"Model: {model_size}")
    print(f"Mode: {'streaming' if streaming else 'standard'}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    vda_path = Path(vda_path)
    if not vda_path.exists():
        raise FileNotFoundError(f"Video-Depth-Anything not found at: {vda_path}")
    
    # Prepare worker arguments
    worker_args = []
    for i, chunk_info in enumerate(chunks):
        gpu_id = gpu_ids[i % len(gpu_ids)]  # Cycle through available GPUs
        args = (i, chunk_info, metadata, vda_path, gpu_id, output_path, model_size, streaming)
        worker_args.append(args)
    
    # Process chunks in parallel
    start_time = time.time()
    
    # Use multiprocessing pool with number of workers = number of GPUs
    with mp.Pool(processes=len(gpu_ids)) as pool:
        results = pool.map(process_chunk_worker, worker_args)
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average per chunk: {total_time/len(results):.1f}s")
    
    if successful > 0:
        avg_chunk_time = sum(r["elapsed"] for r in results if r["success"]) / successful
        print(f"Avg successful chunk time: {avg_chunk_time:.1f}s")
        
        # Estimate speedup vs sequential
        sequential_estimate = avg_chunk_time * len(results)
        speedup = sequential_estimate / total_time
        print(f"Estimated speedup: {speedup:.2f}x")
    
    print(f"{'='*60}")
    
    # Save results
    results_path = output_path / "processing_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "total_time": total_time,
            "chunks_processed": len(results),
            "successful": successful,
            "failed": failed,
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    if failed > 0:
        print(f"\nFailed chunks:")
        for r in results:
            if not r["success"]:
                print(f"  Chunk {r['chunk_id']}: {r.get('error', 'Unknown error')}")
        print(f"\nCheck logs in: {output_path / 'logs'}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Process video chunks in parallel with VDA")
    parser.add_argument("--metadata", required=True, help="Path to chunks metadata JSON")
    parser.add_argument("--vda_path", required=True, help="Path to Video-Depth-Anything repository")
    parser.add_argument("--output_dir", default="depth_chunks", help="Output directory for depth maps")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1, 2, 3, 4], 
                       help="GPU device IDs to use (default: 0 1 2 3 4)")
    parser.add_argument("--model_size", choices=["vits", "vitb", "vitl"], default="vits",
                       help="Model size (vits=28M/7.5GB, vitb=113M/10GB, vitl=382M/14GB)")
    parser.add_argument("--no_streaming", action="store_true", 
                       help="Disable streaming mode (requires more memory)")
    
    args = parser.parse_args()
    
    success = process_chunks_parallel(
        metadata_path=args.metadata,
        vda_path=args.vda_path,
        output_dir=args.output_dir,
        gpu_ids=args.gpu_ids,
        model_size=args.model_size,
        streaming=not args.no_streaming
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
