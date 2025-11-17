#!/usr/bin/env python3
"""
Validation script for Video Depth Anything multi-GPU pipeline.

This script runs basic tests to verify all components are working correctly
before processing the full pantograph scene.

Usage:
    python validate_vda_pipeline.py
"""

import sys
import os
from pathlib import Path
import subprocess
import json


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_status(message, status="info"):
    """Print colored status message."""
    if status == "success":
        print(f"{Colors.GREEN}✓{Colors.RESET} {message}")
    elif status == "error":
        print(f"{Colors.RED}✗{Colors.RESET} {message}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠{Colors.RESET} {message}")
    else:
        print(f"{Colors.BLUE}ℹ{Colors.RESET} {message}")


def check_python_packages():
    """Check if required Python packages are installed."""
    print("\n" + "="*60)
    print("Checking Python Dependencies")
    print("="*60)
    
    packages = {
        "torch": "PyTorch",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "tqdm": "tqdm"
    }
    
    all_good = True
    for module, name in packages.items():
        try:
            if module == "cv2":
                import cv2
                version = cv2.__version__
            elif module == "torch":
                import torch
                version = torch.__version__
            elif module == "numpy":
                import numpy as np
                version = np.__version__
            elif module == "tqdm":
                import tqdm
                version = tqdm.__version__
            
            print_status(f"{name} {version}", "success")
        except ImportError:
            print_status(f"{name} not installed", "error")
            all_good = False
    
    return all_good


def check_cuda():
    """Check CUDA availability."""
    print("\n" + "="*60)
    print("Checking CUDA & GPU")
    print("="*60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print_status(f"CUDA available: {torch.version.cuda}", "success")
            print_status(f"GPUs detected: {torch.cuda.device_count()}", "success")
            
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")
            
            return True
        else:
            print_status("CUDA not available", "error")
            return False
    
    except Exception as e:
        print_status(f"Error checking CUDA: {e}", "error")
        return False


def check_tools_exist():
    """Check if pipeline tools exist."""
    print("\n" + "="*60)
    print("Checking Pipeline Tools")
    print("="*60)
    
    tools_dir = Path("tools")
    required_tools = [
        "split_video_chunks.py",
        "process_chunks_parallel.py",
        "merge_depth_chunks.py",
        "apply_vda_to_tracking.py",
        "setup_vda_multigpu.sh"
    ]
    
    all_good = True
    for tool in required_tools:
        tool_path = tools_dir / tool
        if tool_path.exists():
            print_status(f"{tool}", "success")
        else:
            print_status(f"{tool} not found", "error")
            all_good = False
    
    return all_good


def check_vda_installation():
    """Check if Video-Depth-Anything is installed."""
    print("\n" + "="*60)
    print("Checking Video-Depth-Anything Installation")
    print("="*60)
    
    vda_dir = Path("Video-Depth-Anything")
    
    if not vda_dir.exists():
        print_status("Video-Depth-Anything not found", "warning")
        print("  Run: ./tools/setup_vda_multigpu.sh")
        return False
    
    print_status(f"Repository found: {vda_dir}", "success")
    
    # Check for key files
    key_files = [
        "run.py",
        "run_streaming.py",
        "checkpoints"
    ]
    
    all_good = True
    for file in key_files:
        file_path = vda_dir / file
        if file_path.exists():
            print_status(f"  {file}", "success")
        else:
            print_status(f"  {file} missing", "warning")
            all_good = False
    
    # Check for models
    checkpoints = vda_dir / "checkpoints"
    if checkpoints.exists():
        models = list(checkpoints.glob("*.pth"))
        if models:
            print_status(f"  Found {len(models)} model(s)", "success")
            for model in models:
                print(f"    - {model.name}")
        else:
            print_status("  No models found in checkpoints/", "warning")
            print("  Run setup script to download models")
            all_good = False
    
    return all_good


def test_split_script():
    """Test split_video_chunks.py with --help."""
    print("\n" + "="*60)
    print("Testing split_video_chunks.py")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, "tools/split_video_chunks.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print_status("Script runs without errors", "success")
            if "--input" in result.stdout and "--num_gpus" in result.stdout:
                print_status("Required arguments present", "success")
                return True
            else:
                print_status("Script output unexpected", "warning")
                return False
        else:
            print_status(f"Script failed with code {result.returncode}", "error")
            print(result.stderr)
            return False
    
    except Exception as e:
        print_status(f"Error testing script: {e}", "error")
        return False


def test_process_script():
    """Test process_chunks_parallel.py with --help."""
    print("\n" + "="*60)
    print("Testing process_chunks_parallel.py")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, "tools/process_chunks_parallel.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print_status("Script runs without errors", "success")
            if "--metadata" in result.stdout and "--gpu_ids" in result.stdout:
                print_status("Required arguments present", "success")
                return True
            else:
                print_status("Script output unexpected", "warning")
                return False
        else:
            print_status(f"Script failed with code {result.returncode}", "error")
            return False
    
    except Exception as e:
        print_status(f"Error testing script: {e}", "error")
        return False


def test_merge_script():
    """Test merge_depth_chunks.py with --help."""
    print("\n" + "="*60)
    print("Testing merge_depth_chunks.py")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, "tools/merge_depth_chunks.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print_status("Script runs without errors", "success")
            if "--metadata" in result.stdout and "--depth_dir" in result.stdout:
                print_status("Required arguments present", "success")
                return True
            else:
                print_status("Script output unexpected", "warning")
                return False
        else:
            print_status(f"Script failed with code {result.returncode}", "error")
            return False
    
    except Exception as e:
        print_status(f"Error testing script: {e}", "error")
        return False


def check_test_data():
    """Check if test data (pantograph scene) exists."""
    print("\n" + "="*60)
    print("Checking Test Data")
    print("="*60)
    
    data_dir = Path("data/pantograph_scene")
    
    if not data_dir.exists():
        print_status("Pantograph scene directory not found", "warning")
        return False
    
    print_status(f"Data directory found: {data_dir}", "success")
    
    # Check for video file
    video_files = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.avi"))
    if video_files:
        print_status(f"Found video file: {video_files[0].name}", "success")
    else:
        print_status("No video file found", "warning")
    
    # Check for detections
    detections_file = data_dir / "contact_detections.json"
    if detections_file.exists():
        print_status("Detection file found", "success")
        try:
            with open(detections_file) as f:
                data = json.load(f)
                num_frames = len(data.get("frames", []))
                print(f"  Frames with detections: {num_frames}")
        except Exception as e:
            print_status(f"Error reading detections: {e}", "warning")
    else:
        print_status("Detection file not found", "warning")
    
    return True


def print_summary(results):
    """Print validation summary."""
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for test, result in results.items():
        status = "success" if result else "error"
        print_status(test, status)
    
    print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}All tests passed! Pipeline is ready to use.{Colors.RESET}")
        print("\nNext steps:")
        print("  1. Run setup: ./tools/setup_vda_multigpu.sh")
        print("  2. Process video: ./run_vda_pipeline.sh")
        print("  3. Check documentation: tools/VDA_MULTIGPU_QUICKSTART.md")
        return True
    else:
        print(f"\n{Colors.RED}Some tests failed. See errors above.{Colors.RESET}")
        print("\nTroubleshooting:")
        print("  1. Install missing dependencies: pip install torch opencv-python numpy tqdm")
        print("  2. Run setup script: ./tools/setup_vda_multigpu.sh")
        print("  3. Check CUDA installation: nvidia-smi")
        return False


def main():
    print(f"\n{Colors.BLUE}{'='*60}")
    print("Video Depth Anything Pipeline Validation")
    print(f"{'='*60}{Colors.RESET}\n")
    
    results = {}
    
    # Run all checks
    results["Python Dependencies"] = check_python_packages()
    results["CUDA & GPU"] = check_cuda()
    results["Pipeline Tools"] = check_tools_exist()
    results["VDA Installation"] = check_vda_installation()
    results["Split Script"] = test_split_script()
    results["Process Script"] = test_process_script()
    results["Merge Script"] = test_merge_script()
    results["Test Data"] = check_test_data()
    
    # Print summary
    success = print_summary(results)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
