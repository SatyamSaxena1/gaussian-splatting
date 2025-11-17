# VDA Standalone Processing Guide

## Quick Start

Run the standalone script from any terminal:

```bash
cd ~/gaussian-splatting
./run_vda_standalone.sh
```

This script will:
- Process all 5 chunks sequentially (one at a time)
- Use dedicated GPUs (Chunk 00→GPU 0, Chunk 01→GPU 1, etc.)
- Save output immediately after each chunk completes
- Show colored progress with checkmarks
- Create logs: `data/pantograph_scene/vda_logs/chunk_XX_standalone.log`

## Expected Output

For each chunk, you'll see:
- Progress bar during processing (~7 minutes per chunk)
- NPZ file created: `data/pantograph_scene/vda_depth_final/chunk_XX/chunk_XX_depths.npz`
- MP4 visualizations (optional)

Total time: **~35 minutes** (7 min × 5 chunks)

## Advantages

- ✅ Runs completely independent of VSCode
- ✅ Sequential processing prevents memory issues
- ✅ Output saved immediately after each chunk
- ✅ If one chunk fails, others continue
- ✅ Clear visual feedback with colors
- ✅ Detailed logs for debugging

## Monitoring

While running, you can monitor in another terminal:

```bash
# Watch progress
tail -f ~/gaussian-splatting/data/pantograph_scene/vda_logs/chunk_00_standalone.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check which chunk is running
ps aux | grep run.py
```

## After Completion

Once all chunks complete successfully, proceed with:

```bash
cd ~/gaussian-splatting

# Merge depth chunks
python tools/merge_depth_chunks.py \
    --metadata data/pantograph_scene/vda_chunks/chunks_metadata.json \
    --depth_dir data/pantograph_scene/vda_depth_final \
    --output_dir data/pantograph_scene/vda_merged_depth \
    --format png16 \
    --create_video
```

## Troubleshooting

**Script stops unexpectedly:**
- Check logs in `data/pantograph_scene/vda_logs/`
- Verify GPU is available: `nvidia-smi`
- Ensure virtual environment is activated

**No NPZ files created:**
- The script will show which chunks failed
- Rerun just failed chunks by modifying CHUNKS array in script

**Out of memory:**
- Close other GPU applications
- Run with `watch free -h` to monitor RAM
