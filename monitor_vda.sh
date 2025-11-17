#!/bin/bash
echo "VDA Processing Monitor"
echo "======================"
echo ""
echo "Processes running:"
ps aux | grep "run_streaming_save_frames.py" | grep -v grep | wc -l
echo ""
echo "GPU Usage:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo ""
echo "Depth frames generated per chunk:"
for i in 00 01 02 03 04; do
  if [ -d "data/pantograph_scene/vda_depth_final/chunk_$i/depth_frames" ]; then
    count=$(find data/pantograph_scene/vda_depth_final/chunk_$i/depth_frames -name "*.png" 2>/dev/null | wc -l)
    echo "  Chunk $i: $count PNG frames"
  else
    echo "  Chunk $i: not started"
  fi
done
echo ""
total=$(find data/pantograph_scene/vda_depth_final -name "*.png" 2>/dev/null | wc -l)
progress=$(awk "BEGIN {printf \"%.1f\", ($total/6553)*100}")
echo "Total: $total / 6553 frames ($progress%)"
