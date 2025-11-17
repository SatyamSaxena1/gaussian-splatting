# Pipeline Checklist for a New Video

1. **Stage your inputs**
   - Place the video somewhere accessible (e.g. `/home/akash_gemperts/Videos/new_clip.mp4`).
   - Decide on a short scene name; all outputs will live under `data/<scene_name>`.

2. **Extract frames**
   ```bash
   ./process_webcam_video.sh --video /path/new_clip.mp4 --output-name living_room --fps 2
   ```
   - A progress bar appears if `ffprobe` is available.
   - Frames are written to `data/living_room/input/`.

3. **Run COLMAP (requires GPU + escalated shell)**
   ```bash
   COLMAP_GPU=0 XDG_CACHE_HOME=$PWD/.mamba-cache python3 convert.py -s data/living_room \
     --colmap_executable ./run_colmap_local.sh \
     --matching sequential --sequential_overlap 3 \
     --sift_max_image_size 1600 --sift_gpu_index 0 --sift_num_threads 8
   ```
   - After completion, check `data/living_room/distorted/sparse/0` for `cameras.bin`, `images.bin`, and `points3D.bin`.

4. **Launch training (after COLMAP succeeds)**
   ```bash
   export LD_LIBRARY_PATH=/home/akash_gemperts/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
   ./gpu_env.sh python3 train.py -s data/living_room --data_device cuda
   ```

5. **Render / Visualize**
   ```bash
 ./gpu_env.sh python3 render.py --model_path data/living_room --iteration 30000
 PYTHONPATH=tools/nerfvis:./tools/nerfvis ./gpu_env.sh python3 tools/run_nerfvis.py \
    --model data/living_room --iteration 30000 --outdir data/living_room/nerfvis_export
  ```
  - Open the generated PNGs under `output/.../renders/`.
  - View the interactive export at `data/living_room/nerfvis_export/index.html`.

7. **(Optional) Real-time SIBR viewer**
   - Build once:
     ```bash
     sudo apt-get install libopengl-dev libglew-dev libassimp-dev libboost-all-dev \
       libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev \
       libeigen3-dev libxxf86vm-dev libembree-dev
     cd SIBR_viewers
     cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
     cmake --build build -j$(nproc) --target install
     ```
   - Run on a trained model from a GUI session:
     ```bash
     tools/run_sibr_viewer.sh --model output/<run_id> --data data/<scene_name>
     ```

6. **Archive results (optional)**
   - Copy the entire `output/<run_id>` directory and `point_cloud.ply`.
   - Zip the nerfvis export if you need to share it.
