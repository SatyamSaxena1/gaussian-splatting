![[Video: Pantograph Processing Pipeline|240]](../assets/charts/pantograph_pipeline.webm)

- **Video 1:** `WhatsApp Video 2025-10-27 at 2.00.28 PM.mp4`
  - Frame extraction → motion masks → MiDaS depth → motion + depth fusion → static scene data for COLMAP/Gaussian Splatting → dynamic foreground (pantograph)
- **Static scene:** `static_frames/` for reconstruction; `static_masks_refined/` for visibility control
- **Dynamic scene:** `dynamic_frames/` and `dynamic_masks_refined/` for pantograph contact tracking and pose estimation
