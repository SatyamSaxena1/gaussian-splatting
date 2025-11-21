import cv2
import json
import argparse
from pathlib import Path
import sys
from tqdm import tqdm

def _parse_args():
    parser = argparse.ArgumentParser(description="Overlay metrics on visualization frames.")
    parser.add_argument("frames_dir", type=Path, help="Directory containing input frames (e.g., contact_visualizations)")
    parser.add_argument("metrics_json", type=Path, help="Path to railway_metrics.json")
    parser.add_argument("output_dir", type=Path, help="Directory to save output frames")
    return parser.parse_args()

def main():
    args = _parse_args()
    
    if not args.frames_dir.exists():
        print(f"Error: {args.frames_dir} not found", file=sys.stderr)
        return 1
        
    if not args.metrics_json.exists():
        print(f"Error: {args.metrics_json} not found", file=sys.stderr)
        return 1
        
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    with open(args.metrics_json, 'r') as f:
        data = json.load(f)
        metrics_list = data.get("metrics", [])
        
    # Index metrics by frame name
    metrics_map = {m["frame"]: m for m in metrics_list}
    
    frame_paths = sorted(list(args.frames_dir.glob("*.jpg")) + list(args.frames_dir.glob("*.png")))
    print(f"Processing {len(frame_paths)} frames...")
    
    for frame_path in tqdm(frame_paths):
        frame_name = frame_path.stem
        img = cv2.imread(str(frame_path))
        
        if img is None:
            continue
            
        # Get metrics for this frame
        # Note: frame_path might be "frame_0001", metrics might be "frame_0001"
        # Or input might be "frame_0001_vis" depending on how track_contact_yolo saved it.
        # Let's try exact match first.
        m = metrics_map.get(frame_name)
        
        if m:
            # Overlay text
            # Setup font
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (0, 255, 255) # Yellow
            bg_color = (0, 0, 0)
            
            lines = []
            if m["detected"]:
                lines.append(f"Height: {m['height_m']:.2f} m")
                lines.append(f"Stagger: {m['stagger_mm']:.0f} mm")
                lines.append(f"Dist: {m['distance_m']:.1f} m")
            else:
                lines.append("No Detection")
                
            # Draw text box
            x, y = 20, 40
            line_height = 30
            
            for line in lines:
                # Draw outline/background for readability
                cv2.putText(img, line, (x, y), font, font_scale, bg_color, thickness + 2)
                cv2.putText(img, line, (x, y), font, font_scale, color, thickness)
                y += line_height
                
        output_path = args.output_dir / frame_path.name
        cv2.imwrite(str(output_path), img)
        
    print(f"Saved frames to {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
