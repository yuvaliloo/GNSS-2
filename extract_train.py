import cv2
from pathlib import Path

# 1. Setup paths
video_path = Path('data/training/v11.mp4')
output_dir = Path('data/training/extracted_frames')
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Extract frames (1 frame per second to keep the map size manageable)
cap = cv2.VideoCapture(str(video_path))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = 0
saved_count = 0

print(f"Chopping {video_path.name} into images...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    if frame_count % fps == 0: # Save 1 frame per second
        cv2.imwrite(str(output_dir / f"train_frame_{saved_count:05d}.jpg"), frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"Done! Saved {saved_count} images to {output_dir}")