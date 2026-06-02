from pathlib import Path
import pycolmap
from hloc import extract_features, match_features, reconstruction

# 1. Define your paths
images_dir = Path('data/training/extracted_frames') 
outputs = Path('outputs/')
sfm_pairs = outputs / 'pairs-sequence.txt' # Renamed to avoid confusion
sfm_dir = outputs / 'sfm_map'
matches_path = outputs / 'matches.h5'

# 2. Choose the Neural Networks
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['NN-superpoint']

# --- CUSTOM PAIR GENERATOR TO REPLACE MISSING HLOC MODULE ---
def create_sequential_pairs(image_names, output_path, window_size=5):
    """Generates a text file pairing each frame with its immediate neighbors."""
    pairs = []
    for i in range(len(image_names)):
        for j in range(1, window_size + 1):
            if i + j < len(image_names):
                pairs.append(f"{image_names[i]} {image_names[i+j]}")
                
    with open(output_path, 'w') as f:
        f.write("\n".join(pairs))
    print(f"Successfully generated {len(pairs)} sequential pairs.")
# -----------------------------------------------------------

# 3. The Pipeline Execution
def build_reference_map():
    print("Extracting Features...")
    features = extract_features.main(feature_conf, images_dir, outputs)

    print("Generating Image Pairs (Sequential)...")
    # Sort the list so Frame 1 is mathematically next to Frame 2
    image_list = sorted([p.name for p in images_dir.iterdir()]) 
    
    # Call our custom function instead of the missing hloc one
    create_sequential_pairs(image_list, sfm_pairs, window_size=5)

    print("Matching Features...")
    matches = match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches_path)
    
    print("Building 3D Map...")
    reconstruction.main(sfm_dir, images_dir, sfm_pairs, features, matches)
    print(f"Map successfully built and saved to {sfm_dir}")

if __name__ == '__main__':
    build_reference_map()