import cv2
import re
from pathlib import Path
from hloc import extract_features, match_features, localize_sfm, pairs_from_retrieval

def extract_frames_from_video(video_path, output_dir, frame_skip=30):
    """Extracts frames from the test video to be processed by hloc."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    
    frame_count = 0
    saved_count = 0
    frame_names = []
    
    print(f"Extracting frames from {video_path.name}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % frame_skip == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            frame_names.append(frame_name)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    print(f"Extracted {saved_count} frames for localization.")
    return frame_names

def generate_srt_and_kml(poses_file, output_srt, output_kml, test_srt_path, reference_lat, reference_lon, frame_names):
    """
    Combines visual tracking with telemetry, and generates a KML map comparing the 
    Visual Odometry trajectory against the original Drone GPS trajectory.
    """
    print("Fusing Visual Odometry with provided telemetry...")
    
    # 1. Read the successful visual poses into a dictionary
    visual_poses = {}
    with open(poses_file, 'r') as f:
        pose_lines = [line for line in f.readlines() if not line.startswith('#') and line.strip()]
        for line in pose_lines:
            parts = line.strip().split()
            try:
                frame_idx = int(re.search(r'\d+', parts[0]).group())
                tx, ty, tz = map(float, parts[5:8])
                visual_poses[frame_idx] = (tx, ty)
            except AttributeError:
                continue

    # 2. Read the raw DJI telemetry
    with open(test_srt_path, 'r') as f:
        test_srt_content = f.read()
    
    srt_blocks = re.split(r'\n\n+', test_srt_content.strip())
    srt_content_final = ""
    meters_to_degrees = 1 / 111320.0 
    
    last_known_tx, last_known_ty = 0.0, 0.0
    
    # Lists to store coordinates for the KML map
    estimated_coords = []
    original_coords = []
    
    # 3. Loop through EVERY single second of the video
    for frame_idx in range(len(frame_names)):
        target_block_idx = frame_idx * 30 
        
        if target_block_idx >= len(srt_blocks): 
            break 
            
        if frame_idx in visual_poses:
            last_known_tx, last_known_ty = visual_poses[frame_idx]
            
        est_lat = reference_lat + (last_known_ty * meters_to_degrees)
        est_lon = reference_lon + (last_known_tx * meters_to_degrees)
        
        current_block = srt_blocks[target_block_idx]
        
        try:
            rel_alt_match = re.search(r'(?:rel_alt|osd_alt)\s*:\s*([-0-9.]+)', current_block, re.IGNORECASE)
            pitch_match = re.search(r'(?:gimbal_pitch|camera_pitch|osd_pitch|osd_pt)\s*:\s*([-0-9.]+)', current_block, re.IGNORECASE)
            roll_match = re.search(r'(?:gimbal_roll|camera_roll|osd_roll|osd_r)\s*:\s*([-0-9.]+)', current_block, re.IGNORECASE)
            yaw_match = re.search(r'(?:gimbal_heading|gimbal_yaw|camera_yaw|osd_yaw|osd_head)\s*:\s*([-0-9.]+)', current_block, re.IGNORECASE)
            orig_lat_match = re.search(r'latitude:\s*([-0-9.]+)', current_block)
            orig_lon_match = re.search(r'longitude:\s*([-0-9.]+)', current_block)
            
            rel_alt = float(rel_alt_match.group(1)) if rel_alt_match else 0.0
            pitch = float(pitch_match.group(1)) if pitch_match else 0.0
            roll = float(roll_match.group(1)) if roll_match else 0.0
            yaw = float(yaw_match.group(1)) if yaw_match else 0.0
            
            orig_lat = float(orig_lat_match.group(1)) if orig_lat_match else reference_lat
            orig_lon = float(orig_lon_match.group(1)) if orig_lon_match else reference_lon
            
            # Save coordinates for KML (Format required by KML: Longitude,Latitude,Altitude)
            estimated_coords.append(f"{est_lon:.6f},{est_lat:.6f},{rel_alt:.3f}")
            original_coords.append(f"{orig_lon:.6f},{orig_lat:.6f},{rel_alt:.3f}")
            
            # Construct the New Block for SRT
            lines = current_block.split('\n')
            new_block = []
            for line in lines:
                if "latitude" in line.lower():
                    new_line = (f"[latitude: {est_lat:.6f}] [longitude: {est_lon:.6f}] "
                                f"[rel_alt: {rel_alt:.3f} abs_alt: 0.000] "
                                f"[camera_pitch: {pitch:.2f}] [camera_roll: {roll:.2f}] [camera_yaw: {yaw:.2f}]</font>")
                    new_block.append(new_line)
                else:
                    new_block.append(line)
                    
            srt_content_final += "\n".join(new_block) + "\n\n"
            
        except Exception as e:
            print(f"Failed to build block {frame_idx}. Error: {e}")
            continue

    # Write the SRT file
    with open(output_srt, 'w') as f:
        f.write(srt_content_final)
    print(f"Successfully generated fused trajectory to {output_srt}")

    # Write the KML file
    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Flight Trajectory Comparison</name>
    <Style id="redLine">
      <LineStyle><color>ff0000ff</color><width>4</width></LineStyle>
    </Style>
    <Style id="blueLine">
      <LineStyle><color>ffff0000</color><width>4</width></LineStyle>
    </Style>
    <Placemark>
      <name>Estimated Trajectory (Visual Odometry)</name>
      <description>The path calculated purely from camera images.</description>
      <styleUrl>#redLine</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>{" ".join(estimated_coords)}</coordinates>
      </LineString>
    </Placemark>
    <Placemark>
      <name>Original Trajectory (Drone GPS)</name>
      <description>The ground truth recorded by the drone's hardware.</description>
      <styleUrl>#blueLine</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>{" ".join(original_coords)}</coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""
    with open(output_kml, 'w') as f:
        f.write(kml_content)
    print(f"Successfully generated comparison map to {output_kml}")

def main():
    # --- Configuration Paths ---
    test_video = Path('data/testing/v12.mp4')
    test_srt_file = Path('data/testing/v12.srt') 
    
    query_dir = Path('outputs/test_frames')
    sfm_dir = Path('outputs/sfm_map') 
    
    # Output files
    features_path = Path('outputs/features_query.h5')
    matches_path = Path('outputs/matches_query.h5')
    poses_path = Path('outputs/estimated_poses.txt')
    final_srt = Path('outputs/estimated_trajectory.srt')
    final_kml = Path('outputs/trajectory_comparison.kml') # <-- New KML Output
    
    # 1. Extract Frames
    frame_names = extract_frames_from_video(test_video, query_dir)
    
    # 2. Setup hloc configurations
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['NN-superpoint'] 
    retrieval_conf = extract_features.confs['netvlad']
    
    # 3. Extract features
    print("Extracting features from test frames...")
    extract_features.main(feature_conf, query_dir, image_list=frame_names, feature_path=features_path)
    
    print("Extracting NetVLAD features for the 3D map...")
    map_dir = Path('data/training/extracted_frames')
    map_global_feats = Path('outputs/map_global_feats.h5')
    if not map_global_feats.exists():
        extract_features.main(retrieval_conf, map_dir, feature_path=map_global_feats)
        
    print("Extracting NetVLAD features for the test frames...")
    query_global_feats = Path('outputs/query_global_feats.h5')
    extract_features.main(retrieval_conf, query_dir, image_list=frame_names, feature_path=query_global_feats)
    
    # 4. Retrieval and Pairs
    print("Pairing test frames with the 3D map...")
    pairs_path = Path('outputs/query_pairs.txt')
    pairs_from_retrieval.main(
        descriptors=query_global_feats, 
        output=pairs_path, 
        num_matched=5, 
        db_descriptors=map_global_feats 
    )
    
    print("Filtering out dropped map frames...")
    import pycolmap
    model = pycolmap.Reconstruction(sfm_dir)
    registered_images = {img.name for img in model.images.values()}
    
    with open(pairs_path, 'r') as f:
        pairs_lines = f.readlines()
        
    valid_pairs = []
    for line in pairs_lines:
        test_img, train_img = line.strip().split()
        if train_img in registered_images:
            valid_pairs.append(line)
            
    with open(pairs_path, 'w') as f:
        f.writelines(valid_pairs)
    print(f"Filtered valid pairs: {len(valid_pairs)} / {len(pairs_lines)}")

    # 5. Match the SuperPoint features
    print("Matching exact points...")
    map_features_path = Path('outputs/feats-superpoint-n4096-r1024.h5')
    match_features.main(
        matcher_conf, 
        pairs_path, 
        features=features_path, 
        features_ref=map_features_path, 
        matches=matches_path
    )
    
    # 6. Run the Localization Engine
    print("Preparing camera intrinsics...")
    query_list_path = Path('outputs/query_list_with_intrinsics.txt')
    with open(query_list_path, 'w') as f:
        for name in frame_names:
            f.write(f"{name} SIMPLE_PINHOLE 1920 1080 900 960 540\n")
    
    print("Calculating 3D Poses...")
    localize_sfm.main(
        sfm_dir,             
        query_list_path,     
        pairs_path,          
        features_path,       
        matches_path,        
        poses_path           
    )
    
    # 7. Convert the math into the final SRT AND KML
    ref_lat, ref_lon = 32.102624, 35.209724
    if test_srt_file.exists():
        with open(test_srt_file, 'r') as f:
            content = f.read()
            lat_match = re.search(r'latitude:\s*([-0-9.]+)', content)
            lon_match = re.search(r'longitude:\s*([-0-9.]+)', content)
            if lat_match and lon_match:
                ref_lat = float(lat_match.group(1))
                ref_lon = float(lon_match.group(1))

    generate_srt_and_kml(
        poses_file=poses_path, 
        output_srt=final_srt, 
        output_kml=final_kml, 
        test_srt_path=test_srt_file, 
        reference_lat=ref_lat, 
        reference_lon=ref_lon,
        frame_names=frame_names 
    )

if __name__ == '__main__':
    main()