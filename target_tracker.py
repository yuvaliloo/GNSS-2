import re
import math

def parse_srt_blocks(srt_path, frame_prefix):
    """Reads an SRT file and maps the telemetry to its corresponding video frame."""
    with open(srt_path, 'r') as f:
        content = f.read()
        
    blocks = re.split(r'\n\n+', content.strip())
    data = {}
    
    for idx, block in enumerate(blocks):
        if not block.strip(): continue
        

        if idx % 30 != 0:
            continue
            
        saved_count = idx // 30
        # ----------------------------------------------
        
        # Use robust regex for DJI Mini 3 Pro
        lat_match = re.search(r'latitude:\s*([-0-9.]+)', block)
        lon_match = re.search(r'longitude:\s*([-0-9.]+)', block)
        yaw_match = re.search(r'(?:camera_yaw|gimbal_yaw|osd_yaw|osd_head)\s*:\s*([-0-9.]+)', block, re.IGNORECASE)
        pitch_match = re.search(r'(?:camera_pitch|gimbal_pitch|osd_pitch|osd_pt)\s*:\s*([-0-9.]+)', block, re.IGNORECASE)
        
        if lat_match and lon_match:
            frame_name = f"{frame_prefix}_{saved_count:05d}.jpg"
            data[frame_name] = {
                'lat': float(lat_match.group(1)),
                'lon': float(lon_match.group(1)),
                'yaw': float(yaw_match.group(1)) if yaw_match else 0.0,
                'pitch': float(pitch_match.group(1)) if pitch_match else -60.0
            }
    return data

def main():
    print("Starting GNSS-Denied Tracking using Visual Image Retrieval...")
    
    # 1. File Paths
    train_srt_path = 'data/training/v11.srt' # The Database
    test_srt_path = 'data/testing/v12.srt'   # The Live Flight
    pairs_path = 'outputs/query_pairs.txt'   # The AI Visual Matches
    
    output_kml = 'outputs/FINAL_ASSIGNMENT_MAP.kml'
    
    # 2. Load Telemetry Databases
    print("Loading preprocessing database (v11)...")
    train_db = parse_srt_blocks(train_srt_path, "train_frame")
    
    print("Loading live flight IMU telemetry (v12)...")
    test_db = parse_srt_blocks(test_srt_path, "frame")
    
    # 3. Link the visual matches from NetVLAD
    print("Linking visual matches to geographic coordinates...")
    best_matches = {}
    with open(pairs_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                q_img, db_img = parts
                # NetVLAD ranks the best match first, so we only save the first occurrence
                if q_img not in best_matches:
                    best_matches[q_img] = db_img
                    
    # 4. Calculate Trajectories
    original_coords = []
    estimated_coords = []
    target_coords = []
    meters_to_degrees = 1 / 111320.0
    
    for q_img, test_data in test_db.items():
        # A. Save the original GPS just for our comparison map
        orig_lat, orig_lon = test_data['lat'], test_data['lon']
        original_coords.append(f"{orig_lon:.6f},{orig_lat:.6f},0")
        
        # B. GNSS-DENIED ESTIMATION: Where are we based purely on the AI match?
        matched_db_img = best_matches.get(q_img)
        
        if matched_db_img and matched_db_img in train_db:
            # We found our location in the preprocessing data!
            est_lat = train_db[matched_db_img]['lat']
            est_lon = train_db[matched_db_img]['lon']
        else:
            # If the camera was blurry and matching failed, we skip this frame
            continue
            
        estimated_coords.append(f"{est_lon:.6f},{est_lat:.6f},0")
        
        # C. TARGET RAY TRACING (Using IMU, NOT GPS)
        alt = 119.0
        pitch = test_data['pitch']
        yaw = test_data['yaw']
        
        # Prevent infinity errors
        safe_pitch = abs(pitch)
        if safe_pitch < 5.0: 
            safe_pitch = 5.0
            
        # Trigonometry based on altitude and pitch
        angle_from_nadir = 90.0 - safe_pitch
        distance_forward = alt * math.tan(math.radians(angle_from_nadir))
        
        # Project forward using compass heading
        delta_x_meters = distance_forward * math.sin(math.radians(yaw))
        delta_y_meters = distance_forward * math.cos(math.radians(yaw))
        
        target_lon = est_lon + (delta_x_meters * meters_to_degrees)
        target_lat = est_lat + (delta_y_meters * meters_to_degrees)
        
        target_coords.append(f"{target_lon:.6f},{target_lat:.6f},0")

    # 5. Build the Master KML File
    print("Writing Master KML File...")
    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Optical Target Tracking (Visual Retrieval)</name>
    <Style id="blueLine"><LineStyle><color>ffff0000</color><width>4</width></LineStyle></Style>
    <Style id="redLine"><LineStyle><color>ff0000ff</color><width>4</width></LineStyle></Style>
    <Style id="yellowLine"><LineStyle><color>ff00ffff</color><width>4</width></LineStyle></Style>
    
    <Placemark>
      <name>1. Original Drone GPS (Ground Truth)</name>
      <styleUrl>#blueLine</styleUrl>
      <LineString><tessellate>1</tessellate><coordinates>{" ".join(original_coords)}</coordinates></LineString>
    </Placemark>
    
    <Placemark>
      <name>2. Estimated Drone Position (GNSS-Denied)</name>
      <styleUrl>#redLine</styleUrl>
      <LineString><tessellate>1</tessellate><coordinates>{" ".join(estimated_coords)}</coordinates></LineString>
    </Placemark>
    
    <Placemark>
      <name>3. Camera Target Path</name>
      <styleUrl>#yellowLine</styleUrl>
      <LineString><tessellate>1</tessellate><coordinates>{" ".join(target_coords)}</coordinates></LineString>
    </Placemark>
  </Document>
</kml>"""

    with open(output_kml, 'w') as f:
        f.write(kml_content)

    print(f"DONE! Open {output_kml} in Google Earth!")

if __name__ == '__main__':
    main()