import re
import numpy as np
from scipy.spatial.transform import Rotation

def main():
    poses_file = 'outputs/estimated_poses.txt'
    srt_file = 'outputs/estimated_trajectory.srt'
    output_file = 'outputs/final_trajectory_with_targets.srt'

    print("Extracting 3D camera viewing rays from visual math...")
    
    # 1. Read the visual poses to get the camera's 3D line-of-sight
    ray_vectors = {}
    with open(poses_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = line.strip().split()
            image_name = parts[0]
            
            try:
                frame_idx = int(re.search(r'\d+', image_name).group())
            except AttributeError:
                continue

            # Convert quaternion into a World-to-Camera rotation matrix
            qw, qx, qy, qz = map(float, parts[1:5])
            r = Rotation.from_quat([qx, qy, qz, qw])
            R_matrix = r.as_matrix() 

            # In COLMAP, the camera looks down the +Z axis [0, 0, 1]
            # Multiply by Camera-to-World (R_matrix.T) to get the real-world arrow
            view_vector = R_matrix.T @ np.array([0, 0, 1])
            ray_vectors[frame_idx] = view_vector

    print("Projecting rays onto the ground to find target coordinates...")
    
    # 2. Parse the SRT and inject the target coordinates
    with open(srt_file, 'r') as f:
        srt_content = f.read()

    srt_blocks = re.split(r'\n\n+', srt_content.strip())
    final_srt = ""
    meters_to_degrees = 1 / 111320.0

    for block in srt_blocks:
        if not block.strip(): continue

        try:
            frame_idx = int(re.search(r'FrameCnt:\s*(\d+)', block).group(1))
            lat = float(re.search(r'latitude:\s*([-0-9.]+)', block).group(1))
            lon = float(re.search(r'longitude:\s*([-0-9.]+)', block).group(1))
            alt = float(re.search(r'rel_alt:\s*([-0-9.]+)', block).group(1))
        except AttributeError:
            final_srt += block + "\n\n"
            continue

        target_lat, target_lon = lat, lon # Default to directly below the drone

        # If we have a visual line-of-sight for this frame, Ray Trace it!
        if frame_idx in ray_vectors:
            Vx, Vy, Vz = ray_vectors[frame_idx]

            # Prevent dividing by zero if the drone accidentally looks straight up at the sky
            if abs(Vz) > 0.05: 
                # How many meters horizontal for every 1 meter vertical?
                horizontal_multiplier = alt / abs(Vz)

                # Calculate the X (East) and Y (North) offset in meters
                delta_x_meters = Vx * horizontal_multiplier
                delta_y_meters = Vy * horizontal_multiplier

                # Convert the meter offset into GPS degrees
                target_lon = lon + (delta_x_meters * meters_to_degrees)
                target_lat = lat + (delta_y_meters * meters_to_degrees)

        # Inject the new target coordinates into the SRT block
        new_block = block.replace("</font>", f" [target_lat: {target_lat:.6f}] [target_lon: {target_lon:.6f}]</font>")
        final_srt += new_block + "\n\n"

    with open(output_file, 'w') as f:
        f.write(final_srt)

    print(f"Success! Target coordinates saved to {output_file}")

if __name__ == '__main__':
    main()