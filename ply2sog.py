# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:04:54 2026

@author: Sleepyehead
"""

import numpy as np
from plyfile import PlyData, PlyElement
import os
import subprocess
import argparse

# ==========================================
# CONFIGURATION & VARIABLES
# Edit these default values before running
# ==========================================
CONFIG = {
    # File Paths
    "DEFAULT_INPUT_PATH": r"F:\iceland2024\finalas\try\indoor12.ply",
    
    # Point Cloud Processing
    "DENSITY_STEP": 1,           # 1 = 100% density. 2 = 50% density.
    "Y_OFFSET": 200.0,           # Adds depth so the camera starts away from the object
    
    # Gaussian Splat Properties
    "GAUSSIAN_SCALE": -3.0,      # -4.0 is a "tight" scale. -3.0 is slightly looser.
    "GAUSSIAN_OPACITY": 20.0,    # Default base opacity for splats
    
    # System & Memory
    "NODE_MAX_MEMORY": "16384",  # Megabytes allocated for Node.js SOG conversion
}
# ==========================================

def process_and_convert(input_path, config):
    """
    Reads a point cloud, adjusts coordinates for WebGL, injects Gaussian parameters,
    and converts to a .sog file using splat-transform.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at {input_path}")
        return

    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path).replace('.ply', '')
    cleaned_ply = os.path.join(directory, f"{filename}_fixed_30.ply")
    output_sog = os.path.join(directory, f"{filename}.sog")

    print(f"--- Loading {input_path} ---")
    try:
        plydata = PlyData.read(input_path)
    except Exception as e:
        print(f"❌ Error reading PLY file: {e}")
        return

    v_data = plydata['vertex']
    
    # Apply density step
    v = v_data[::config["DENSITY_STEP"]] 
    num_verts = len(v['x'])

    print(f"--- Processing {num_verts} points with High Precision ---")
    
    # 1. Read as float64 to prevent jitter on large coordinate offsets
    raw_x = np.array(v['x'], dtype=np.float64)
    raw_y = np.array(v['y'], dtype=np.float64)
    raw_z = np.array(v['z'], dtype=np.float64)
    
    # 2. Local Offsets: Center the model without losing decimal precision
    x_mean, y_mean, z_mean = raw_x.mean(), raw_y.mean(), raw_z.mean()
    
    # 3. Coordinate Fix: Swap Z-up to Y-up and FLIP vertical for WebGL
    # WebGL X = Lidar X
    # WebGL Y = Lidar -Z (Vertical Flip)
    # WebGL Z = Lidar Y
    x_final = (raw_x - x_mean).astype(np.float32)
    y_final = (-(raw_z - z_mean)).astype(np.float32) 
    z_final = (raw_y - y_mean).astype(np.float32)

    # 4. Apply camera offset
    y_final += config["Y_OFFSET"] 

    # 5. Color Normalization for Spherical Harmonics (DC band)
    f_dc_0 = (np.array(v['red'], dtype=np.float32) / 255.0 - 0.5) / 0.28209
    f_dc_1 = (np.array(v['green'], dtype=np.float32) / 255.0 - 0.5) / 0.28209
    f_dc_2 = (np.array(v['blue'], dtype=np.float32) / 255.0 - 0.5) / 0.28209

    # 6. Define full vertex properties required by Gaussian Splatting
    vertex_properties = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]

    vertex_data = np.empty(num_verts, dtype=vertex_properties)
    vertex_data['x'] = x_final
    vertex_data['y'] = y_final
    vertex_data['z'] = z_final
    vertex_data['f_dc_0'], vertex_data['f_dc_1'], vertex_data['f_dc_2'] = f_dc_0, f_dc_1, f_dc_2
    
    # 7. Apply standard Gaussian scales, rotations, and opacities
    vertex_data['scale_0'] = vertex_data['scale_1'] = vertex_data['scale_2'] = config["GAUSSIAN_SCALE"]
    vertex_data['opacity'] = np.ones(num_verts, dtype=np.float32) * config["GAUSSIAN_OPACITY"]
    
    vertex_data['rot_0'] = 1.0
    vertex_data['rot_1'] = 0.0
    vertex_data['rot_2'] = 0.0
    vertex_data['rot_3'] = 0.0

    print(f"--- Saving Intermediate PLY ---")
    el = PlyElement.describe(vertex_data, 'vertex')
    
    # Stability fix for large files: explicit byte order
    with open(cleaned_ply, 'wb') as f:
        PlyData([el], text=False, byte_order='<').write(f)

    print(f"--- Converting to SOG via Node.js ---")
    try:
        env = os.environ.copy()
        env["NODE_OPTIONS"] = f"--max-old-space-size={config['NODE_MAX_MEMORY']}"
        
        # Call the external splat-transform tool
        subprocess.run(
            ['splat-transform', '--input', cleaned_ply, '--output', output_sog], 
            check=True, 
            shell=True, 
            env=env
        )
        print(f"\n✅ SUCCESS! Coordinates are precise. SOG saved to:\n{output_sog}")
        
        # Clean up the intermediate file
        os.remove(cleaned_ply)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ SOG Conversion Failed. Ensure 'splat-transform' is installed via npm. Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PLY point clouds into SOG Gaussian Splats.")
    parser.add_argument("-i", "--input", type=str, default=CONFIG["DEFAULT_INPUT_PATH"], 
                        help="Path to the input .ply file")
    
    args = parser.parse_args()
    
    process_and_convert(args.input, CONFIG)