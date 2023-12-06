# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:36:39 2023

@author: Daniel.Nugroho

Usage:
    -m r -i INPUTFILE.LAZ -o OUTPUTFILE.LAZ

v0.1-231206 - improvements
    features:
        - correct transform from RPG to MGA and vice versa using PDAL
        - Using command line arguments instead hardcoded values

    todo:
        - enable transform for raster & line vectors

"""

import os, sys, time
import numpy as np
import pdal
import argparse

RPG_TO_MGA = 0
MGA_TO_RPG = 1

def create_affine_matrix_3d(translation, scaling_factor, scaling_center, rotation_angles, rotation_center, mode):
    # Translation matrix
    tx, ty, tz = translation
    translation_matrix = np.array([[1, 0, 0, tx],
                                   [0, 1, 0, ty],
                                   [0, 0, 1, tz],
                                   [0, 0, 0, 1]])
    # Scaling matrix
    sx, sy, sz = scaling_factor
    cx, cy, cz = scaling_center
    scaling_matrix = np.array([[sx, 0, 0, cx * (1 - sx)],
                               [0, sy, 0, cy * (1 - sy)],
                               [0, 0, sz, cz * (1 - sz)],
                               [0, 0, 0, 1]])

    # Translation matrix to move pivot to origin
    T_to_origin = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 1, -cz],
        [0, 0, 0, 1]
    ])

    # Scaling matrix
    S = np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

    # Translation matrix to move back from origin
    T_back = np.array([
        [1, 0, 0, cx],
        [0, 1, 0, cy],
        [0, 0, 1, cz],
        [0, 0, 0, 1]
    ])

    # Combined transformation matrix
    scaling_matrix = T_back @ S @ T_to_origin

    # Rotation matrices for X, Y, Z
    rx, ry, rz = map(np.deg2rad, rotation_angles)  # Convert angles to radians
    cos_rx, sin_rx = np.cos(rx), np.sin(rx)
    cos_ry, sin_ry = np.cos(ry), np.sin(ry)
    cos_rz, sin_rz = np.cos(rz), np.sin(rz)

    rotation_x_matrix = np.array([[1, 0, 0, 0],
                                  [0, cos_rx, -sin_rx, 0],
                                  [0, sin_rx, cos_rx, 0],
                                  [0, 0, 0, 1]])

    rotation_y_matrix = np.array([[cos_ry, 0, sin_ry, 0],
                                  [0, 1, 0, 0],
                                  [-sin_ry, 0, cos_ry, 0],
                                  [0, 0, 0, 1]])

    rotation_z_matrix = np.array([[cos_rz, -sin_rz, 0, 0],
                                  [sin_rz, cos_rz, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

    # Combined rotation matrix
    combined_rotation_matrix = rotation_x_matrix @ rotation_y_matrix @ rotation_z_matrix

    # Adjust for rotation center
    cx, cy, cz = rotation_center
    rotation_center_matrix = np.array([[1, 0, 0, -cx],
                                       [0, 1, 0, -cy],
                                       [0, 0, 1, -cz],
                                       [0, 0, 0, 1]])
    rotation_center_inverse_matrix = np.array([[1, 0, 0, cx],
                                               [0, 1, 0, cy],
                                               [0, 0, 1, cz],
                                               [0, 0, 0, 1]])

    # Apply rotation around the center
    centered_rotation_matrix = rotation_center_inverse_matrix @ combined_rotation_matrix @ rotation_center_matrix

    # Combined transformation matrix
    
    if mode == RPG_TO_MGA:
        combined_matrix = translation_matrix @ scaling_matrix @ centered_rotation_matrix
    else:
        combined_matrix = centered_rotation_matrix @ scaling_matrix @ translation_matrix   
    

    return combined_matrix


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", required=True,
	help="transformation mode")
ap.add_argument("-i", "--input", required=True,
	help="input point cloud filename")
ap.add_argument("-o", "--output", required=True,
	help="output point cloud filename")

args = vars(ap.parse_args())

modestr = args["mode"]
inputpath = args["input"]
outputpath = args["output"]


# ensure mode value is valid
if modestr == "r":
    print("RPG to MGA transformation selected.")
    mode = RPG_TO_MGA
elif modestr == "m":
    print("MGA to RPG transformation selected.")
    mode = MGA_TO_RPG  
else:
    print("Bad mode switch, exiting...")
    sys.exit(0)

#ensure the input file exist
if not os.path.isfile(inputpath):
    print("Input file " + inputpath + " does not exist.")
    sys.exit(0)


# initialize parameters according to transformation mode

if mode == RPG_TO_MGA:
    itx = 139.545
    ity = 144.887
    itz = 0.0
    isx = 0.999662294
    isy = 0.999662294
    isz = 1.0
    icx =  412531.220
    icy = 6322482.837
    icz = 0.0   
    rot = 0.002778094
else:
    itx = -139.545
    ity = -144.887
    itz = 0.0
    isx = 1.00033782
    isy = 1.00033782
    isz = 1.0
    icx =  412531.220
    icy = 6322482.837
    icz = 0.0
    rot = -0.002778094

input_file = inputpath
output_file = outputpath

translation = (itx, ity, itz)  # Translation in X, Y, Z
scaling_factor = (isx, isy, isz)  # Uniform scaling factor
scaling_center = (icx, icy, icz)  # Center of scaling
rotation_angles = (0, 0, 0)  # Rotation angles in degrees for X, Y, Z axes
rotation_center = (icx, icy, icz)  # Center of rotation

affine_matrix_3d = create_affine_matrix_3d(translation, scaling_factor, scaling_center, rotation_angles, rotation_center, mode)

# Convert the NumPy array to a Python list for JSON
row_major_single_row = affine_matrix_3d.flatten()
single_row_list = row_major_single_row.tolist()

# construct JSON for the pipeline

thelist = ""

for i in range(len(single_row_list)):
    thelist += str(single_row_list[i]) + " "

json1 = "[" + "\"" + input_file + "\",{\"type\":\"filters.transformation\"," + \
        "\"matrix\":\"" + thelist +"\"" + \
        "},\"" + output_file + "\"" + "]"

print("Input file : " + str(input_file))
print("Output file: " + str(output_file))

start_time = time.time()

# Execute the pipeline using PDAL
pipeline = pdal.Pipeline(json1)
count = pipeline.execute()
arrays = pipeline.arrays
metadata = pipeline.metadata
log = pipeline.log

end_time = time.time()

print("Point cloud transformation completed.")
print("Elapsed time : " + str(end_time - start_time) + " seconds")  # time in seconds
