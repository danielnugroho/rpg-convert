# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:36:39 2023

@author: Daniel.Nugroho

Usage:
    -m [MODE] -i [INPUTFILE] -o [OUTPUTFILE]
    
    MODE       : r for RPG to MGA, m for MGA to RPG
    INPUTFILE  : input file in TIF or LAZ format
    OUTPUTFILE : output file in TIF or LAZ format
    
Example usage    
    -m m -i INPUTFILE_MGA94Z50.TIF -o OUTPUTFILE_RPG.TIF
    -m r -i INPUTFILE_RPG.LAZ -o OUTPUTFILE_MGA94Z50.LAZ

v0.11-231223
    code cleanup, ability to export BIGTIFF
    documentation update

v0.10-231216
    worked well by using simpler method of translation, rotation, & scaling.

v0.9-231212
    code cleanup, but the inconsistencies in Y still persists.
    need to check what happened by doing test on various locations and note the
    XY deviations

v0.6-231208
    this version works but with varying inconsistensies in Y coordinates away 
    from center of rotation/translation/scaling

v0.1-231206 - improvements
    features:
        - correct transform from RPG to MGA and vice versa using PDAL
        - Using command line arguments instead hardcoded values

    todo:
        - enable transform for raster & line vectors

"""

import os, sys, time
import math
import argparse
import pdal
import rasterio
import numpy as np

RPG_TO_MGA = 0
MGA_TO_RPG = 1

# Define a function for applying all transformations
def transform_corner(corner, pivot, isx, itx, ity, mode):
    """
    Transform the bounding box using affine transformation.

    """

    if mode == "MGA_TO_RPG":
        # Translate first
        corner = translate_point(corner, itx, ity)
        # Rotate and scale
        corner = rotate_point(corner, pivot, -rot)
        corner = scale_point(corner, pivot, isx)

    else:
        # Scale & rotate first
        corner = scale_point(corner, pivot, isx)
        corner = rotate_point(corner, pivot, -rot)
        # Translate
        corner = translate_point(corner, itx, ity)

    return corner

def rotate_point(point, pivot, angle_deg):
    """
    Rotate a point counterclockwise by a given angle around a given pivot.

    The angle should be given in degrees.
    """
    angle_rad = math.radians(angle_deg)

    # Translate point to origin
    translated_point = (point[0] - pivot[0], point[1] - pivot[1])

    # Rotate point
    rotated_point = (
        translated_point[0] * math.cos(angle_rad) - translated_point[1] * math.sin(angle_rad),
        translated_point[0] * math.sin(angle_rad) + translated_point[1] * math.cos(angle_rad))

    # Translate point back
    new_point = (rotated_point[0] + pivot[0], rotated_point[1] + pivot[1])

    return new_point


def scale_point(point, pivot, scale_factor):
    """
    Scale a point by a given scale factor around a given pivot point.

    :param point: Tuple (x, y) representing the point to be scaled.
    :param pivot: Tuple (x, y) representing the pivot point for scaling.
    :param scale_factor: Float representing the scaling factor.
    :return: Tuple (x, y) representing the scaled point.
    """
    # Translate point to origin (pivot)
    translated_point = (point[0] - pivot[0], point[1] - pivot[1])

    # Scale point
    scaled_point = (translated_point[0] * scale_factor, translated_point[1] * scale_factor)

    # Translate point back
    new_point = (scaled_point[0] + pivot[0], scaled_point[1] + pivot[1])

    return new_point

def translate_point(point, x_translation, y_translation):
    """
    Translate a 2D point by given distances along the X and Y axes.

    :param point: Tuple (x, y) representing the original point.
    :param x_translation: The distance to translate along the X axis.
    :param y_translation: The distance to translate along the Y axis.
    :return: Tuple (x, y) representing the translated point.
    """
    # Translate the point
    new_x = point[0] + x_translation
    new_y = point[1] + y_translation

    return new_x, new_y


def get_file_value(filename):
    # Split the filename into its base and extension
    _ , extension = filename.split('.', 1) if '.' in filename else (filename, '')

    # Convert the extension to uppercase for case-insensitive comparison
    extension = extension.upper()

    if extension == "LAZ":
        return 1
    elif extension == "TIF":
        return 2
    elif extension == "DXF":
        return 3
    else:
        # Handle other extensions or cases here, if needed
        # Return a default value, or raise an exception
        return 0  # Default value (you can change this to suit your needs)

def transform_2d_to_3d_matrix(affine_2d):
    """
    Convert a 2D affine transformation matrix (3x3) to a 3D affine transformation matrix (4x4).

    :param affine_2d: A 3x3 2D affine transformation matrix.
    :return: A 4x4 3D affine transformation matrix.
    """
    # Initialize a 4x4 identity matrix
    affine_3d = np.eye(4)

    # Copy the 2D transformation to the top-left 2x2 sub-matrix
    affine_3d[:2, :2] = affine_2d[:2, :2]

    # Copy the translation components
    affine_3d[:2, 3] = affine_2d[:2, 2]

    return affine_3d


def create_affine_matrix_3d(
        translation, scaling_factor, scaling_center, rotation_angles, rotation_center, mode):

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
    centered_rotation_matrix = \
        rotation_center_inverse_matrix @ combined_rotation_matrix @ rotation_center_matrix

    # Combined transformation matrix

    if mode == RPG_TO_MGA:
        combined_matrix = translation_matrix @ scaling_matrix @ centered_rotation_matrix
    else:
        combined_matrix = centered_rotation_matrix @ scaling_matrix @ translation_matrix


    return combined_matrix


def create_affine_matrix_2d(
        translation, scaling_factor, scaling_center, rotation_angle, rotation_center, mode):

    # Translation matrix
    tx, ty = translation
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    # Scaling matrix
    sx, sy = scaling_factor
    cx, cy = scaling_center
    scaling_matrix = np.array([[sx, 0, cx * (1 - sx)],
                               [0, sy, cy * (1 - sy)],
                               [0, 0, 1]])

    # Rotation matrix
    angle_rad = np.deg2rad(rotation_angle)  # Convert angle to radians
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    rotation_matrix = np.array([[cos_a, -sin_a, 0],
                                [sin_a, cos_a, 0],
                                [0, 0, 1]])

    # Adjust for rotation center
    cx, cy = rotation_center
    rotation_center_matrix = np.array([[1, 0, -cx],
                                       [0, 1, -cy],
                                       [0, 0, 1]])
    rotation_center_inverse_matrix = np.array([[1, 0, cx],
                                               [0, 1, cy],
                                               [0, 0, 1]])

    # Apply rotation around the center
    centered_rotation_matrix = \
        rotation_center_inverse_matrix @ rotation_matrix @ rotation_center_matrix

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
elif mode == MGA_TO_RPG:
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
else:
    # for testing purposes
    itx = 0.0
    ity = 0.0
    itz = 0.0
    isx = 1.0
    isy = 1.0
    isz = 1.0
    icx =  412531.220
    icy = 6322482.837
    icz = 0.0
    rot = 25


input_file = inputpath
output_file = outputpath

"""
# THIS SECTION OF CODE IS FOR PURE LAZ TRANSFORMATION in 3D, INSTEAD OF 2D

translation = (itx, ity, itz)  # Translation in X, Y, Z
scaling_factor = (isx, isy, isz)  # Uniform scaling factor
scaling_center = (icx, icy, icz)  # Center of scaling
rotation_angles = (0, 0, 0)  # Rotation angles in degrees for X, Y, Z axes
rotation_center = (icx, icy, icz)  # Center of rotation

# create real affine 3D transformation matrix
affine_matrix_3d = create_affine_matrix_3d(
    translation, scaling_factor, scaling_center, rotation_angles, rotation_center, mode)

"""

translation = (itx, ity)  # Translation in X, Y
scaling_factor = (isx, isy)  # Uniform scaling factor
scaling_center = (icx, icy)  # Center of scaling
rotation_angles = rot  # Rotation angles in degrees for X, Y axes
rotation_center = (icx, icy)  # Center of rotation

affine_matrix_2d = create_affine_matrix_2d(
    translation, scaling_factor, scaling_center, rotation_angles, rotation_center, mode)

# create a faux 3D affine transformation matrix
affine_matrix_3d = transform_2d_to_3d_matrix(affine_matrix_2d)


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

filetype = get_file_value(input_file)
start_time = time.time()
    
if filetype == 1:
    print("Type is point cloud.")

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


elif filetype == 2:
    print("Type is raster.")

    # Open the source raster dataset
    with rasterio.open(input_file) as src:

        # Extract bounding box corners
        bbox_corners = [(src.bounds.left, src.bounds.top),
                        (src.bounds.right, src.bounds.top),
                        (src.bounds.left, src.bounds.bottom),
                        (src.bounds.right, src.bounds.bottom)]

        pivot = (icx, icy)

        # Apply transformations to all corners
        transformed_corners = \
            [transform_corner(corner, pivot, isx, itx, ity, mode) for corner in bbox_corners]

        # Extract east, west, north, south from bounding box
        trans_east, trans_west, trans_north, trans_south = \
            transformed_corners[1][0], transformed_corners[0][0], \
            transformed_corners[0][1], transformed_corners[2][1]

        # Open the source raster dataset
        with rasterio.open(input_file) as src:
            src_transform = src.transform

            # get the affine transformation from the transformed bounds

            dst_transform = rasterio.transform.from_bounds(
                trans_west, trans_south, trans_east, trans_north, src.width, src.height)


            data = src.read()
            kwargs = src.meta.copy()

            kwargs.update({
                'transform': dst_transform,
                'compress': 'lzw'
            })

            # Create a new output raster dataset
            with rasterio.open(output_file, 'w', BIGTIFF='YES', **kwargs) as dst:

                # Reproject and resample the source raster to the output raster
                # Actually there is no reprojection was done, it's just changing the extents

                for i, band in enumerate(data, 1):

                    dest = np.zeros((src.height, src.width))

                    dest = band

                    dst.write(dest, indexes=i)

    end_time = time.time()
        
    print("Raster data transformation completed.")
    print("Elapsed time : " + str(end_time - start_time) + " seconds")  # time in seconds
        
elif filetype == 3:
    print("Type is vector.")
    print("Vector data transformation is not yet supported.")
else:
    print("Type is unknown.")

