"""
Utility functions for the RAIT (Robot-AI Toolkit) system.
"""

import os
import sys
import numpy as np
import pyrealsense2 as rs
import itertools

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import load_config

def load_camera_config():
    """
    Load camera configurations dynamically from the YAML file.

    Returns:
        dict: Camera transformation and intrinsic parameters.
    """
    return load_config("VisionAI/config/vision_ai_config.yaml")['Camera']['D435I']['India']

#----------------------------------------------------------------#
def get_valid_depth(depth_array, x, y):
    """
    Find the first non-zero depth value within a 10-pixel radius around the given point.

    Searches in increasing radius up to 10 pixels until a valid (non-zero) depth value
    is found. This helps handle cases where the target pixel has invalid depth data.

    Args:
        depth_array (numpy.ndarray): 2D array containing depth values.
        x (int): Target x-coordinate in the depth array.
        y (int): Target y-coordinate in the depth array.

    Returns:
        tuple: (depth, x, y) where:
            - depth (float): Valid depth value or 0 if none found.
            - x (int): X-coordinate of valid depth point.
            - y (int): Y-coordinate of valid depth point.
    """
    height, width = depth_array.shape
    if depth_array[y, x] > 0:
        return depth_array[y, x], x, y

    max_radius = 10
    directions = list(itertools.product([-1, 0, 1], repeat=2))

    for radius in range(1, max_radius + 1):
        for dx, dy in directions:
            new_x, new_y = x + dx * radius, y + dy * radius
            if 0 <= new_x < width and 0 <= new_y < height and depth_array[new_y, new_x] > 0:
                return depth_array[new_y, new_x], new_x, new_y

    return 0, x, y

def deproject_pixel_to_point(depth_array, pixel_coords, intrinsics):
    """
    Convert pixel coordinates and depth to a 3D point in camera space.

    Uses RealSense intrinsics to compute the real-world 3D position of a given pixel.

    Args:
        depth_array (numpy.ndarray): Depth image array.
        pixel_coords (tuple): (x, y) pixel coordinates.
        intrinsics (rs.intrinsics): Camera intrinsic parameters.

    Returns:
        numpy.ndarray: 3D point in camera coordinate space (x, y, z).
    """
    x, y = int(pixel_coords[0]), int(pixel_coords[1])

    if x < 0 or x >= depth_array.shape[1] or y < 0 or y >= depth_array.shape[0]:
        print(f"[VisionAI Utils] Pixel ({x}, {y}) out of bounds.")
        return np.array([0, 0, 0])

    depth, valid_x, valid_y = get_valid_depth(depth_array, x, y)

    if depth == 0:
        print(f"[VisionAI Utils] No valid depth found near pixel ({x}, {y}).")
        return np.array([0, 0, 0])

    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [valid_x, valid_y], depth)
    print(f"[VisionAI Utils] Deprojected 3D point: {point_3d}")

    return np.array(point_3d)

def transform_coordinates(x, y, z):
    """
    Transforms coordinates from camera space to the robot's base frame.

    Applies a series of transformations using calibration matrices to convert
    coordinates from the camera's reference frame to the robot's base frame.

    Args:
        x (float): X-coordinate in camera space (millimeters).
        y (float): Y-coordinate in camera space (millimeters).
        z (float): Z-coordinate in camera space (millimeters).

    Returns:
        tuple: (transformed_x, transformed_y, transformed_z) in the robot base frame (millimeters).
    """
    try:
        config = load_camera_config()
        calib_matrix_x = np.array(config['Transformations']['X'])
        calib_matrix_y = np.array(config['Transformations']['Y'])

        A = calib_matrix_y @ np.eye(4) @ np.linalg.inv(calib_matrix_x)
        transformed_x, transformed_y, transformed_z = A[:3, 3] * 1000

        print(f"[VisionAI Utils] Transformed coordinates: ({transformed_x:.2f}, {transformed_y:.2f}, {transformed_z:.2f})")
        return float(transformed_x), float(transformed_y), float(transformed_z)

    except Exception as e:
        print(f"[VisionAI Utils] Error in coordinate transformation: {e}")
        return x, y, z  # Return original coordinates if the transformation fails
