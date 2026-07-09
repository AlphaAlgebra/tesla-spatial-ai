mkdir -p src/utils
cat << 'EOF' > src/utils/geometry.py
import numpy as np

def project_3d_to_2d_numpy(points_3d, intrinsic_matrix, extrinsic_matrix):
    """
    Pure NumPy implementation of 3D-to-2D pinhole camera projection geometry.
    
    Args:
        points_3d: Array of shape (N, 3) representing [X, Y, Z] world coordinates.
        intrinsic_matrix: Array of shape (3, 3) representing camera internal calibration.
        extrinsic_matrix: Array of shape (4, 4) representing camera transformation matrix.
        
    Returns:
        Array of shape (N, 2) representing target [u, v] camera pixel locations.
    """
    num_points = points_3d.shape[0]
    
    # 1. Convert to homogeneous coordinates by appending vector ones [X, Y, Z, 1]
    ones = np.ones((num_points, 1))
    points_homogeneous = np.hstack([points_3d, ones]) # Shape: (N, 4)
    
    # 2. Transform world coordinate metrics directly into local camera tracking frames
    # Equation: X_cam = Extrinsic * X_world
    points_cam = points_homogeneous @ extrinsic_matrix.T # Shape: (N, 4)
    points_cam = points_cam[:, :3] # Slice off scaling dimension to keep [X_c, Y_c, Z_c]
    
    # 3. Project spatial vector depths onto the physical pixel coordinate map plane
    # Equation: x = Intrinsic * X_cam
    points_pixel_homo = points_cam @ intrinsic_matrix.T # Shape: (N, 3)
    
    # 4. Resolve lens depth scaling factor maps (Z dimensions)
    depth = points_pixel_homo[:, 2:3] + 1e-6 # Safeguard against division by zero errors
    pixels_2d = points_pixel_homo[:, :2] / depth
    
    return pixels_2d
EOF
