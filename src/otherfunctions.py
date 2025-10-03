import numpy as np

def smooth_angles(angles, smooth_window=5, jump_threshold=30):
    """Apply smoothing to angle data with handling for jumps at ±180° boundary.
    
    Args:
        angles: Array of angles in degrees
        smooth_window: Window size for smoothing
        jump_threshold: Threshold in degrees to identify jumps
        
    Returns:
        ndarray: Smoothed angles
    """
    from scipy.ndimage import gaussian_filter1d
    
    # Calculate angle differences
    angle_diff = np.abs(np.diff(angles))
    # Add zero at beginning to match length
    angle_diff = np.insert(angle_diff, 0, 0)
    
    # Mask jumps
    jumps = angle_diff > jump_threshold
    angles_masked = np.ma.array(angles, mask=jumps)
    
    # Interpolate masked values
    angles_interp = angles_masked.filled(np.nan)
    mask = np.isnan(angles_interp)
    
    # Only interpolate if we have valid data
    if np.any(~mask):
        indices = np.arange(len(angles_interp))
        valid_indices = indices[~mask]
        valid_values = angles_interp[~mask]
        angles_interp[mask] = np.interp(indices[mask], valid_indices, valid_values)
    
    # Apply smoothing
    smoothed = gaussian_filter1d(angles_interp, smooth_window/3.0)
    
    return smoothed

def normalize_angle(angle):
    """Normalize angle to -180 to 180 degrees range."""
    return (angle + 180) % 360 - 180

def circular_diff(a, b):
    """Compute minimal angular difference between two angles in degrees."""
    diff = a - b
    diff = (diff + 180) % 360 - 180
    return diff

def compute_orientation_tail_to_center(larva_data, frame=None):
    """Calculate larva orientation angle from tail to center.
    
    Args:
        larva_data: Dict containing larva tracking data (trx file)
        frame: Optional frame index to get orientation for a specific frame
        
    Returns:
        ndarray: Orientation angles in degrees (-180 to 180)
    """
    # Extract coordinates
    x_center = np.array(larva_data['x_center']).flatten()
    y_center = np.array(larva_data['y_center']).flatten()
    
    # Handle different spine data shapes
    x_spine = np.array(larva_data['x_spine'])
    y_spine = np.array(larva_data['y_spine'])
    
    if x_spine.ndim > 1:  # 2D array
        x_tail = x_spine[-1].flatten()
        y_tail = y_spine[-1].flatten()
    else:  # 1D array
        x_tail = x_spine
        y_tail = y_spine
    
    # Calculate direction vector from tail to center
    dx = x_center - x_tail
    dy = y_center - y_tail
    
    # Convert to angle in degrees (arctan2 returns angles in radians)
    # -dx to make 0 degrees = wind direction (negative x-axis)
    angles = np.degrees(np.arctan2(dy, -dx))
    
    # If frame is specified, return only that frame's angle
    if frame is not None:
        if 0 <= frame < len(angles):
            return angles[frame]
        else:
            return None
    
    return angles

def compute_orientation_tail_to_neck(x_tail, y_tail, x_neck, y_neck):
    """
    Compute orientation angle between tail-to-neck vector and negative x-axis.
    Returns angle in degrees, where 0° = facing -x (downstream), ±180° = +x (upstream).
    """
    v_x = x_neck - x_tail
    v_y = y_neck - y_tail
    angle_rad = np.arctan2(v_y, -v_x)  # -v_x for -x axis
    angle_deg = np.degrees(angle_rad)
    angle_deg = (angle_deg + 180) % 360 - 180
    return angle_deg

def determine_cast_direction(init_angle, cast_angle):
    """Determine if cast is upstream or downstream based on orientation.
    
    Args:
        init_angle: Initial orientation angle before cast (degrees)
        cast_angle: Angle during/after cast (degrees)
        
    Returns:
        str: 'upstream' or 'downstream'
    """
    # Normalize angles to -180 to 180
    init_angle = normalize_angle(init_angle)
    cast_angle = normalize_angle(cast_angle)
    
    # Calculate angle difference
    angle_diff = circular_diff(cast_angle, init_angle)
    
    # Calculate orientation relative to wind direction (0° = upwind)
    abs_orientation = abs(init_angle)
    
    # Determine direction based on orientation quadrant and turn direction
    if abs_orientation < 90:  # Facing partially upwind
        return 'upstream' if abs(angle_diff) < 90 else 'downstream'
    else:  # Facing partially downwind
        return 'upstream' if abs(angle_diff) > 90 else 'downstream'
    
    