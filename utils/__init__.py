"""
Utils Package - utils/__init__.py
Initialization file for the boxing analysis utilities package
"""

from .video_utils import (
    VideoProcessor, 
    VideoWriter, 
    VideoInfo,
    create_comparison_video,
    get_video_duration,
    extract_key_frames,
    convert_video_format
)

from .visualization_utils import (
    BoxingVisualizer,
    create_3d_pose_plot,
    create_performance_dashboard,
    create_timeline_analysis,
    create_comparison_chart,
    save_visualization_report,
    generate_pose_skeleton,
    create_heatmap_analysis,
    plot_movement_trajectory
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Future Sportler AI Engineer Candidate"
__description__ = "Utility modules for boxing performance analysis"

# Package-level constants
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
DEFAULT_OUTPUT_RESOLUTION = (1280, 720)
DEFAULT_FPS = 30.0

# Color schemes for visualizations
BOXING_COLOR_SCHEME = {
    'excellent': '#2E8B57',      # Sea Green
    'good': '#32CD32',           # Lime Green  
    'average': '#FFD700',        # Gold
    'poor': '#FF8C00',           # Dark Orange
    'critical': '#DC143C',       # Crimson
    'background': '#1E1E1E',     # Dark Gray
    'text': '#FFFFFF',           # White
    'accent': '#00CED1'          # Dark Turquoise
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'excellent': 85,
    'good': 70,
    'average': 55,
    'poor': 40,
    'critical': 0
}

# Pose landmark connections for visualization
POSE_CONNECTIONS = [
    # Head and torso
    (0, 1), (1, 2), (2, 3), (3, 7),    # Face outline
    (0, 4), (4, 5), (5, 6), (6, 8),    # Face outline
    (9, 10),                            # Mouth
    (11, 12),                           # Shoulders
    (11, 23), (12, 24),                 # Shoulder to hip
    (23, 24),                           # Hip line
    
    # Arms
    (11, 13), (13, 15),                 # Left arm
    (12, 14), (14, 16),                 # Right arm
    (15, 17), (15, 19), (15, 21),       # Left hand
    (16, 18), (16, 20), (16, 22),       # Right hand
    
    # Legs  
    (23, 25), (25, 27),                 # Left leg
    (24, 26), (26, 28),                 # Right leg
    (27, 29), (27, 31),                 # Left foot
    (28, 30), (28, 32)                  # Right foot
]

# Analysis configuration
ANALYSIS_CONFIG = {
    'pose_detection': {
        'model_complexity': 2,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'enable_segmentation': True
    },
    'boxing_analysis': {
        'punch_angle_threshold': 140,        # degrees
        'ideal_stance_width': 0.3,           # normalized
        'ideal_guard_height': 0.15,          # relative to body
        'movement_smoothing_window': 5,      # frames
        'punch_speed_multiplier': 30         # fps conversion
    },
    'visualization': {
        'default_width': 1200,
        'default_height': 800,
        'marker_size': 8,
        'line_width': 3,
        'font_size': 12
    }
}

def get_color_for_score(score: float) -> str:
    """Get color based on performance score"""
    if score >= PERFORMANCE_THRESHOLDS['excellent']:
        return BOXING_COLOR_SCHEME['excellent']
    elif score >= PERFORMANCE_THRESHOLDS['good']:
        return BOXING_COLOR_SCHEME['good']
    elif score >= PERFORMANCE_THRESHOLDS['average']:
        return BOXING_COLOR_SCHEME['average']
    elif score >= PERFORMANCE_THRESHOLDS['poor']:
        return BOXING_COLOR_SCHEME['poor']
    else:
        return BOXING_COLOR_SCHEME['critical']

def get_performance_level(score: float) -> str:
    """Get performance level text based on score"""
    if score >= PERFORMANCE_THRESHOLDS['excellent']:
        return 'Excellent'
    elif score >= PERFORMANCE_THRESHOLDS['good']:
        return 'Good'
    elif score >= PERFORMANCE_THRESHOLDS['average']:
        return 'Average'
    elif score >= PERFORMANCE_THRESHOLDS['poor']:
        return 'Poor'
    else:
        return 'Critical'

def validate_analysis_data(analysis_data: list) -> bool:
    """Validate analysis data structure"""
    if not analysis_data or not isinstance(analysis_data, list):
        return False
    
    required_keys = ['frame', 'landmarks', 'stance', 'punch', 'defense', 'movement']
    
    for frame_data in analysis_data[:5]:  # Check first 5 frames
        if not isinstance(frame_data, dict):
            return False
        
        for key in required_keys:
            if key not in frame_data:
                return False
    
    return True

def setup_logging():
    """Setup logging configuration for the package"""
    import logging
    
    # Create logger
    logger = logging.getLogger('boxing_analysis')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger

# Initialize package logger
logger = setup_logging()

# Export commonly used items
__all__ = [
    # Classes
    'VideoProcessor',
    'VideoWriter', 
    'VideoInfo',
    'BoxingVisualizer',
    
    # Functions
    'create_comparison_video',
    'get_video_duration',
    'extract_key_frames',
    'convert_video_format',
    'create_3d_pose_plot',
    'create_performance_dashboard',
    'create_timeline_analysis',
    'create_comparison_chart',
    'save_visualization_report',
    'generate_pose_skeleton',
    'create_heatmap_analysis',
    'plot_movement_trajectory',
    
    # Utility functions
    'get_color_for_score',
    'get_performance_level',
    'validate_analysis_data',
    'setup_logging',
    
    # Constants
    'BOXING_COLOR_SCHEME',
    'PERFORMANCE_THRESHOLDS',
    'POSE_CONNECTIONS',
    'ANALYSIS_CONFIG',
    'SUPPORTED_VIDEO_FORMATS',
    'DEFAULT_OUTPUT_RESOLUTION',
    'DEFAULT_FPS',
    
    # Package info
    '__version__',
    '__author__',
    '__description__'
]

# Package initialization message
def print_package_info():
    """Print package information"""
    print(f"Boxing Analysis Utils v{__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")
    print("=" * 50)


