"""
Video Utilities Module - utils/video_utils.py
Handles video processing, frame extraction, and video-related operations
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Generator
import json
import time
from dataclasses import dataclass

@dataclass
class VideoInfo:
    """Data class to store video information"""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    file_size: int

class VideoProcessor:
    """Enhanced video processing utilities for boxing analysis"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    def get_video_info(self, video_path: str) -> VideoInfo:
        """Extract comprehensive video information"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Get codec information
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        # Get file size
        file_size = Path(video_path).stat().st_size
        
        cap.release()
        
        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec=codec,
            file_size=file_size
        )
    
    def extract_frames_generator(self, video_path: str, 
                               start_frame: int = 0, 
                               end_frame: Optional[int] = None,
                               step: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator to extract frames efficiently"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Set start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_idx = start_frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if end_frame is None:
            end_frame = total_frames
        
        while frame_idx < min(end_frame, total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if (frame_idx - start_frame) % step == 0:
                yield frame_idx, frame
            
            frame_idx += 1
        
        cap.release()
    
    def extract_frame_at_time(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """Extract frame at specific timestamp (seconds)"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        # Set position to timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None
    
    def resize_frame(self, frame: np.ndarray, 
                    target_size: Tuple[int, int], 
                    maintain_aspect: bool = True) -> np.ndarray:
        """Resize frame with optional aspect ratio preservation"""
        if not maintain_aspect:
            return cv2.resize(frame, target_size)
        
        # Calculate scaling factor
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create canvas and center the image
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def enhance_frame_quality(self, frame: np.ndarray) -> np.ndarray:
        """Apply quality enhancement filters to frame"""
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original and sharpened (70% enhanced, 30% original)
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def create_video_thumbnail(self, video_path: str, 
                             output_path: str, 
                             timestamp: float = None,
                             size: Tuple[int, int] = (320, 240)) -> bool:
        """Create thumbnail image from video"""
        try:
            # Use middle frame if no timestamp specified
            if timestamp is None:
                video_info = self.get_video_info(video_path)
                timestamp = video_info.duration / 2
            
            frame = self.extract_frame_at_time(video_path, timestamp)
            if frame is None:
                return False
            
            # Resize and save thumbnail
            thumbnail = self.resize_frame(frame, size, maintain_aspect=True)
            cv2.imwrite(output_path, thumbnail)
            return True
            
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            return False
    
    def extract_motion_vectors(self, video_path: str, 
                             max_frames: int = 100) -> List[Dict]:
        """Extract motion vectors for movement analysis"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return []
        
        motion_data = []
        prev_frame = None
        frame_count = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_frame, gray, 
                    np.array([[frame.shape[1]//2, frame.shape[0]//2]], dtype=np.float32),
                    None
                )
                
                # Calculate motion magnitude
                if len(flow[0]) > 0:
                    motion_magnitude = np.linalg.norm(flow[0][0] - 
                                                    np.array([frame.shape[1]//2, frame.shape[0]//2]))
                    
                    motion_data.append({
                        'frame': frame_count,
                        'motion_magnitude': float(motion_magnitude),
                        'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS)
                    })
            
            prev_frame = gray.copy()
            frame_count += 1
        
        cap.release()
        return motion_data
    
    def validate_video_file(self, video_path: str) -> Dict[str, bool]:
        """Comprehensive video file validation"""
        validation_result = {
            'file_exists': False,
            'supported_format': False,
            'readable': False,
            'has_frames': False,
            'valid_codec': False
        }
        
        file_path = Path(video_path)
        
        # Check file existence
        validation_result['file_exists'] = file_path.exists()
        if not validation_result['file_exists']:
            return validation_result
        
        # Check format
        validation_result['supported_format'] = file_path.suffix.lower() in self.supported_formats
        
        # Try to open and read
        try:
            cap = cv2.VideoCapture(video_path)
            validation_result['readable'] = cap.isOpened()
            
            if validation_result['readable']:
                # Check frame count
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                validation_result['has_frames'] = frame_count > 0
                
                # Try to read first frame
                ret, frame = cap.read()
                validation_result['valid_codec'] = ret and frame is not None
            
            cap.release()
            
        except Exception as e:
            print(f"Video validation error: {e}")
        
        return validation_result
    
    def batch_process_videos(self, video_paths: List[str], 
                           processor_func, 
                           progress_callback=None) -> Dict[str, any]:
        """Process multiple videos with progress tracking"""
        results = {}
        total_videos = len(video_paths)
        
        for i, video_path in enumerate(video_paths):
            try:
                print(f"Processing video {i+1}/{total_videos}: {Path(video_path).name}")
                
                # Validate video first
                validation = self.validate_video_file(video_path)
                if not all(validation.values()):
                    results[video_path] = {
                        'success': False,
                        'error': 'Video validation failed',
                        'validation': validation
                    }
                    continue
                
                # Process video
                start_time = time.time()
                result = processor_func(video_path)
                processing_time = time.time() - start_time
                
                results[video_path] = {
                    'success': True,
                    'result': result,
                    'processing_time': processing_time
                }
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(i + 1, total_videos, video_path)
                    
            except Exception as e:
                results[video_path] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"Error processing {video_path}: {e}")
        
        return results

class VideoWriter:
    """Utility class for creating output videos with annotations"""
    
    def __init__(self, output_path: str, fps: float = 30.0, 
                 codec: str = 'mp4v', resolution: Tuple[int, int] = (1280, 720)):
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.resolution = resolution
        self.writer = None
        self.is_initialized = False
    
    def initialize_writer(self, frame_shape: Tuple[int, int, int]):
        """Initialize video writer with frame dimensions"""
        if self.is_initialized:
            return
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.fps, 
            self.resolution
        )
        self.is_initialized = True
    
    def add_frame(self, frame: np.ndarray):
        """Add frame to output video"""
        if not self.is_initialized:
            self.initialize_writer(frame.shape)
        
        # Resize frame to match output resolution
        resized_frame = cv2.resize(frame, self.resolution)
        self.writer.write(resized_frame)
    
    def add_annotated_frame(self, frame: np.ndarray, annotations: Dict):
        """Add frame with overlay annotations"""
        annotated_frame = frame.copy()
        
        # Add text annotations
        if 'text' in annotations:
            for text_info in annotations['text']:
                cv2.putText(
                    annotated_frame,
                    text_info.get('text', ''),
                    text_info.get('position', (10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_info.get('scale', 0.7),
                    text_info.get('color', (255, 255, 255)),
                    text_info.get('thickness', 2)
                )
        
        # Add circle annotations (for landmarks)
        if 'circles' in annotations:
            for circle in annotations['circles']:
                cv2.circle(
                    annotated_frame,
                    circle.get('center', (0, 0)),
                    circle.get('radius', 5),
                    circle.get('color', (0, 255, 0)),
                    circle.get('thickness', -1)
                )
        
        # Add line annotations (for connections)
        if 'lines' in annotations:
            for line in annotations['lines']:
                cv2.line(
                    annotated_frame,
                    line.get('start', (0, 0)),
                    line.get('end', (10, 10)),
                    line.get('color', (255, 0, 0)),
                    line.get('thickness', 2)
                )
        
        self.add_frame(annotated_frame)
    
    def finalize(self):
        """Finalize and close video writer"""
        if self.writer:
            self.writer.release()
            self.is_initialized = False
            print(f"Video saved: {self.output_path}")

def create_comparison_video(video_paths: List[str], 
                          output_path: str, 
                          layout: str = 'horizontal') -> bool:
    """Create side-by-side comparison video from multiple inputs"""
    if len(video_paths) < 2:
        print("At least 2 videos required for comparison")
        return False
    
    try:
        caps = [cv2.VideoCapture(path) for path in video_paths]
        
        # Check if all videos opened successfully
        if not all(cap.isOpened() for cap in caps):
            print("Error opening one or more videos")
            for cap in caps:
                cap.release()
            return False
        
        # Get video properties
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
        min_frames = min(frame_counts)
        
        # Calculate output dimensions
        if layout == 'horizontal':
            out_width = sum(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps)
            out_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:  # vertical
            out_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
            out_height = sum(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps)
        
        # Initialize output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        for frame_idx in range(min_frames):
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break
            
            if len(frames) == len(caps):
                if layout == 'horizontal':
                    combined_frame = np.hstack(frames)
                else:  # vertical
                    combined_frame = np.vstack(frames)
                
                out.write(combined_frame)
        
        # Cleanup
        for cap in caps:
            cap.release()
        out.release()
        
        print(f"Comparison video created: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating comparison video: {e}")
        return False

# Utility functions for common video operations
def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds"""
    processor = VideoProcessor()
    try:
        info = processor.get_video_info(video_path)
        return info.duration
    except:
        return 0.0

def extract_key_frames(video_path: str, num_frames: int = 5) -> List[np.ndarray]:
    """Extract evenly distributed key frames from video"""
    processor = VideoProcessor()
    info = processor.get_video_info(video_path)
    
    if info.frame_count < num_frames:
        num_frames = info.frame_count
    
    frame_indices = np.linspace(0, info.frame_count - 1, num_frames, dtype=int)
    frames = []
    
    for frame_idx in frame_indices:
        timestamp = frame_idx / info.fps
        frame = processor.extract_frame_at_time(video_path, timestamp)
        if frame is not None:
            frames.append(frame)
    
    return frames

def convert_video_format(input_path: str, output_path: str, 
                        target_codec: str = 'mp4v') -> bool:
    """Convert video to different format/codec"""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output writer
        fourcc = cv2.VideoWriter_fourcc(*target_codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Copy frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Video converted: {output_path}")
        return True
        
    except Exception as e:
        print(f"Conversion error: {e}")
        return False