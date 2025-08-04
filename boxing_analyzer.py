import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

@dataclass
class BoxingMetrics:
    """Data class to store boxing performance metrics"""
    stance_stability: float
    guard_position: float
    punch_speed: float
    punch_form: float
    defensive_posture: float
    movement_efficiency: float
    overall_score: float

class BoxingAnalyzer:
    """
    Advanced Boxing Performance Analysis System
    Analyzes video input to provide 3D visualization and corrective feedback
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key pose landmarks for boxing analysis
        self.key_landmarks = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'nose': 0, 'left_eye': 1, 'right_eye': 2
        }
        
        # Initialize tracking variables
        self.pose_history = []
        self.punch_events = []
        self.stance_data = []
        
    def extract_pose_landmarks(self, frame) -> Optional[Dict]:
        """Extract pose landmarks from frame"""
        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = {}
            for name, idx in self.key_landmarks.items():
                landmark = results.pose_landmarks.landmark[idx]
                landmarks[name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            return landmarks
        return None
    
    def analyze_stance_and_guard(self, landmarks: Dict) -> Dict:
        """Analyze boxing stance and guard position"""
        # Calculate stance width
        left_ankle = landmarks['left_ankle']
        right_ankle = landmarks['right_ankle']
        stance_width = abs(left_ankle['x'] - right_ankle['x'])
        
        # Analyze weight distribution
        left_hip = landmarks['left_hip']
        right_hip = landmarks['right_hip']
        weight_balance = abs(left_hip['y'] - right_hip['y'])
        
        # Guard position analysis
        left_wrist = landmarks['left_wrist']
        right_wrist = landmarks['right_wrist']
        nose = landmarks['nose']
        
        # Calculate guard height relative to face
        guard_height_left = nose['y'] - left_wrist['y']
        guard_height_right = nose['y'] - right_wrist['y']
        
        # Guard positioning score (0-100)
        ideal_stance_width = 0.3  # Normalized width
        ideal_guard_height = 0.15  # Relative to body height
        
        stance_score = max(0, 100 - abs(stance_width - ideal_stance_width) * 300)
        guard_score = max(0, 100 - abs(max(guard_height_left, guard_height_right) - ideal_guard_height) * 500)
        
        return {
            'stance_width': stance_width,
            'weight_balance': weight_balance,
            'guard_height_left': guard_height_left,
            'guard_height_right': guard_height_right,
            'stance_score': stance_score,
            'guard_score': guard_score
        }
    
    def detect_punch_mechanics(self, landmarks: Dict, frame_idx: int) -> Dict:
        """Detect and analyze punch mechanics"""
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']
        left_elbow = landmarks['left_elbow']
        right_elbow = landmarks['right_elbow']
        left_wrist = landmarks['left_wrist']
        right_wrist = landmarks['right_wrist']
        
        # Calculate arm extension
        def calculate_arm_extension(shoulder, elbow, wrist):
            # Vector from shoulder to elbow
            se_vector = np.array([elbow['x'] - shoulder['x'], elbow['y'] - shoulder['y']])
            # Vector from elbow to wrist
            ew_vector = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])
            
            # Calculate angle between vectors
            dot_product = np.dot(se_vector, ew_vector)
            norms = np.linalg.norm(se_vector) * np.linalg.norm(ew_vector)
            if norms > 0:
                angle = np.arccos(np.clip(dot_product / norms, -1, 1))
                return np.degrees(angle)
            return 0
        
        left_arm_angle = calculate_arm_extension(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_arm_extension(right_shoulder, right_elbow, right_wrist)
        
        # Detect punch (arm extension > threshold)
        punch_threshold = 140  # degrees
        left_punch = left_arm_angle > punch_threshold
        right_punch = right_arm_angle > punch_threshold
        
        # Calculate punch speed (if previous frame available)
        punch_speed = 0
        if len(self.pose_history) > 0:
            prev_landmarks = self.pose_history[-1]
            if left_punch:
                prev_wrist = prev_landmarks['left_wrist']
                speed = np.sqrt((left_wrist['x'] - prev_wrist['x'])**2 + 
                              (left_wrist['y'] - prev_wrist['y'])**2)
                punch_speed = speed * 30  # Assuming 30 FPS
            elif right_punch:
                prev_wrist = prev_landmarks['right_wrist']
                speed = np.sqrt((right_wrist['x'] - prev_wrist['x'])**2 + 
                              (right_wrist['y'] - prev_wrist['y'])**2)
                punch_speed = speed * 30
        
        return {
            'left_arm_angle': left_arm_angle,
            'right_arm_angle': right_arm_angle,
            'left_punch': left_punch,
            'right_punch': right_punch,
            'punch_speed': punch_speed,
            'frame_idx': frame_idx
        }
    
    def analyze_defensive_posture(self, landmarks: Dict) -> Dict:
        """Analyze defensive positioning and movement"""
        # Head position relative to shoulders
        nose = landmarks['nose']
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']
        
        # Calculate head centering
        shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        head_alignment = abs(nose['x'] - shoulder_center_x)
        
        # Chin tuck analysis
        chin_height = nose['y'] - (left_shoulder['y'] + right_shoulder['y']) / 2
        
        # Shoulder positioning for defense
        shoulder_level = abs(left_shoulder['y'] - right_shoulder['y'])
        
        # Calculate defensive score
        defense_score = max(0, 100 - head_alignment * 500 - shoulder_level * 300)
        
        return {
            'head_alignment': head_alignment,
            'chin_height': chin_height,
            'shoulder_level': shoulder_level,
            'defense_score': defense_score
        }
    
    def analyze_movement_footwork(self, landmarks: Dict) -> Dict:
        """Analyze movement patterns and footwork"""
        left_ankle = landmarks['left_ankle']
        right_ankle = landmarks['right_ankle']
        left_knee = landmarks['left_knee']
        right_knee = landmarks['right_knee']
        
        # Calculate knee bend (indicates readiness)
        def calculate_knee_bend(hip, knee, ankle):
            hip_knee = np.array([knee['x'] - hip['x'], knee['y'] - hip['y']])
            knee_ankle = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])
            
            dot_product = np.dot(hip_knee, knee_ankle)
            norms = np.linalg.norm(hip_knee) * np.linalg.norm(knee_ankle)
            if norms > 0:
                angle = np.arccos(np.clip(dot_product / norms, -1, 1))
                return np.degrees(angle)
            return 180
        
        left_knee_bend = calculate_knee_bend(landmarks['left_hip'], left_knee, left_ankle)
        right_knee_bend = calculate_knee_bend(landmarks['right_hip'], right_knee, right_ankle)
        
        # Movement efficiency score
        ideal_knee_bend = 165  # Slight bend for readiness
        movement_score = max(0, 100 - abs(left_knee_bend - ideal_knee_bend) - 
                           abs(right_knee_bend - ideal_knee_bend))
        
        return {
            'left_knee_bend': left_knee_bend,
            'right_knee_bend': right_knee_bend,
            'movement_score': movement_score
        }
    
    def calculate_overall_metrics(self, analysis_data: List[Dict]) -> BoxingMetrics:
        """Calculate overall performance metrics from analysis data"""
        if not analysis_data:
            return BoxingMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Aggregate scores
        stance_scores = [d['stance']['stance_score'] for d in analysis_data if 'stance' in d]
        guard_scores = [d['stance']['guard_score'] for d in analysis_data if 'stance' in d]
        defense_scores = [d['defense']['defense_score'] for d in analysis_data if 'defense' in d]
        movement_scores = [d['movement']['movement_score'] for d in analysis_data if 'movement' in d]
        
        # Punch form analysis
        punch_data = [d['punch'] for d in analysis_data if 'punch' in d]
        punch_speeds = [p['punch_speed'] for p in punch_data if p['punch_speed'] > 0]
        
        # Calculate averages
        stance_stability = np.mean(stance_scores) if stance_scores else 0
        guard_position = np.mean(guard_scores) if guard_scores else 0
        defensive_posture = np.mean(defense_scores) if defense_scores else 0
        movement_efficiency = np.mean(movement_scores) if movement_scores else 0
        punch_speed = np.mean(punch_speeds) if punch_speeds else 0
        
        # Punch form score based on technique consistency
        punch_form = 75 if punch_speeds else 50  # Base score, improved with actual analysis
        
        # Overall score
        overall_score = (stance_stability + guard_position + punch_form + 
                        defensive_posture + movement_efficiency) / 5
        
        return BoxingMetrics(
            stance_stability=stance_stability,
            guard_position=guard_position,
            punch_speed=punch_speed,
            punch_form=punch_form,
            defensive_posture=defensive_posture,
            movement_efficiency=movement_efficiency,
            overall_score=overall_score
        )
    
    def process_video(self, video_path: str) -> Tuple[List[Dict], BoxingMetrics]:
        """Process video and return analysis data and metrics"""
        cap = cv2.VideoCapture(video_path)
        analysis_data = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Skip empty or broken frames
            if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                print(f"⚠️ Skipping invalid frame {frame_idx}")
                continue
            resized_frame = cv2.resize(frame, (640, 480))
            landmarks = self.extract_pose_landmarks(frame)
            if landmarks:
                # Analyze different aspects
                stance_analysis = self.analyze_stance_and_guard(landmarks)
                punch_analysis = self.detect_punch_mechanics(landmarks, frame_idx)
                defense_analysis = self.analyze_defensive_posture(landmarks)
                movement_analysis = self.analyze_movement_footwork(landmarks)
                
                frame_data = {
                    'frame': frame_idx,
                    'landmarks': landmarks,
                    'stance': stance_analysis,
                    'punch': punch_analysis,
                    'defense': defense_analysis,
                    'movement': movement_analysis
                }
                
                analysis_data.append(frame_data)
                self.pose_history.append(landmarks)
            
            frame_idx += 1
        
        cap.release()
        
        # Calculate overall metrics
        metrics = self.calculate_overall_metrics(analysis_data)
        
        return analysis_data, metrics
    
    def create_3d_visualization(self, analysis_data: List[Dict], metrics: BoxingMetrics, 
                              output_path: str):
        """Create 3D visualization with feedback"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]],
            subplot_titles=("3D Pose Analysis", "Performance Metrics",
                           "Punch Detection Timeline", "Movement Analysis"),
            vertical_spacing=0.1
        )
        
        # 1. 3D Pose visualization
        if analysis_data:
            sample_data = analysis_data[len(analysis_data)//2]  # Middle frame
            landmarks = sample_data['landmarks']
            
            # Extract 3D coordinates
            x_coords = [landmarks[name]['x'] for name in self.key_landmarks.keys()]
            y_coords = [landmarks[name]['y'] for name in self.key_landmarks.keys()]
            z_coords = [landmarks[name]['z'] for name in self.key_landmarks.keys()]
            
            fig.add_trace(
                go.Scatter3d(x=x_coords, y=y_coords, z=z_coords,
                           mode='markers+lines', name='Body Pose',
                           marker=dict(size=5, color='blue')),
                row=1, col=1
            )
        
        # 2. Performance metrics bar chart
        metrics_names = ['Stance', 'Guard', 'Punch Form', 'Defense', 'Movement']
        metrics_values = [metrics.stance_stability, metrics.guard_position,
                         metrics.punch_form, metrics.defensive_posture,
                         metrics.movement_efficiency]
        
        colors = ['green' if v >= 70 else 'orange' if v >= 50 else 'red' for v in metrics_values]
        
        fig.add_trace(
            go.Bar(x=metrics_names, y=metrics_values, name='Scores',
                  marker_color=colors),
            row=1, col=2
        )
        
        # 3. Punch detection timeline
        frames = [d['frame'] for d in analysis_data]
        punch_events = [1 if (d['punch']['left_punch'] or d['punch']['right_punch']) else 0 
                       for d in analysis_data]
        
        fig.add_trace(
            go.Scatter(x=frames, y=punch_events, mode='markers+lines',
                      name='Punch Events', marker_color='red'),
            row=2, col=1
        )
        
        # 4. Movement efficiency over time
        movement_scores = [d['movement']['movement_score'] for d in analysis_data]
        
        fig.add_trace(
            go.Scatter(x=frames, y=movement_scores, mode='lines',
                      name='Movement Efficiency', line_color='blue'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Boxing Performance Analysis - Overall Score: {metrics.overall_score:.1f}/100",
            height=800,
            showlegend=True
        )
        
        # Save visualization
        fig.write_html(output_path)
        print(f"3D visualization saved to: {output_path}")
    
    def generate_feedback_report(self, metrics: BoxingMetrics, analysis_data: List[Dict]) -> str:
        """Generate detailed feedback report"""
        report = f"""
BOXING PERFORMANCE ANALYSIS REPORT
================================

OVERALL SCORE: {metrics.overall_score:.1f}/100

DETAILED BREAKDOWN:
------------------
• Stance Stability: {metrics.stance_stability:.1f}/100
• Guard Position: {metrics.guard_position:.1f}/100  
• Punch Form: {metrics.punch_form:.1f}/100
• Defensive Posture: {metrics.defensive_posture:.1f}/100
• Movement Efficiency: {metrics.movement_efficiency:.1f}/100
• Average Punch Speed: {metrics.punch_speed:.2f} units/frame

CORRECTIVE FEEDBACK:
-------------------
"""
        
        # Generate specific feedback based on scores
        if metrics.stance_stability < 70:
            report += "⚠️ STANCE: Improve foot positioning and weight distribution. Keep feet shoulder-width apart.\n"
        else:
            report += "✅ STANCE: Good stability and positioning.\n"
            
        if metrics.guard_position < 70:
            report += "⚠️ GUARD: Keep hands higher to protect your face. Maintain consistent guard height.\n"
        else:
            report += "✅ GUARD: Excellent defensive positioning.\n"
            
        if metrics.punch_form < 70:
            report += "⚠️ PUNCH FORM: Focus on full arm extension and proper joint alignment.\n"
        else:
            report += "✅ PUNCH FORM: Good technique and execution.\n"
            
        if metrics.defensive_posture < 70:
            report += "⚠️ DEFENSE: Keep chin tucked and head centered over shoulders.\n"
        else:
            report += "✅ DEFENSE: Strong defensive positioning.\n"
            
        if metrics.movement_efficiency < 70:
            report += "⚠️ MOVEMENT: Maintain slight knee bend for better mobility and reaction time.\n"
        else:
            report += "✅ MOVEMENT: Efficient footwork and positioning.\n"
        
        # Punch analysis
        punch_count = sum(1 for d in analysis_data if d['punch']['left_punch'] or d['punch']['right_punch'])
        report += f"\nPUNCH ANALYSIS:\n• Total punches detected: {punch_count}\n"
        
        if metrics.punch_speed > 0:
            report += f"• Average punch speed: {metrics.punch_speed:.2f} units/frame\n"
        
        report += f"\nTRAINING RECOMMENDATIONS:\n"
        report += "• Focus on areas scoring below 70/100\n"
        report += "• Practice shadow boxing for form improvement\n"
        report += "• Work on defensive drills for better posture\n"
        report += "• Strengthen core for better stance stability\n"
        
        return report

def main():
    """Main function to process boxing videos"""
    analyzer = BoxingAnalyzer()
    
    # Video paths (update these with actual paths from Google Drive)
    video_paths = [
        "C:/Users/akkis/OneDrive/Desktop/Boxing_Analysis_Project/data/boxing_video_1.mp4",
        "C:/Users/akkis/OneDrive/Desktop/Boxing_Analysis_Project/data/boxing_video_2.mp4", 
        "C:/Users/akkis/OneDrive/Desktop/Boxing_Analysis_Project/data/boxing_video_3.mp4",
        "C:/Users/akkis/OneDrive/Desktop/Boxing_Analysis_Project/data/boxing_video_4.mp4",
        "C:/Users/akkis\OneDrive/Desktop/Boxing_Analysis_Project/data/boxing_video_5.mp4"
    ]
    
    results = {}
    
    for i, video_path in enumerate(video_paths, 1):
        if os.path.exists(video_path):
            print(f"Processing Video {i}: {video_path}")
            
            # Process video
            analysis_data, metrics = analyzer.process_video(video_path)
            
            # Create 3D visualization
            viz_output = f"boxing_analysis_3d_video_{i}.html"
            analyzer.create_3d_visualization(analysis_data, metrics, viz_output)
            
            # Generate feedback report
            report = analyzer.generate_feedback_report(metrics, analysis_data)
            
            # Save report
            report_output = f"boxing_feedback_report_video_{i}.txt"
            with open(report_output, 'w', encoding='utf-8') as f:
                f.write(report)
            
            results[f"video_{i}"] = {
                'metrics': metrics,
                'analysis_data': analysis_data,
                'visualization': viz_output,
                'report': report_output
            }
            
            print(f"✅ Completed Video {i}")
            print(f"Overall Score: {metrics.overall_score:.1f}/100")
            print("-" * 50)
        else:
            print(f"❌ Video {i} not found: {video_path}")
    
    # Save combined results
    print("\n📊 SUMMARY OF ALL VIDEOS:")
    for video_name, result in results.items():
        metrics = result['metrics']
        print(f"{video_name}: {metrics.overall_score:.1f}/100")
    
    print(f"\n✅ Analysis complete! Generated {len(results)} video analyses.")
    print("Files created:")
    for result in results.values():
        print(f"  - {result['visualization']}")
        print(f"  - {result['report']}")

if __name__ == "__main__":
    main()