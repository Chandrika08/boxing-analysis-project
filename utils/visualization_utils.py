"""
Visualization Utilities Module - utils/visualization_utils.py
Advanced visualization tools for boxing performance analysis
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import cv2
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    width: int = 1200
    height: int = 800
    background_color: str = '#1E1E1E'
    text_color: str = '#FFFFFF'
    primary_color: str = '#00CED1'
    secondary_color: str = '#FFD700'
    success_color: str = '#2E8B57'
    warning_color: str = '#FF8C00'
    error_color: str = '#DC143C'

class BoxingVisualizer:
    """Advanced visualization class for boxing analysis"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
        # Define pose connections for skeleton visualization
        self.pose_connections = [
            # Head and face
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # Torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Arms
            (11, 13), (13, 15), (12, 14), (14, 16),
            # Legs
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]
        
        # Color mapping for different body parts
        self.body_part_colors = {
            'head': '#FF6B6B',
            'torso': '#4ECDC4', 
            'left_arm': '#45B7D1',
            'right_arm': '#96CEB4',
            'left_leg': '#FFEAA7',
            'right_leg': '#DDA0DD'
        }
    
    def create_3d_pose_visualization(self, landmarks: Dict, 
                                   frame_info: Dict = None) -> go.Figure:
        """Create 3D pose visualization with skeleton"""
        fig = go.Figure()
        
        # Extract coordinates
        landmark_names = list(landmarks.keys())
        x_coords = [landmarks[name]['x'] for name in landmark_names]
        y_coords = [landmarks[name]['y'] for name in landmark_names]
        z_coords = [landmarks[name]['z'] for name in landmark_names]
        
        # Create landmark points
        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='markers',
            marker=dict(
                size=8,
                color=self.config.primary_color,
                opacity=0.8
            ),
            name='Body Landmarks',
            text=landmark_names,
            hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
        ))
        
        # Add skeleton connections
        for connection in self.pose_connections:
            if connection[0] < len(landmark_names) and connection[1] < len(landmark_names):
                name1, name2 = landmark_names[connection[0]], landmark_names[connection[1]]
                if name1 in landmarks and name2 in landmarks:
                    fig.add_trace(go.Scatter3d(
                        x=[landmarks[name1]['x'], landmarks[name2]['x']],
                        y=[landmarks[name1]['y'], landmarks[name2]['y']],
                        z=[landmarks[name1]['z'], landmarks[name2]['z']],
                        mode='lines',
                        line=dict(color=self.config.secondary_color, width=4),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Update layout
        fig.update_layout(
            title='3D Boxing Pose Analysis',
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position', 
                zaxis_title='Z Depth',
                bgcolor=self.config.background_color,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.text_color),
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def create_performance_dashboard(self, metrics: Dict, 
                                   detailed_scores: Dict = None) -> go.Figure:
        """Create comprehensive performance dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "radar"}, {"type": "pie"}]],
            subplot_titles=("Overall Score", "Component Breakdown", 
                          "Performance Radar", "Score Distribution"),
            vertical_spacing=0.15
        )
        
        # Overall score indicator
        overall_score = metrics.get('overall_score', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=overall_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Performance"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': self._get_color_for_score(overall_score)},
                'steps': [
                    {'range': [0, 40], 'color': self.config.error_color},
                    {'range': [40, 70], 'color': self.config.warning_color},
                    {'range': [70, 100], 'color': self.config.success_color}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=1, col=1)
        
        # Component breakdown bar chart
        components = ['Stance', 'Guard', 'Punch Form', 'Defense', 'Movement']
        scores = [
            metrics.get('stance_stability', 0),
            metrics.get('guard_position', 0),
            metrics.get('punch_form', 0),
            metrics.get('defensive_posture', 0),
            metrics.get('movement_efficiency', 0)
        ]
        
        colors = [self._get_color_for_score(score) for score in scores]
        
        fig.add_trace(go.Bar(
            x=components,
            y=scores,
            marker_color=colors,
            name='Component Scores',
            text=[f'{score:.1f}' for score in scores],
            textposition='auto'
        ), row=1, col=2)
        
        # Performance radar chart
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=components,
            fill='toself',
            name='Performance Profile',
            line_color=self.config.primary_color
        ), row=2, col=1)
        
        # Score distribution pie chart
        score_ranges = ['Excellent (85-100)', 'Good (70-84)', 'Average (55-69)', 'Poor (0-54)']
        excellent_count = sum(1 for s in scores if s >= 85)
        good_count = sum(1 for s in scores if 70 <= s < 85)
        average_count = sum(1 for s in scores if 55 <= s < 70)
        poor_count = sum(1 for s in scores if s < 55)
        
        fig.add_trace(go.Pie(
            labels=score_ranges,
            values=[excellent_count, good_count, average_count, poor_count],
            marker_colors=[self.config.success_color, self.config.primary_color, 
                          self.config.warning_color, self.config.error_color]
        ), row=2, col=2)
        
        fig.update_layout(
            title='Boxing Performance Dashboard',
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.text_color),
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def create_timeline_analysis(self, analysis_data: List[Dict]) -> go.Figure:
        """Create timeline analysis of performance metrics"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Punch Detection Timeline", "Movement Efficiency", "Stance Stability"),
            vertical_spacing=0.08
        )
        
        frames = [d['frame'] for d in analysis_data]
        
        # Punch detection timeline
        punch_events = []
        punch_speeds = []
        for d in analysis_data:
            punch_detected = d['punch']['left_punch'] or d['punch']['right_punch']
            punch_events.append(1 if punch_detected else 0)
            punch_speeds.append(d['punch']['punch_speed'] if punch_detected else 0)
        
        fig.add_trace(go.Scatter(
            x=frames, 
            y=punch_events,
            mode='markers+lines',
            name='Punch Events',
            marker=dict(color=self.config.error_color, size=8),
            line=dict(color=self.config.error_color, width=2)
        ), row=1, col=1)
        
        # Movement efficiency over time
        movement_scores = [d['movement']['movement_score'] for d in analysis_data]
        smoothed_movement = gaussian_filter1d(movement_scores, sigma=2)
        
        fig.add_trace(go.Scatter(
            x=frames,
            y=movement_scores,
            mode='lines+markers',
            name='Raw Movement Score',
            line=dict(color=self.config.primary_color, width=1),
            opacity=0.5
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=frames,
            y=smoothed_movement,
            mode='lines',
            name='Smoothed Movement',
            line=dict(color=self.config.primary_color, width=3)
        ), row=2, col=1)
        
        # Stance stability over time
        stance_scores = [d['stance']['stance_score'] for d in analysis_data]
        
        fig.add_trace(go.Scatter(
            x=frames,
            y=stance_scores,
            mode='lines',
            name='Stance Stability',
            fill='tonexty',
            line=dict(color=self.config.success_color, width=2)
        ), row=3, col=1)
        
        fig.update_layout(
            title='Performance Timeline Analysis',
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.text_color),
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def create_heatmap_analysis(self, analysis_data: List[Dict]) -> go.Figure:
        """Create heatmap showing performance patterns"""
        # Prepare data matrix
        metrics = ['stance_score', 'guard_score', 'defense_score', 'movement_score']
        frames = range(0, len(analysis_data), max(1, len(analysis_data) // 50))  # Sample frames
        
        heatmap_data = []
        for frame_idx in frames:
            if frame_idx < len(analysis_data):
                frame_data = analysis_data[frame_idx]
                row = [
                    frame_data['stance']['stance_score'],
                    frame_data['stance']['guard_score'],
                    frame_data['defense']['defense_score'],
                    frame_data['movement']['movement_score']
                ]
                heatmap_data.append(row)
        
        heatmap_matrix = np.array(heatmap_data).T
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_matrix,
            x=list(frames),
            y=['Stance', 'Guard', 'Defense', 'Movement'],
            colorscale='RdYlGn',
            zmin=0,
            zmax=100,
            colorbar=dict(title="Score")
        ))
        
        fig.update_layout(
            title='Performance Heatmap Over Time',
            xaxis_title='Frame Number',
            yaxis_title='Performance Component',
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.text_color),
            width=self.config.width,
            height=self.config.height // 2
        )
        
        return fig
    
    def plot_movement_trajectory(self, analysis_data: List[Dict], 
                               landmark_name: str = 'nose') -> go.Figure:
        """Plot 3D movement trajectory of specific landmark"""
        if not analysis_data or landmark_name not in analysis_data[0]['landmarks']:
            return go.Figure()
        
        # Extract trajectory data
        x_coords = []
        y_coords = []
        z_coords = []
        frames = []
        
        for i, data in enumerate(analysis_data):
            if landmark_name in data['landmarks']:
                landmark = data['landmarks'][landmark_name]
                x_coords.append(landmark['x'])
                y_coords.append(landmark['y'])
                z_coords.append(landmark['z'])
                frames.append(i)
        
        # Create trajectory plot
        fig = go.Figure()
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines+markers',
            marker=dict(
                size=4,
                color=frames,
                colorscale='Viridis',
                colorbar=dict(title="Frame")
            ),
            line=dict(color=self.config.primary_color, width=6),
            name=f'{landmark_name.title()} Trajectory'
        ))
        
        # Add start and end markers
        if len(x_coords) > 1:
            fig.add_trace(go.Scatter3d(
                x=[x_coords[0]], y=[y_coords[0]], z=[z_coords[0]],
                mode='markers',
                marker=dict(size=12, color=self.config.success_color, symbol='diamond'),
                name='Start Position'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[x_coords[-1]], y=[y_coords[-1]], z=[z_coords[-1]],
                mode='markers',
                marker=dict(size=12, color=self.config.error_color, symbol='square'),
                name='End Position'
            ))
        
        fig.update_layout(
            title=f'3D Movement Trajectory - {landmark_name.title()}',
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Depth',
                bgcolor=self.config.background_color
            ),
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.text_color),
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def create_comparison_chart(self, video_results: Dict[str, Any]) -> go.Figure:
        """Create comparison chart for multiple video analyses"""
        video_names = list(video_results.keys())
        
        # Extract metrics for comparison
        metrics_data = {
            'Overall Score': [],
            'Stance': [],
            'Guard': [],
            'Punch Form': [],
            'Defense': [],
            'Movement': []
        }
        
        for video_name in video_names:
            if 'metrics' in video_results[video_name]:
                metrics = video_results[video_name]['metrics']
                metrics_data['Overall Score'].append(metrics.overall_score)
                metrics_data['Stance'].append(metrics.stance_stability)
                metrics_data['Guard'].append(metrics.guard_position)
                metrics_data['Punch Form'].append(metrics.punch_form)
                metrics_data['Defense'].append(metrics.defensive_posture)
                metrics_data['Movement'].append(metrics.movement_efficiency)
        
        # Create comparison chart
        fig = go.Figure()
        
        for metric_name, values in metrics_data.items():
            fig.add_trace(go.Bar(
                name=metric_name,
                x=video_names,
                y=values,
                text=[f'{v:.1f}' for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Performance Comparison Across Videos',
            xaxis_title='Video',
            yaxis_title='Score',
            barmode='group',
            paper_bgcolor=self.config.background_color,
            font=dict(color=self.config.text_color),
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def _get_color_for_score(self, score: float) -> str:
        """Get color based on performance score"""
        if score >= 85:
            return self.config.success_color
        elif score >= 70:
            return self.config.primary_color
        elif score >= 55:
            return self.config.warning_color
        else:
            return self.config.error_color
    
    def save_all_visualizations(self, analysis_data: List[Dict], 
                              metrics: Any, video_name: str,
                              output_dir: str = "outputs/visualizations") -> Dict[str, str]:
        """Save all visualizations for a video analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            # 3D Pose visualization
            if analysis_data:
                sample_landmarks = analysis_data[len(analysis_data)//2]['landmarks']
                pose_fig = self.create_3d_pose_visualization(sample_landmarks)
                pose_file = output_path / f"{video_name}_3d_pose.html"
                pose_fig.write_html(str(pose_file))
                saved_files['3d_pose'] = str(pose_file)
            
            # Performance dashboard
            metrics_dict = {
                'overall_score': metrics.overall_score,
                'stance_stability': metrics.stance_stability,
                'guard_position': metrics.guard_position,
                'punch_form': metrics.punch_form,
                'defensive_posture': metrics.defensive_posture,
                'movement_efficiency': metrics.movement_efficiency
            }
            dashboard_fig = self.create_performance_dashboard(metrics_dict)
            dashboard_file = output_path / f"{video_name}_dashboard.html"
            dashboard_fig.write_html(str(dashboard_file))
            saved_files['dashboard'] = str(dashboard_file)
            
            # Timeline analysis
            timeline_fig = self.create_timeline_analysis(analysis_data)
            timeline_file = output_path / f"{video_name}_timeline.html"
            timeline_fig.write_html(str(timeline_file))
            saved_files['timeline'] = str(timeline_file)
            
            # Heatmap analysis
            heatmap_fig = self.create_heatmap_analysis(analysis_data)
            heatmap_file = output_path / f"{video_name}_heatmap.html"
            heatmap_fig.write_html(str(heatmap_file))
            saved_files['heatmap'] = str(heatmap_file)
            
            # Movement trajectory
            trajectory_fig = self.plot_movement_trajectory(analysis_data, 'nose')
            trajectory_file = output_path / f"{video_name}_trajectory.html"
            trajectory_fig.write_html(str(trajectory_file))
            saved_files['trajectory'] = str(trajectory_file)
            
        except Exception as e:
            print(f"Error saving visualizations: {e}")
        
        return saved_files

# Standalone utility functions
def create_3d_pose_plot(landmarks: Dict, title: str = "3D Pose Analysis") -> go.Figure:
    """Create a simple 3D pose plot"""
    visualizer = BoxingVisualizer()
    fig = visualizer.create_3d_pose_visualization(landmarks)
    fig.update_layout(title=title)
    return fig

def create_performance_dashboard(metrics: Dict, title: str = "Performance Dashboard") -> go.Figure:
    """Create a performance dashboard"""
    visualizer = BoxingVisualizer()
    fig = visualizer.create_performance_dashboard(metrics)
    fig.update_layout(title=title)
    return fig

def create_timeline_analysis(analysis_data: List[Dict], title: str = "Timeline Analysis") -> go.Figure:
    """Create timeline analysis"""
    visualizer = BoxingVisualizer()
    fig = visualizer.create_timeline_analysis(analysis_data)
    fig.update_layout(title=title)
    return fig

def create_comparison_chart(video_results: Dict, title: str = "Video Comparison") -> go.Figure:
    """Create comparison chart"""
    visualizer = BoxingVisualizer()
    fig = visualizer.create_comparison_chart(video_results)
    fig.update_layout(title=title)
    return fig

def save_visualization_report(figures: List[go.Figure], output_file: str) -> bool:
    """Save multiple visualizations to a single HTML report"""
    try:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Boxing Analysis Report</title>
            <style>
                body { background-color: #1E1E1E; color: white; font-family: Arial, sans-serif; }
                .visualization { margin: 20px 0; }
                h1, h2 { color: #00CED1; }
            </style>
        </head>
        <body>
            <h1>Boxing Performance Analysis Report</h1>
        """
        
        for i, fig in enumerate(figures):
            html_content += f'<div class="visualization">{fig.to_html(include_plotlyjs="inline")}</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return True
    except Exception as e:
        print(f"Error saving visualization report: {e}")
        return False

def generate_pose_skeleton(landmarks: Dict) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Generate skeleton connections for pose visualization"""
    connections = []
    pose_connections = [
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle')
    ]
    
    for start_point, end_point in pose_connections:
        if start_point in landmarks and end_point in landmarks:
            start_coord = (
                landmarks[start_point]['x'],
                landmarks[start_point]['y'],
                landmarks[start_point]['z']
            )
            end_coord = (
                landmarks[end_point]['x'],
                landmarks[end_point]['y'],
                landmarks[end_point]['z']
            )
            connections.append((start_coord, end_coord))
    
    return connections

def create_heatmap_analysis(analysis_data: List[Dict], title: str = "Performance Heatmap") -> go.Figure:
    """Create heatmap analysis"""
    visualizer = BoxingVisualizer()
    fig = visualizer.create_heatmap_analysis(analysis_data)
    fig.update_layout(title=title)
    return fig

def plot_movement_trajectory(analysis_data: List[Dict], landmark: str = 'nose', 
                           title: str = "Movement Trajectory") -> go.Figure:
    """Plot movement trajectory"""
    visualizer = BoxingVisualizer()
    fig = visualizer.plot_movement_trajectory(analysis_data, landmark)
    fig.update_layout(title=title)
    return fig