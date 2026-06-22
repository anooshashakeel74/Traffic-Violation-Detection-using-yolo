"""
Traffic light detection module
"""

import cv2
import numpy as np

class TrafficLight:
    """Traffic light class to store detection information"""
    def __init__(self, id, x, y, width, height, state, stop_line_y, confidence):
        self.id = id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.state = state
        self.stop_line_y = stop_line_y
        self.confidence = confidence
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "state": self.state,
            "stop_line_y": self.stop_line_y,
            "confidence": self.confidence
        }

class TrafficLightDetector:
    """Traffic light detector"""
    def __init__(self, sensitivity=0.8):
        self.sensitivity = sensitivity
        self.traffic_lights = []
        self.frame_count = 0
    
    def initialize_traffic_lights(self, width, height):
        """Initialize traffic light detection regions"""
        self.traffic_lights = [
            TrafficLight(
                id=1,
                x=int(width * 0.85),
                y=int(height * 0.05),
                width=int(width * 0.08),
                height=int(height * 0.15),
                state="red",  # Start with red light
                stop_line_y=int(height * 0.45),
                confidence=0.95
            ),
            TrafficLight(
                id=2,
                x=int(width * 0.1),
                y=int(height * 0.05),
                width=int(width * 0.08),
                height=int(height * 0.15),
                state="red",  # Start with red light
                stop_line_y=int(height * 0.75),
                confidence=0.92
            )
        ]
    
    def detect(self, frame):
        """Detect traffic lights in the frame"""
        self.frame_count += 1
        
        # If traffic lights are not initialized, do it now
        if not self.traffic_lights:
            height, width = frame.shape[:2]
            self.initialize_traffic_lights(width, height)
        
        # Process each traffic light
        for light in self.traffic_lights:
            # Extract traffic light region
            roi = frame[light.y:light.y+light.height, light.x:light.x+light.width]
            
            if roi.size == 0:
                continue
            
            # In a real implementation, we would use a more sophisticated approach
            # For now, we'll simulate traffic light state detection using a time-based cycle
            cycle_length = 200  # Shorter cycle for faster detection
            position = (self.frame_count % cycle_length) / cycle_length
            
            if position < 0.7:
                # Longer red phase (70% of the cycle)
                light.state = "red"
            elif position < 0.8:
                light.state = "yellow"
            else:
                light.state = "green"
        
        return self.traffic_lights
