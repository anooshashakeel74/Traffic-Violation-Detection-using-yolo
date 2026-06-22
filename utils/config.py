"""
Configuration module for the traffic violation detection system
"""

import os

class Config:
    """Configuration class for the traffic violation detection system"""
    def __init__(self):
        # Detection settings
        self.speeding_threshold = 50  # km/h
        self.vehicle_confidence = 70  # percentage
        self.red_light_sensitivity = 80  # percentage
        self.show_traffic_lights = True
        self.enable_audio = False
        
        # YOLO model paths
        self.yolo_model_path = os.path.join("models", "yolov3")
        self.yolo_config_path = os.path.join("models", "yolov3.cfg")
        self.yolo_weights_path = os.path.join("models", "yolov3.weights")
        self.yolo_classes_path = os.path.join("models", "coco.names")
        
        # DeepSORT model path
        self.deepsort_model_path = os.path.join("models", "mars-small128.pb")
        
        # Tracker settings
        self.tracker_max_age = 30
        self.tracker_n_init = 3
        
        # Output settings
        self.output_dir = os.path.join("output")
