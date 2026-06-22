"""
Vehicle tracker module using DeepSORT
"""

import cv2
import numpy as np
import os
import sys
import urllib.request
import math

class TrackedVehicle:
    """Extended vehicle class with tracking information"""
    def __init__(self, id, vehicle, track_id):
        self.id = track_id  # Use track_id as the vehicle ID
        self.original_id = vehicle.id
        self.type = vehicle.type
        self.x = vehicle.x
        self.y = vehicle.y
        self.width = vehicle.width
        self.height = vehicle.height
        self.confidence = vehicle.confidence
        self.color = vehicle.color
        
        # Tracking specific attributes
        self.track_id = track_id
        self.positions = []
        self.speeds = []
        self.speed = 0
        self.first_detected = 0
        self.last_seen = 0
        self.is_violating = False
        self.violation_type = "none"
        self.violations = set()
        self.has_violated_red_light = False
    
    def update_position(self, x, y, width, height, timestamp):
        """Update vehicle position and calculate speed"""
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.last_seen = timestamp
        
        # Add position to history
        center_x = x + width // 2
        center_y = y + height // 2
        self.positions.append({"x": center_x, "y": center_y, "time": timestamp})
        
        # Keep only recent positions
        if len(self.positions) > 10:
            self.positions.pop(0)
        
        # Calculate speed
        self._calculate_speed()
    
    def _calculate_speed(self):
        """Calculate vehicle speed based on position history"""
        if len(self.positions) < 3:
            return
        
        # Use last 3 positions for speed calculation
        recent = self.positions[-3:]
        total_distance = 0
        total_time = 0
        
        for i in range(1, len(recent)):
            dx = recent[i]["x"] - recent[i-1]["x"]
            dy = recent[i]["y"] - recent[i-1]["y"]
            distance = math.sqrt(dx*dx + dy*dy)
            time_diff = recent[i]["time"] - recent[i-1]["time"]
            
            if time_diff > 0:
                total_distance += distance
                total_time += time_diff
        
        if total_time > 0:
            # Convert pixels to meters (calibration factor) and calculate km/h
            pixels_per_meter = 15  # This should be calibrated for the specific camera
            distance_meters = total_distance / pixels_per_meter
            speed_mps = distance_meters / total_time
            calculated_speed = speed_mps * 3.6  # Convert to km/h
            
            # Add to speed history
            self.speeds.append(calculated_speed)
            if len(self.speeds) > 5:
                self.speeds.pop(0)
            
            # Average speed for stability
            self.speed = sum(self.speeds) / len(self.speeds) if self.speeds else 0

class SimpleTracker:
    """Simple tracker implementation when DeepSORT is not available"""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_disappeared = 10
    
    def update(self, detections):
        """Update tracker with new detections"""
        if not detections:
            # Mark all tracks as disappeared
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return []
        
        # If no existing tracks, create new ones
        if not self.tracks:
            for detection in detections:
                self.tracks[self.next_id] = {
                    'detection': detection,
                    'disappeared': 0
                }
                self.next_id += 1
        else:
            # Match detections to existing tracks
            self._match_detections_to_tracks(detections)
        
        # Return current tracks
        return [track['detection'] for track in self.tracks.values()]
    
    def _match_detections_to_tracks(self, detections):
        """Match detections to existing tracks using simple distance"""
        # Calculate distances between detections and tracks
        track_ids = list(self.tracks.keys())
        distances = np.zeros((len(detections), len(track_ids)))
        
        for i, detection in enumerate(detections):
            det_center = (detection.x + detection.width // 2, detection.y + detection.height // 2)
            
            for j, track_id in enumerate(track_ids):
                track_det = self.tracks[track_id]['detection']
                track_center = (track_det.x + track_det.width // 2, track_det.y + track_det.height // 2)
                
                # Calculate Euclidean distance
                distances[i, j] = np.sqrt((det_center[0] - track_center[0])**2 + 
                                        (det_center[1] - track_center[1])**2)
        
        # Simple assignment: match closest detections to tracks
        used_detections = set()
        used_tracks = set()
        
        # Find best matches
        for _ in range(min(len(detections), len(track_ids))):
            min_dist = np.inf
            best_det_idx = -1
            best_track_idx = -1
            
            for i in range(len(detections)):
                if i in used_detections:
                    continue
                for j in range(len(track_ids)):
                    if j in used_tracks:
                        continue
                    if distances[i, j] < min_dist and distances[i, j] < 100:  # Distance threshold
                        min_dist = distances[i, j]
                        best_det_idx = i
                        best_track_idx = j
            
            if best_det_idx != -1 and best_track_idx != -1:
                # Update existing track
                track_id = track_ids[best_track_idx]
                self.tracks[track_id]['detection'] = detections[best_det_idx]
                self.tracks[track_id]['disappeared'] = 0
                used_detections.add(best_det_idx)
                used_tracks.add(best_track_idx)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                self.tracks[self.next_id] = {
                    'detection': detection,
                    'disappeared': 0
                }
                self.next_id += 1
        
        # Mark unmatched tracks as disappeared
        for j, track_id in enumerate(track_ids):
            if j not in used_tracks:
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]

class VehicleTracker:
    """Vehicle tracker using DeepSORT or simple tracking"""
    def __init__(self, model_path, max_age=30, n_init=3):
        self.model_path = model_path
        self.max_age = max_age
        self.n_init = n_init
        self.tracked_vehicles = []
        self.next_id = 1
        
        # Try to initialize DeepSORT, fallback to simple tracker
        self.use_deepsort = False  # Set to False to use simple tracker
        print("Using simple tracker instead of DeepSORT")
        self.simple_tracker = SimpleTracker()
        
        print(f"Vehicle tracker initialized (DeepSORT: {self.use_deepsort})")
    
    def update(self, vehicles, frame):
        """Update tracker with new detections"""
        if not vehicles:
            return []
        
        # Use simple tracker
        return self._update_with_simple_tracker(vehicles, frame)
    
    def _update_with_simple_tracker(self, vehicles, frame):
        """Update using simple tracker"""
        # Update simple tracker
        tracked_detections = self.simple_tracker.update(vehicles)
        
        # Convert to TrackedVehicle objects
        current_time = frame.shape[0] / 30
        updated_vehicles = []
        
        for i, vehicle in enumerate(tracked_detections):
            # Find existing tracked vehicle or create new one
            existing_vehicle = next((v for v in self.tracked_vehicles if v.original_id == vehicle.id), None)
            
            if existing_vehicle:
                # Update existing vehicle
                existing_vehicle.update_position(vehicle.x, vehicle.y, vehicle.width, vehicle.height, current_time)
                existing_vehicle.type = vehicle.type
                existing_vehicle.confidence = vehicle.confidence
                updated_vehicles.append(existing_vehicle)
            else:
                # Create new tracked vehicle
                new_vehicle = TrackedVehicle(
                    id=self.next_id,
                    vehicle=vehicle,
                    track_id=self.next_id
                )
                new_vehicle.first_detected = current_time
                new_vehicle.update_position(vehicle.x, vehicle.y, vehicle.width, vehicle.height, current_time)
                updated_vehicles.append(new_vehicle)
                self.next_id += 1
        
        self.tracked_vehicles = updated_vehicles
        return self.tracked_vehicles
