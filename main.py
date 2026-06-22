#!/usr/bin/env python3
"""
Traffic Violation Detection System
Main application entry point
"""

import os
import sys
import cv2
import numpy as np
import time
import argparse
import threading
import webbrowser
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_socketio import SocketIO, emit
import base64
import json
from utils.detector import VehicleDetector
from utils.tracker import VehicleTracker
from utils.traffic_light import TrafficLightDetector
from utils.violation_detector import ViolationDetector
from utils.config import Config

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'traffic_violation_detection'

# Initialize SocketIO with threading mode (compatible with Python 3.13)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
config = Config()
vehicle_detector = None
vehicle_tracker = None
traffic_light_detector = None
violation_detector = None

processing = False
current_frame = None
processed_frame = None
frame_count = 0
fps = 0
last_fps_time = 0
stats = {
    "total_vehicles": 0,
    "red_light_violations": 0,
    "speeding_violations": 0,
    "processing_fps": 0,
    "detection_accuracy": 0
}
violation_log = []
notification_queue = []

def initialize_system():
    """Initialize the detection system components"""
    global vehicle_detector, vehicle_tracker, traffic_light_detector, violation_detector
    
    print("Initializing detection system...")
    
    # Initialize vehicle detector with YOLO
    vehicle_detector = VehicleDetector(
        config.yolo_model_path,
        config.yolo_config_path,
        config.yolo_weights_path,
        config.yolo_classes_path,
        confidence_threshold=config.vehicle_confidence/100
    )
    
    # Initialize DeepSORT tracker
    vehicle_tracker = VehicleTracker(
        config.deepsort_model_path,
        max_age=config.tracker_max_age,
        n_init=config.tracker_n_init
    )
    
    # Initialize traffic light detector
    traffic_light_detector = TrafficLightDetector(
        sensitivity=config.red_light_sensitivity/100
    )
    
    # Initialize violation detector
    violation_detector = ViolationDetector(
        speed_threshold=config.speeding_threshold,
        red_light_sensitivity=config.red_light_sensitivity/100
    )
    
    print("Detection system initialized successfully!")

def process_frame(frame, timestamp):
    """Process a single frame for violations"""
    global frame_count, fps, last_fps_time, stats, violation_log, notification_queue
    
    if frame is None:
        return None
    
    # Calculate FPS
    frame_count += 1
    current_time = time.time()
    if current_time - last_fps_time >= 1:
        fps = frame_count
        frame_count = 0
        last_fps_time = current_time
        stats["processing_fps"] = fps
    
    # Copy frame for processing
    processed_image = frame.copy()
    
    # Step 1: Detect traffic lights
    traffic_lights = traffic_light_detector.detect(processed_image)
    
    # Step 2: Detect vehicles using YOLO
    vehicles = vehicle_detector.detect(processed_image)
    
    # Step 3: Track vehicles using DeepSORT
    tracked_vehicles = vehicle_tracker.update(vehicles, processed_image)
    
    # Step 4: Detect violations
    violations, new_red_light_violations, new_speeding_violations = violation_detector.detect_violations(
        tracked_vehicles, 
        traffic_lights, 
        timestamp
    )
    
    # Update statistics
    stats["total_vehicles"] = len(tracked_vehicles)
    stats["red_light_violations"] += new_red_light_violations
    stats["speeding_violations"] += new_speeding_violations
    
    # Calculate detection accuracy (based on confidence scores)
    if tracked_vehicles:
        high_confidence_vehicles = [v for v in tracked_vehicles if v.confidence > 0.8]
        stats["detection_accuracy"] = int((len(high_confidence_vehicles) / len(tracked_vehicles)) * 100)
    
    # Add new violation logs
    for violation in violations:
        if violation.is_new:
            log_message = f"[{time.strftime('%H:%M:%S')}] Vehicle #{violation.vehicle_id} ({violation.vehicle_type}) "
            
            if violation.type == "redLight":
                log_message += f"ran red light at traffic light #{violation.traffic_light_id}"
                notification = {
                    "type": "redLight",
                    "message": f"RED LIGHT VIOLATION DETECTED! Vehicle #{violation.vehicle_id} ({violation.vehicle_type.upper()}) crossed red light #{violation.traffic_light_id}"
                }
                notification_queue.append(notification)
            
            elif violation.type == "speeding":
                log_message += f"detected speeding at {int(violation.speed)} km/h"
                notification = {
                    "type": "speeding",
                    "message": f"SPEEDING VIOLATION! Vehicle #{violation.vehicle_id} ({violation.vehicle_type.upper()}) exceeding {config.speeding_threshold} km/h at {int(violation.speed)} km/h"
                }
                notification_queue.append(notification)
            
            violation_log.append(log_message)
    
    # Keep only the last 50 logs
    violation_log = violation_log[-50:] if violation_log else []
    
    # Draw visualizations if enabled
    if config.show_traffic_lights:
        for light in traffic_lights:
            # Draw traffic light box
            cv2.rectangle(processed_image, 
                         (light.x, light.y), 
                         (light.x + light.width, light.y + light.height), 
                         (255, 255, 255), 2)
            
            # Draw traffic light state
            color_map = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}
            color = color_map.get(light.state, (128, 128, 128))
            cv2.rectangle(processed_image, 
                         (light.x + 5, light.y + 5), 
                         (light.x + light.width - 5, light.y + light.height // 3), 
                         color, -1)
            
            # Draw stop line
            line_color = (68, 68, 239) if light.state == "red" else (128, 128, 128)  # BGR format
            line_thickness = 4 if light.state == "red" else 2
            cv2.line(processed_image, 
                    (0, light.stop_line_y), 
                    (processed_image.shape[1], light.stop_line_y), 
                    line_color, line_thickness)
            
            # Label
            cv2.putText(processed_image, 
                       f"TL{light.id}: {light.state.upper()}", 
                       (light.x, light.y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw violation indicators
    for violation in violations:
        if violation.is_active:
            center_x = violation.x + violation.width // 2
            center_y = violation.y + violation.height // 2
            
            if violation.type in ["redLight", "both"]:
                # Red light violation indicator
                cv2.circle(processed_image, (center_x, center_y), 30, (68, 68, 239), -1)
                
                # Add text label
                cv2.putText(processed_image, "RED LIGHT", (center_x - 40, center_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(processed_image, "VIOLATION", (center_x - 40, center_y + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            elif violation.type == "speeding":
                # Speeding violation indicator
                cv2.circle(processed_image, (center_x, center_y), 30, (11, 158, 245), -1)
                
                # Add text label
                cv2.putText(processed_image, "SPEEDING", (center_x - 30, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return processed_image

# Flask routes
@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/detection')
def detection():
    """Detection page route"""
    return render_template('detection.html')

@app.route('/api/stats')
def get_stats():
    """Get detection statistics"""
    return jsonify(stats)

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """Get or update detection settings"""
    global config
    
    if request.method == 'POST':
        data = request.json
        if data:
            config.speeding_threshold = data.get('speedingThreshold', config.speeding_threshold)
            config.vehicle_confidence = data.get('vehicleConfidence', config.vehicle_confidence)
            config.red_light_sensitivity = data.get('redLightSensitivity', config.red_light_sensitivity)
            config.show_traffic_lights = data.get('showTrafficLights', config.show_traffic_lights)
            config.enable_audio = data.get('enableAudio', config.enable_audio)
            
            # Update components with new settings
            if vehicle_detector:
                vehicle_detector.confidence_threshold = config.vehicle_confidence / 100
            if traffic_light_detector:
                traffic_light_detector.sensitivity = config.red_light_sensitivity / 100
            if violation_detector:
                violation_detector.speed_threshold = config.speeding_threshold
                violation_detector.red_light_sensitivity = config.red_light_sensitivity / 100
                
        return jsonify({"success": True})
    else:
        return jsonify({
            "speedingThreshold": config.speeding_threshold,
            "vehicleConfidence": config.vehicle_confidence,
            "redLightSensitivity": config.red_light_sensitivity,
            "showTrafficLights": config.show_traffic_lights,
            "enableAudio": config.enable_audio
        })

@app.route('/api/violations')
def get_violations():
    """Get violation log"""
    return jsonify({"violations": violation_log})

@app.route('/api/traffic-lights')
def get_traffic_lights():
    """Get traffic light status"""
    if traffic_light_detector:
        return jsonify([light.to_dict() for light in traffic_light_detector.traffic_lights])
    return jsonify([])

# Socket.IO event handlers
@socketio.on('start_processing')
def handle_start_processing():
    """Start video processing"""
    global processing
    processing = True
    emit('processing_started', {'status': 'started'})

@socketio.on('stop_processing')
def handle_stop_processing():
    """Stop video processing"""
    global processing
    processing = False
    emit('processing_stopped', {'status': 'stopped'})

@socketio.on('process_frame')
def handle_process_frame(data):
    """Process a video frame"""
    global current_frame, processed_frame, notification_queue
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            # Initialize traffic lights if not done
            if traffic_light_detector and not traffic_light_detector.traffic_lights:
                height, width = frame.shape[:2]
                traffic_light_detector.initialize_traffic_lights(width, height)
            
            # Process frame
            timestamp = data.get('timestamp', time.time())
            processed_image = process_frame(frame, timestamp)
            
            # Encode processed frame back to base64
            _, buffer = cv2.imencode('.jpg', processed_image)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Get notifications
            notifications = notification_queue.copy()
            notification_queue = []
            
            # Get tracked vehicles for frontend
            vehicles_data = []
            if vehicle_tracker and vehicle_tracker.tracked_vehicles:
                for vehicle in vehicle_tracker.tracked_vehicles:
                    vehicles_data.append({
                        "id": vehicle.id,
                        "type": vehicle.type,
                        "x": vehicle.x,
                        "y": vehicle.y,
                        "width": vehicle.width,
                        "height": vehicle.height,
                        "confidence": vehicle.confidence,
                        "speed": vehicle.speed,
                        "is_violating": vehicle.is_violating,
                        "violation_type": vehicle.violation_type,
                        "color": vehicle.color
                    })
            
            # Send response
            emit('frame_processed', {
                'frame': f"data:image/jpeg;base64,{processed_image_base64}",
                'stats': stats,
                'vehicles': vehicles_data,
                'traffic_lights': [light.to_dict() for light in traffic_light_detector.traffic_lights] if traffic_light_detector else [],
                'notifications': notifications
            })
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        emit('error', {'message': str(e)})

def open_browser():
    """Open web browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Traffic Violation Detection System')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the web server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the web server on')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Initialize the detection system
    initialize_system()
    
    # Open browser automatically unless disabled
    if not args.no_browser:
        threading.Thread(target=open_browser).start()
    
    # Start the Flask server
    print(f"Starting server on http://{args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=False)
