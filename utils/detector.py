"""
Vehicle detector module using YOLO
"""

import cv2
import numpy as np
import os
import urllib.request

class Vehicle:
    """Vehicle class to store detection information"""
    def __init__(self, id, type, x, y, width, height, confidence):
        self.id = id
        self.type = type
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.speed = 0
        self.is_violating = False
        self.violation_type = "none"
        self.color = self._get_color_for_type(type)
    
    def _get_color_for_type(self, type):
        """Get color based on vehicle type"""
        color_map = {
            "car": "#3B82F6",      # Blue
            "bus": "#10B981",      # Green
            "truck": "#EF4444",    # Red
            "motorcycle": "#8B5CF6", # Purple
            "bicycle": "#F59E0B"   # Yellow
        }
        return color_map.get(type, "#3B82F6")  # Default to blue

class VehicleDetector:
    """YOLO-based vehicle detector"""
    def __init__(self, model_path, config_path, weights_path, classes_path, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = 0.4
        
        # Load YOLO model
        self.model_path = model_path
        self.config_path = config_path
        self.weights_path = weights_path
        
        # Download required files if they don't exist
        self._download_yolo_files()
        
        # Load YOLO network
        try:
            self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
            
            # Use GPU if available
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using GPU for YOLO detection")
            except:
                print("GPU not available, using CPU for YOLO detection")
        except Exception as e:
            print(f"Error loading YOLO network: {e}")
            # Create a dummy network for testing
            self.net = None
        
        # Load classes
        self.classes = self._load_classes(classes_path)
        
        # Get output layer names
        if self.net:
            self.layer_names = self.net.getLayerNames()
            try:
                self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            except:
                self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Vehicle classes we're interested in
        self.vehicle_classes = {
            "car": 2,
            "motorcycle": 3,
            "bus": 5,
            "truck": 7,
            "bicycle": 1
        }
        
        # Reverse mapping
        self.vehicle_class_names = {v: k for k, v in self.vehicle_classes.items()}
        
        print(f"Vehicle detector initialized with confidence threshold: {self.confidence_threshold}")
    
    def _download_yolo_files(self):
        """Download YOLO configuration and weights files"""
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Download YOLOv3 config
        if not os.path.exists(self.config_path):
            print("Downloading YOLOv3 configuration...")
            try:
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
                    self.config_path
                )
            except Exception as e:
                print(f"Error downloading config: {e}")
                # Create a minimal config file
                self._create_minimal_config()
        
        # Download YOLOv3 weights
        if not os.path.exists(self.weights_path):
            print("Downloading YOLOv3 weights (this may take a while)...")
            try:
                urllib.request.urlretrieve(
                    "https://pjreddie.com/media/files/yolov3.weights",
                    self.weights_path
                )
            except Exception as e:
                print(f"Error downloading weights: {e}")
        
        # Create classes file
        classes_path = os.path.join("models", "coco.names")
        if not os.path.exists(classes_path):
            self._create_classes_file(classes_path)
    
    def _create_minimal_config(self):
        """Create a minimal YOLO config for testing"""
        config_content = """[net]
batch=1
subdivisions=1
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear

[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)
    
    def _create_classes_file(self, classes_path):
        """Create COCO classes file"""
        classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        
        with open(classes_path, 'w') as f:
            for class_name in classes:
                f.write(class_name + '\n')
    
    def _load_classes(self, classes_path):
        """Load class names"""
        try:
            with open(classes_path, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            return classes
        except:
            print("Error loading classes file. Using default classes.")
            return ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
    
    def detect(self, frame):
        """Detect vehicles in the frame using YOLO"""
        if self.net is None:
            # Return simulated detections for testing
            return self._simulate_detections(frame)
        
        height, width, _ = frame.shape
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Process outputs
        class_ids = []
        confidences = []
        boxes = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for vehicle classes and confidence threshold
                if class_id in self.vehicle_class_names and confidence > self.confidence_threshold:
                    # YOLO returns center, width, height
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Create vehicle objects
        vehicles = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                confidence = confidences[i]
                
                # Create vehicle object
                vehicle_type = self.vehicle_class_names.get(class_id, "unknown")
                vehicle = Vehicle(
                    id=i,  # Temporary ID, will be updated by tracker
                    type=vehicle_type,
                    x=max(0, x),
                    y=max(0, y),
                    width=w,
                    height=h,
                    confidence=confidence
                )
                
                vehicles.append(vehicle)
        
        return vehicles
    
    def _simulate_detections(self, frame):
        """Simulate vehicle detections for testing when YOLO is not available"""
        height, width = frame.shape[:2]
        vehicles = []
        
        # Create some simulated vehicles
        simulated_vehicles = [
            {"type": "car", "x": int(width * 0.3), "y": int(height * 0.6), "w": 80, "h": 60},
            {"type": "bus", "x": int(width * 0.7), "y": int(height * 0.4), "w": 120, "h": 80},
        ]
        
        for i, sim_vehicle in enumerate(simulated_vehicles):
            vehicle = Vehicle(
                id=i,
                type=sim_vehicle["type"],
                x=sim_vehicle["x"],
                y=sim_vehicle["y"],
                width=sim_vehicle["w"],
                height=sim_vehicle["h"],
                confidence=0.85
            )
            vehicles.append(vehicle)
        
        return vehicles
