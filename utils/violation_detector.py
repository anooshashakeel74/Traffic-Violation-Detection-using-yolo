"""
Violation detection module
"""

import time

class Violation:
    """Violation class to store violation information"""
    def __init__(self, vehicle_id, vehicle_type, x, y, width, height, type, traffic_light_id=None, speed=0):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = type  # "redLight", "speeding", "both"
        self.traffic_light_id = traffic_light_id
        self.speed = speed
        self.timestamp = time.time()
        self.is_new = True
        self.is_active = True

class ViolationDetector:
    """Violation detector for red light and speeding violations"""
    def __init__(self, speed_threshold=50, red_light_sensitivity=0.8):
        self.speed_threshold = speed_threshold
        self.red_light_sensitivity = red_light_sensitivity
        self.active_violations = {}  # vehicle_id -> Violation
        self.violation_history = {}  # vehicle_id -> set of violation types
    
    def detect_violations(self, vehicles, traffic_lights, timestamp):
        """Detect violations for the given vehicles and traffic lights"""
        new_violations = []
        new_red_light_violations = 0
        new_speeding_violations = 0
        
        # Reset is_new flag for all active violations
        for violation in self.active_violations.values():
            violation.is_new = False
        
        # Check each vehicle for violations
        for vehicle in vehicles:
            # Get or initialize violation history for this vehicle
            if vehicle.id not in self.violation_history:
                self.violation_history[vehicle.id] = set()
            
            # Check for speeding violation
            is_speeding = vehicle.speed > self.speed_threshold
            
            if is_speeding and "speeding" not in self.violation_history[vehicle.id]:
                # New speeding violation
                self.violation_history[vehicle.id].add("speeding")
                
                # Create violation object
                violation = Violation(
                    vehicle_id=vehicle.id,
                    vehicle_type=vehicle.type,
                    x=vehicle.x,
                    y=vehicle.y,
                    width=vehicle.width,
                    height=vehicle.height,
                    type="speeding",
                    speed=vehicle.speed
                )
                
                # Add to active violations
                violation_key = f"{vehicle.id}_speeding"
                self.active_violations[violation_key] = violation
                new_violations.append(violation)
                new_speeding_violations += 1
            
            # Check for red light violation
            for light in traffic_lights:
                if light.state == "red":
                    # Get vehicle position relative to stop line
                    vehicle_bottom = vehicle.y + vehicle.height
                    vehicle_top = vehicle.y
                    
                    # Check if vehicle is crossing or has crossed the stop line
                    is_crossing = vehicle_bottom >= light.stop_line_y and vehicle_top <= light.stop_line_y
                    has_crossed = vehicle_top > light.stop_line_y
                    
                    violation_key = f"{vehicle.id}_redLight_{light.id}"
                    
                    # If vehicle is crossing or has crossed the stop line during red light, mark as violation
                    if (is_crossing or has_crossed) and violation_key not in self.active_violations:
                        # Check if this is a new violation
                        if violation_key not in self.violation_history[vehicle.id]:
                            self.violation_history[vehicle.id].add(violation_key)
                            
                            # Create violation object
                            violation = Violation(
                                vehicle_id=vehicle.id,
                                vehicle_type=vehicle.type,
                                x=vehicle.x,
                                y=vehicle.y,
                                width=vehicle.width,
                                height=vehicle.height,
                                type="redLight",
                                traffic_light_id=light.id
                            )
                            
                            # Add to active violations
                            self.active_violations[violation_key] = violation
                            new_violations.append(violation)
                            new_red_light_violations += 1
            
            # Update vehicle violation status
            has_red_light_violation = any(k.startswith(f"{vehicle.id}_redLight") for k in self.active_violations)
            
            if is_speeding and has_red_light_violation:
                vehicle.violation_type = "both"
                vehicle.is_violating = True
            elif is_speeding:
                vehicle.violation_type = "speeding"
                vehicle.is_violating = True
            elif has_red_light_violation:
                vehicle.violation_type = "redLight"
                vehicle.is_violating = True
            else:
                vehicle.violation_type = "none"
                vehicle.is_violating = False
        
        # Update active violations with current vehicle positions
        for vehicle in vehicles:
            for key, violation in list(self.active_violations.items()):
                if str(vehicle.id) in key:
                    violation.x = vehicle.x
                    violation.y = vehicle.y
                    violation.width = vehicle.width
                    violation.height = vehicle.height
                    
                    # Update speed for speeding violations
                    if violation.type == "speeding":
                        violation.speed = vehicle.speed
        
        # Remove old violations (keep for 5 seconds)
        current_time = time.time()
        for key, violation in list(self.active_violations.items()):
            if current_time - violation.timestamp > 5:
                violation.is_active = False
            
            if current_time - violation.timestamp > 10:
                del self.active_violations[key]
        
        # Return all active violations
        return list(self.active_violations.values()), new_red_light_violations, new_speeding_violations
