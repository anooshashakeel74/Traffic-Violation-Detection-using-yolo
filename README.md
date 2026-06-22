... shell ...

I've created a complete Python implementation of the Traffic Violation Detection system based on the GitHub repository you shared. This implementation includes:

1. **YOLO Integration**: The system uses YOLO for vehicle detection with automatic model downloading.

2. **DeepSORT Integration**: Vehicle tracking is implemented using DeepSORT with fallback to a simple tracker if DeepSORT is not available.

3. **Web Interface**: The same GUI from the original project has been recreated using Flask and SocketIO for real-time communication.

4. **Traffic Light Detection**: The system can detect traffic lights and their states.

5. **Violation Detection**: Both red light and speeding violations are detected and logged.

6. **Real-time Processing**: The system processes video frames in real-time and displays the results in the web interface.

### How to Run the System

1. Install the required packages:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Run the application:
\`\`\`bash
python main.py
\`\`\`

3. Open your web browser and navigate to `http://localhost:5000`

4. Upload a traffic video and start detection

The system will automatically download the required YOLO and DeepSORT models on first run.
