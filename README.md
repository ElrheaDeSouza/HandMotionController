# HandMotionController

A computer vision project that enables users to interact with and manipulate 2D objects using hand gestures captured through a webcam.

## Features

- **Real-time Hand Tracking**: Utilizes MediaPipe to detect and track hand movements
- **Pinch Gesture Recognition**: Detects pinch gestures by measuring the distance between thumb and index finger
- **Object Manipulation**: Allows users to grab, move, and release 2D objects on screen
- **Visual Feedback**: Provides status indicators and debugging information during interaction
- **Dual Hand Functionality**: Left hand controls object manipulation while right hand enables keyboard navigation

## Requirements

- Python 3.6+
- OpenCV (cv2)
- MediaPipe
- PyAutoGUI
- NumPy

## Installation

1. Clone this repository:
   git clone https://github.com/ElrheaDeSouza/HandMotionController.git cd HandMotionController


2. Install required dependencies:

pip install opencv-python mediapipe pyautogui numpy


## Usage

1. Run the main script:

python gesture_figure.py


2. Position yourself in front of your webcam

3. Interact with the green rectangle on screen:
- Use your **left hand** to manipulate the object:
  - Position your hand over the green rectangle
  - Pinch your thumb and index finger together to grab it (turns red)
  - Move your hand while maintaining the pinch to relocate the object
  - Release the pinch to drop the object

- Use your **right hand** for keyboard navigation:
  - Swipe right → Right arrow key
  - Swipe left → Left arrow key
  - Swipe up → Up arrow key
  - Swipe down → Down arrow key

4. Press 'q' to quit the application

## How It Works

GestureFigure leverages computer vision and machine learning techniques to enable natural interaction with digital objects:

1. **Hand Detection**: MediaPipe's hand landmark detection identifies key points on the user's hands
2. **Gesture Recognition**: The distance between thumb and index finger is calculated to detect pinching gestures
3. **Object Manipulation**: When a pinch is detected over the object, the system calculates offset values to enable smooth movement tracking
4. **Visual Feedback**: The application provides real-time feedback through color changes and on-screen text

## Customization

### Modifying the 2D Object

You can customize the appearance of the 2D object by changing these parameters:
```python
# Create a 2D figure (modify these values)
figure_width = 100  # Width of the rectangle
figure_height = 100  # Height of the rectangle
figure_color = (0, 255, 0)  # Color in BGR format (Green)

Adjusting Sensitivity

For more precise control, adjust these parameters:

python

# Increase for easier pinch detection, decrease for more precision
pinch_threshold = 0.06

# Adjust hand detection confidence
hands = mp_hands.Hands(
    min_detection_confidence=0.7,  # Decrease for better detection in low light
    min_tracking_confidence=0.7    # Decrease if tracking seems unstable
)




