import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
# Increase detection confidence to make hand tracking more stable
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)
mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)
prev_x = None
prev_y = None

# Create a 2D figure (a simple rectangle)
figure_x = 300
figure_y = 200
figure_width = 100
figure_height = 100
figure_color = (0, 255, 0)  # Green color

# Pinch detection variables
pinching = False
pinch_threshold = 0.06  # Slightly increased threshold for easier pinch detection
pinch_offset_x = 0
pinch_offset_y = 0
last_valid_hand_position = (0, 0)  # Store last valid hand position

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Mirror image
    frame = cv2.flip(frame, 1)
    
    # Display instructions
    cv2.putText(frame, "Pinch thumb & index over the green box to move it", 
               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", 
               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape
    
    # Draw the 2D figure on the frame
    cv2.rectangle(frame, (figure_x, figure_y), 
                 (figure_x + figure_width, figure_y + figure_height), 
                 figure_color, -1)
    
    # Add a border to make the figure more visible
    cv2.rectangle(frame, (figure_x, figure_y), 
                 (figure_x + figure_width, figure_y + figure_height), 
                 (0, 0, 0), 2)
    
    # Convert to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get handedness (Left or Right)
            handedness = results.multi_handedness[hand_idx].classification[0].label
            
            # Get thumb and index finger landmarks
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
            
            # Convert to pixel coordinates
            thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
            index_x, index_y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
            wrist_x, wrist_y = int(wrist.x * frame_width), int(wrist.y * frame_height)
            
            # Store hand position to avoid jumps
            last_valid_hand_position = (index_x, index_y)
            
            # Calculate Euclidean distance between thumb and index
            pinch_distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2) / frame_width
            
            # Draw a line between thumb and index finger
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 2)
            
            # Display the pinch distance
            cv2.putText(frame, f"Pinch: {pinch_distance:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Check if pinching
            is_pinching = pinch_distance < pinch_threshold
            
            # Check if finger is over the figure
            finger_over_figure = (figure_x <= index_x <= figure_x + figure_width and 
                                 figure_y <= index_y <= figure_y + figure_height)
            
            if handedness == "Left":  # Use left hand for figure control
                # Check if finger is over the figure
                finger_over_figure = (figure_x <= index_x <= figure_x + figure_width and 
                                    figure_y <= index_y <= figure_y + figure_height)
                
                # Debug information
                cv2.putText(frame, f"Over figure: {finger_over_figure}", (50, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Pinching: {pinching}", (50, 110), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Case 1: Start pinching when over figure
                if is_pinching and not pinching and finger_over_figure:
                    pinching = True
                    pinch_offset_x = index_x - figure_x
                    pinch_offset_y = index_y - figure_y
                    figure_color = (0, 0, 255)  # Red when pinched
                    cv2.putText(frame, "GRABBED!", (figure_x, figure_y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Case 2: Continue moving while pinching
                elif is_pinching and pinching:
                    # Move the figure with the hand
                    figure_x = index_x - pinch_offset_x
                    figure_y = index_y - pinch_offset_y
                    
                    # Keep figure within frame bounds
                    figure_x = max(0, min(figure_x, frame_width - figure_width))
                    figure_y = max(0, min(figure_y, frame_height - figure_height))
                    
                    cv2.putText(frame, "MOVING...", (figure_x, figure_y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Case 3: Release when not pinching anymore
                elif not is_pinching and pinching:
                    pinching = False
                    figure_color = (0, 255, 0)  # Green when not pinched
            
            elif handedness == "Right":  # right keyboard
                # Original keyboard functionality
                x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
                if prev_x is not None and prev_y is not None:
                    dx = x - prev_x
                    dy = y - prev_y

                    if abs(dx) > abs(dy):
                        if dx > 50:  # right
                            pyautogui.press('right')
                        elif dx < -50:  # left
                            pyautogui.press('left')
                    else:  # Vertical swipe
                        if dy > 50:  # down
                            pyautogui.press('down')
                        elif dy < -50:  # up
                            pyautogui.press('up')

                prev_x = x
                prev_y = y
    
    # Add text instructions
    cv2.putText(frame, "Pinch left hand over figure to move it", (10, frame_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    # Display the resulting frame
    cv2.imshow("Gesture Recognition with Movable Figure", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
