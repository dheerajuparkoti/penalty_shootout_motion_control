# import cv2
# import mediapipe as mp
#
# # Initialize Mediapipe pose detection
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# # Open webcam
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Flip the frame for a mirror view
#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process the frame
#     results = pose.process(rgb_frame)
#
#     # Draw landmarks and display detected points
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#         )
#
#         # Display the name of each detected body part
#         for idx, landmark in enumerate(results.pose_landmarks.landmark):
#             x = int(landmark.x * frame.shape[1])
#             y = int(landmark.y * frame.shape[0])
#
#             # Body part names based on mediapipe Pose Landmarks index
#             body_parts = mp_pose.PoseLandmark
#             body_part_name = body_parts(idx).name if idx in body_parts.__members__.values() else f"Part {idx}"
#
#             cv2.putText(frame, body_part_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.4, (0, 255, 0), 1, cv2.LINE_AA)
#
#     # Display the frame
#     cv2.imshow('Pose Detection with Labels', frame)
#
#     # Exit with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# LEFT OR RIGHT ANKLE DETECTION
# import cv2
# import mediapipe as mp
# import pyautogui
#
# # Initialize Mediapipe pose detection
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# # Open webcam
# cap = cv2.VideoCapture(0)
#
# # Store previous X position for ankle detection
# prev_left_ankle_x = None
# prev_right_ankle_x = None
# movement_threshold = 0.05  # Threshold for detecting significant movement (change in X)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Flip the frame for a mirror view
#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process the frame
#     results = pose.process(rgb_frame)
#
#     # Draw landmarks and detect leg movement
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#         )
#
#         # Get required landmarks for left and right ankle
#         left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
#         right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
#
#         # Check if the left and right ankles are detected (visible landmarks)
#         if left_ankle.visibility > 0.5 and right_ankle.visibility > 0.5:  # Ensures ankle visibility
#             # Detect movement based on X-coordinate of left ankle
#             if prev_left_ankle_x is not None:
#                 if left_ankle.x < prev_left_ankle_x - movement_threshold:  # Left leg moves left (threshold check)
#                     print("Left leg moved left!")
#                     pyautogui.write('L')  # Simulate typing 'L'
#                 elif left_ankle.x > prev_left_ankle_x + movement_threshold:  # Left leg moves right (threshold check)
#                     print("Left leg moved right!")
#                     pyautogui.write('R')  # Simulate typing 'R'
#
#             # Update previous left ankle position
#             prev_left_ankle_x = left_ankle.x
#
#             # Detect movement based on X-coordinate of right ankle
#             if prev_right_ankle_x is not None:
#                 if right_ankle.x < prev_right_ankle_x - movement_threshold:  # Right leg moves left (threshold check)
#                     print("Right leg moved left!")
#                     pyautogui.write('L')  # Simulate typing 'L'
#                 elif right_ankle.x > prev_right_ankle_x + movement_threshold:  # Right leg moves right (threshold check)
#                     print("Right leg moved right!")
#                     pyautogui.write('R')  # Simulate typing 'R'
#
#             # Update previous right ankle position
#             prev_right_ankle_x = right_ankle.x
#
#     # Display the frame
#     cv2.imshow('Pose Detection with Labels', frame)
#
#     # Exit with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# LEFT MOUSE BUTTON HOLD WHEN LEFT ANKLE OR RIGHT ANKLE MOVED TO LEFT OR RIGHT AND RELEASE WHEN COME BACK TO ITS NEUTRAL POSITION
# import cv2
# import mediapipe as mp
# import pyautogui
# import time
#
# # Initialize Mediapipe pose detection
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# # Open webcam
# cap = cv2.VideoCapture(0)
#
# # Store previous X position for ankle detection
# prev_left_ankle_x = None
# prev_right_ankle_x = None
# movement_threshold = 0.05  # Threshold for detecting significant movement (change in X)
# clicking = False  # Flag to track if the mouse button is being held
#
# # Define neutral ankle position (middle position where the ankle should be)
# neutral_position_threshold = 0.02  # Allowable variation from the neutral position
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Flip the frame for a mirror view
#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process the frame
#     results = pose.process(rgb_frame)
#
#     # Draw landmarks and detect leg movement
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#         )
#
#         # Get required landmarks for left and right ankle
#         left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
#         right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
#
#         # Check if the left and right ankles are detected (visible landmarks)
#         if left_ankle.visibility > 0.5 and right_ankle.visibility > 0.5:  # Ensures ankle visibility
#             # Detect movement based on X-coordinate of left ankle
#             if prev_left_ankle_x is not None:
#                 if left_ankle.x < prev_left_ankle_x - movement_threshold:  # Left leg moves left (threshold check)
#                     if not clicking:
#                         pyautogui.mouseDown()  # Simulate holding the left mouse button
#                         clicking = True
#                         print("Left click hold")  # Print "Left click hold" in the console
#                 elif left_ankle.x > prev_left_ankle_x + movement_threshold:  # Left leg moves right (threshold check)
#                     if not clicking:
#                         pyautogui.mouseDown()  # Simulate holding the left mouse button
#                         clicking = True
#                         print("Left click hold")  # Print "Left click hold" in the console
#
#             # Detect if ankle has returned to neutral position (ankle centered)
#             if prev_left_ankle_x is not None and abs(left_ankle.x - prev_left_ankle_x) < neutral_position_threshold:
#                 if clicking:
#                     pyautogui.mouseUp()  # Release the mouse button
#                     clicking = False
#                     print("Left click released")  # Print "Left click released" in the console
#
#             # Update previous left ankle position
#             prev_left_ankle_x = left_ankle.x
#
#             # Detect movement based on X-coordinate of right ankle
#             if prev_right_ankle_x is not None:
#                 if right_ankle.x < prev_right_ankle_x - movement_threshold:  # Right leg moves left (threshold check)
#                     if not clicking:
#                         pyautogui.mouseDown()  # Simulate holding the left mouse button
#                         clicking = True
#                         print("Left click hold")  # Print "Left click hold" in the console
#                 elif right_ankle.x > prev_right_ankle_x + movement_threshold:  # Right leg moves right (threshold check)
#                     if not clicking:
#                         pyautogui.mouseDown()  # Simulate holding the left mouse button
#                         clicking = True
#                         print("Left click hold")  # Print "Left click hold" in the console
#
#             # Detect if ankle has returned to neutral position (ankle centered)
#             if prev_right_ankle_x is not None and abs(right_ankle.x - prev_right_ankle_x) < neutral_position_threshold:
#                 if clicking:
#                     pyautogui.mouseUp()  # Release the mouse button
#                     clicking = False
#                     print("Left click released")  # Print "Left click released" in the console
#
#             # Update previous right ankle position
#             prev_right_ankle_x = right_ankle.x
#
#     # Display the frame
#     cv2.imshow('Pose Detection with Labels', frame)
#
#     # Exit with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# RELEASED ONLY HAND RAISED
# import cv2
# import mediapipe as mp
# import pyautogui
# import time
#
# # Initialize Mediapipe pose detection
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# # Open webcam
# cap = cv2.VideoCapture(0)
#
# # Store previous X position for ankle detection
# prev_left_ankle_x = None
# prev_right_ankle_x = None
# movement_threshold = 0.05  # Threshold for detecting significant horizontal movement (change in X)
# clicking = False  # Flag to track if the mouse button is being held
#
# # Define neutral ankle position (middle position where the ankle should be)
# neutral_position_threshold = 0.02  # Allowable variation from the neutral position
#
# # Define thresholds for detecting if the hand is raised
# hand_raised_threshold = 0.1  # Y-coordinate threshold to detect hand raised
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#
#     # Flip the frame for a mirror view
#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process the frame
#     results = pose.process(rgb_frame)
#
#     # Draw landmarks and detect leg movement
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#         )
#
#         # Get required landmarks for left and right ankle
#         left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
#         right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
#
#         # Get required landmarks for left and right wrist to detect raised hand
#         left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
#         left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
#         right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#         right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
#
#         # Check if the left and right ankles are detected (visible landmarks)
#         if left_ankle.visibility > 0.5 and right_ankle.visibility > 0.5:  # Ensures ankle visibility
#             # Detect movement based on X-coordinate of left ankle (horizontal movement)
#             if prev_left_ankle_x is not None:
#                 # Check if the left leg moves left or right beyond threshold
#                 if left_ankle.x < prev_left_ankle_x - movement_threshold:  # Left leg moves left (threshold check)
#                     if not clicking:
#                         pyautogui.mouseDown()  # Simulate holding the left mouse button
#                         clicking = True
#                         print("Left click hold")  # Print "Left click hold" in the console
#                 elif left_ankle.x > prev_left_ankle_x + movement_threshold:  # Left leg moves right (threshold check)
#                     if not clicking:
#                         pyautogui.mouseDown()  # Simulate holding the left mouse button
#                         clicking = True
#                         print("Left click hold")  # Print "Left click hold" in the console
#
#             # Update previous left ankle position
#             prev_left_ankle_x = left_ankle.x
#
#             # Detect movement based on X-coordinate of right ankle (horizontal movement)
#             if prev_right_ankle_x is not None:
#                 # Check if the right leg moves left or right beyond threshold
#                 if right_ankle.x < prev_right_ankle_x - movement_threshold:  # Right leg moves left (threshold check)
#                     if not clicking:
#                         pyautogui.mouseDown()  # Simulate holding the left mouse button
#                         clicking = True
#                         print("Left click hold")  # Print "Left click hold" in the console
#                 elif right_ankle.x > prev_right_ankle_x + movement_threshold:  # Right leg moves right (threshold check)
#                     if not clicking:
#                         pyautogui.mouseDown()  # Simulate holding the left mouse button
#                         clicking = True
#                         print("Left click hold")  # Print "Left click hold" in the console
#
#             # Update previous right ankle position
#             prev_right_ankle_x = right_ankle.x
#
#             # Only release the click when the hand is raised (based on wrist position relative to shoulder)
#             if (left_wrist.y < left_shoulder.y - hand_raised_threshold) or (right_wrist.y < right_shoulder.y - hand_raised_threshold):
#                 if clicking:
#                     pyautogui.mouseUp()  # Release the mouse button
#                     clicking = False
#                     print("Left click released (hand raised)")  # Print "Left click released" in the console
#
#     # Display the frame
#     cv2.imshow('Pose Detection with Labels', frame)
#
#     # Exit with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()



#THIS CODE WORKS FOR PENALTY
# import cv2
# import mediapipe as mp
# import pyautogui
# import time
# from playsound import playsound
#
# # Initialize Mediapipe pose detection
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# # Open webcam
# cap = cv2.VideoCapture(0)
#
# # Store previous X positions for ankle detection
# prev_left_ankle_x = None
# prev_right_ankle_x = None
# movement_threshold = 0.05  # Threshold for detecting significant horizontal movement (change in X)
# clicking = False  # Flag to track if the mouse button is being held
#
# # Define neutral ankle position (middle position where the ankle should be)
# neutral_position_threshold = 0.02  # Allowable variation from the neutral position
#
# # Set up the frame size (for 640x480 resolution)
# frame_width = 640
# frame_height = 480
#
# # Define the section lines for the left, middle, and right sections
# left_section_line = frame_width // 3  # Left section at 1/3 of the width
# right_section_line = 2 * frame_width // 3  # Right section at 2/3 of the width
#
# # Center the mouse cursor immediately on the whole screen when the program starts
# screen_width, screen_height = pyautogui.size()  # Get the screen resolution
# pyautogui.moveTo(screen_width // 2, screen_height // 2)  # Move cursor to the center of the screen
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#
#     # Resize the frame for consistent section sizes
#     frame = cv2.resize(frame, (frame_width, frame_height))
#
#     # Flip the frame for a mirror view
#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process the frame
#     results = pose.process(rgb_frame)
#
#     # Draw landmarks and detect leg movement
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#         )
#
#         # Get required landmarks for the ankles
#         left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
#         right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
#
#         # Check if the left ankle is detected (visible landmark)
#         if left_ankle.visibility > 0.5:  # Ensures ankle visibility
#             # Detect movement based on X-coordinate of left ankle (horizontal movement)
#             if prev_left_ankle_x is not None:
#                 # Check if the left leg crosses the section lines
#                 if left_ankle.x < left_section_line / frame_width:  # Left ankle in the left section
#                     if not clicking:
#                         pyautogui.mouseDown()  # Simulate holding the left mouse button
#                         clicking = True
#                         print("Left click hold - Left section")  # Print message
#                 elif left_ankle.x > right_section_line / frame_width:  # Left ankle in the right section
#                     if not clicking:
#                         pyautogui.mouseDown()  # Simulate holding the left mouse button
#                         clicking = True
#                         print("Left click hold - Right section")  # Print message
#
#         # Check if the right ankle is detected (visible landmark)
#         if right_ankle.visibility > 0.5:  # Ensures ankle visibility
#             # Detect movement based on X-coordinate of right ankle (horizontal movement)
#             if prev_right_ankle_x is not None:
#                 # Check if the right leg crosses the section lines
#                 if right_ankle.x < left_section_line / frame_width:  # Right ankle in the left section
#                     if not clicking:
#                         pyautogui.mouseDown()  # Simulate holding the left mouse button
#                         clicking = True
#                         print("Left click hold - Left section (Right ankle)")  # Print message
#                 elif right_ankle.x > right_section_line / frame_width:  # Right ankle in the right section
#                     if not clicking:
#                         pyautogui.mouseDown()  # Simulate holding the left mouse button
#                         clicking = True
#                         print("Left click hold - Right section (Right ankle)")  # Print message
#
#         # Detect if the ankles are no longer on any section line (release condition)
#         if prev_left_ankle_x is not None and prev_right_ankle_x is not None:
#             if (left_ankle.x > left_section_line / frame_width and left_ankle.x < right_section_line / frame_width) and \
#                (right_ankle.x > left_section_line / frame_width and right_ankle.x < right_section_line / frame_width):
#                 if clicking:
#                     pyautogui.mouseUp()  # Release the mouse button
#                     clicking = False
#                     print("Left click released - Neutral section")  # Print message
#                     playsound("sounds\\whistle.mp3")  # Play a whistle sound (make sure the path is correct)
#                     # Center the mouse cursor immediately on the whole screen when the program starts
#                     pyautogui.moveTo(screen_width // 2, screen_height // 2)  # Move cursor to the center of the screen
#
#         # Update previous ankle positions
#         prev_left_ankle_x = left_ankle.x
#         prev_right_ankle_x = right_ankle.x
#
#     # Draw the section lines on the frame (for visual reference)
#     cv2.line(frame, (left_section_line, 0), (left_section_line, frame_height), (0, 255, 0), 2)  # Left section line
#     cv2.line(frame, (right_section_line, 0), (right_section_line, frame_height), (0, 255, 0), 2)  # Right section line
#
#     # Display the frame
#     cv2.imshow('Pose Detection with Sections', frame)
#
#     # Exit with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

#
# THIS CODE WORKS FOR GOALKEEPER
# import cv2
# import mediapipe as mp
# import pyautogui
# import numpy as np
#
# # Initialize MediaPipe Hands module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
#
# # Initialize MediaPipe drawing module for visualizing hand landmarks
# mp_drawing = mp.solutions.drawing_utils
#
# # Initialize camera
# cap = cv2.VideoCapture(0)
#
# # Get screen dimensions
# screen_width, screen_height = pyautogui.size()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Flip the frame for better mirror effect
#     frame = cv2.flip(frame, 1)
#
#     # Convert the BGR frame to RGB as MediaPipe uses RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process the frame and get hand landmarks
#     results = hands.process(rgb_frame)
#
#     # If hands are detected
#     if results.multi_hand_landmarks:
#         closest_hand = None
#         closest_hand_distance = float('inf')  # Start with an infinitely large value
#
#         # Iterate through the detected hands
#         for landmarks in results.multi_hand_landmarks:
#             # Get the wrist position (or any other key points)
#             wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
#             # Calculate the "depth" (z-coordinate) for the wrist or another key point
#             distance_to_camera = wrist.z
#
#             # If this hand is closer, update the closest hand
#             if distance_to_camera < closest_hand_distance:
#                 closest_hand_distance = distance_to_camera
#                 closest_hand = landmarks
#
#         # If  closest hand was found
#         if closest_hand:
#             # Draw the hand landmarks on the frame
#             mp_drawing.draw_landmarks(frame, closest_hand, mp_hands.HAND_CONNECTIONS)
#
#             # Get the wrist position
#             wrist = closest_hand.landmark[mp_hands.HandLandmark.WRIST]
#             thumb = closest_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
#             index = closest_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#
#             # Map the wrist position to the screen
#             wrist_x = int(wrist.x * frame.shape[1])
#             wrist_y = int(wrist.y * frame.shape[0])
#
#             # Map wrist position to screen coordinates
#             mouse_x = np.interp(wrist_x, [0, frame.shape[1]], [0, screen_width])
#             mouse_y = np.interp(wrist_y, [0, frame.shape[0]], [0, screen_height])
#
#             # Move the mouse to the wrist position
#             pyautogui.moveTo(mouse_x, mouse_y)
#
#             # Optionally, track other points (like the index finger) to create more control gestures
#             # Example: Clicking the mouse when the thumb and index are close together
#             distance = np.linalg.norm(np.array([thumb.x - index.x, thumb.y - index.y]))
#             if distance < 0.05:  # Threshold for clicking
#                 pyautogui.click()
#                 print("Mouse clicked!")
#
#                 # Center the cursor after click
#                 pyautogui.moveTo(screen_width // 2, screen_height // 2)
#                 print("Cursor moved to center.")
#
#     # Display the resulting frame
#     cv2.imshow("Hand Detection", frame)
#
#     # Exit on 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import pyautogui
# import numpy as np
# from playsound import playsound
#
# # Initialize MediaPipe Pose for Penalty Mode (Legs)
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# # Initialize MediaPipe Hands for Goalkeeper Mode (Hands)
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
#
# # Open webcam
# cap = cv2.VideoCapture(0)
#
# # Mode flag: 0 for Penalty, 1 for Goalkeeper
# mode = 0
#
# # Store previous X positions for ankle detection (for Penalty mode)
# prev_left_ankle_x = None
# prev_right_ankle_x = None
# movement_threshold = 0.05  # Threshold for detecting significant horizontal movement (change in X)
# clicking = False  # Flag to track if the mouse button is being held
#
# # Define neutral ankle position (for Penalty mode)
# neutral_position_threshold = 0.02  # Allowable variation from the neutral position
#
# # Set up the frame size
# frame_width = 640
# frame_height = 480
#
# # Define the section lines for Penalty mode
# left_section_line = frame_width // 3
# right_section_line = 2 * frame_width // 3
#
# # Center the mouse cursor
# screen_width, screen_height = pyautogui.size()
# pyautogui.moveTo(screen_width // 2, screen_height // 2)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#
#     # Resize the frame for consistent section sizes
#     frame = cv2.resize(frame, (frame_width, frame_height))
#
#     # Flip the frame for better mirror effect
#     frame = cv2.flip(frame, 1)
#
#     if mode == 0:  # Penalty Mode - Leg detection
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(rgb_frame)
#
#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                       landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#
#             # Get required landmarks for the ankles (Leg detection)
#             left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
#             right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
#
#             # Left ankle detection
#             if left_ankle.visibility > 0.5:
#                 if prev_left_ankle_x is not None:
#                     if left_ankle.x < left_section_line / frame_width:
#                         if not clicking:
#                             pyautogui.mouseDown()
#                             clicking = True
#                             print("Left click hold - Left section")
#                     elif left_ankle.x > right_section_line / frame_width:
#                         if not clicking:
#                             pyautogui.mouseDown()
#                             clicking = True
#                             print("Left click hold - Right section")
#
#             # Right ankle detection
#             if right_ankle.visibility > 0.5:
#                 if prev_right_ankle_x is not None:
#                     if right_ankle.x < left_section_line / frame_width:
#                         if not clicking:
#                             pyautogui.mouseDown()
#                             clicking = True
#                             print("Left click hold - Left section (Right ankle)")
#                     elif right_ankle.x > right_section_line / frame_width:
#                         if not clicking:
#                             pyautogui.mouseDown()
#                             clicking = True
#                             print("Left click hold - Right section (Right ankle)")
#
#             # Release condition
#             if prev_left_ankle_x is not None and prev_right_ankle_x is not None:
#                 if (left_ankle.x > left_section_line / frame_width and left_ankle.x < right_section_line / frame_width) and \
#                    (right_ankle.x > left_section_line / frame_width and right_ankle.x < right_section_line / frame_width):
#                     if clicking:
#                         pyautogui.mouseUp()
#                         clicking = False
#                         print("Left click released - Neutral section")
#                         # playsound("sounds\\whistle.mp3")
#                         pyautogui.moveTo(screen_width // 2, screen_height // 2)
#
#             prev_left_ankle_x = left_ankle.x
#             prev_right_ankle_x = right_ankle.x
#
#         # Draw the section lines on the frame (visual reference)
#         cv2.line(frame, (left_section_line, 0), (left_section_line, frame_height), (0, 255, 0), 2)
#         cv2.line(frame, (right_section_line, 0), (right_section_line, frame_height), (0, 255, 0), 2)
#
#     elif mode == 1:  # Goalkeeper Mode - Hand detection
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(rgb_frame)
#
#         if results.multi_hand_landmarks:
#             closest_hand = None
#             closest_hand_distance = float('inf')
#
#             for landmarks in results.multi_hand_landmarks:
#                 wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
#                 distance_to_camera = wrist.z
#
#                 if distance_to_camera < closest_hand_distance:
#                     closest_hand_distance = distance_to_camera
#                     closest_hand = landmarks
#
#             if closest_hand:
#                 mp_drawing.draw_landmarks(frame, closest_hand, mp_hands.HAND_CONNECTIONS)
#
#                 wrist = closest_hand.landmark[mp_hands.HandLandmark.WRIST]
#                 thumb = closest_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
#                 index = closest_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#
#                 wrist_x = int(wrist.x * frame.shape[1])
#                 wrist_y = int(wrist.y * frame.shape[0])
#
#                 mouse_x = np.interp(wrist_x, [0, frame.shape[1]], [0, screen_width])
#                 mouse_y = np.interp(wrist_y, [0, frame.shape[0]], [0, screen_height])
#
#                 pyautogui.moveTo(mouse_x, mouse_y)
#
#                 distance = np.linalg.norm(np.array([thumb.x - index.x, thumb.y - index.y]))
#                 if distance < 0.05:
#                     pyautogui.click()
#                     print("Mouse clicked!")
#                     pyautogui.moveTo(screen_width // 2, screen_height // 2)
#                     print("Cursor moved to center.")
#
#     # Display the resulting frame
#     cv2.imshow("Pose/Hand Detection", frame)
#
#     # Switch modes when 'm' is pressed
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('m'):
#         mode = 1 - mode  # Toggle between 0 and 1
#
#     # Exit on 'q'
#     if key == ord('q'):
#         break
#
# # Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()






















#
# # testing mode
# import cv2
# import mediapipe as mp
# import pyautogui
# import numpy as np
# from playsound import playsound
# import time
#
# # Initialize MediaPipe Pose for Penalty Mode (Legs)
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# # Initialize MediaPipe Hands for Goalkeeper Mode (Hands)
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
#
# # Open webcam
# cap = cv2.VideoCapture(0)
#
# # Mode flag: 0 for Penalty, 1 for Goalkeeper
# mode = 0
#
# # Store previous X positions for ankle detection (for Penalty mode)
# prev_left_ankle_x = None
# prev_right_ankle_x = None
# clicking = False  # Flag to track if the mouse button is being held
#
# # Define neutral ankle position (for Penalty mode)
# neutral_position_threshold = 0.02  # Allowable variation from the neutral position
#
# # Set up the frame size
# frame_width = 640
# frame_height = 480
#
# # Define the section lines for Penalty mode
# left_section_line = frame_width // 3
# right_section_line = 2 * frame_width // 3
#
# # Center the mouse cursor
# screen_width, screen_height = pyautogui.size()
# pyautogui.moveTo(screen_width // 2, screen_height // 2)
#
# # Mode switching timeout in seconds
# mode_switch_timer = time.time()
# mode_switch_timeout = 10  # Timeout in seconds (after 10 seconds of inactivity in the current mode, switch mode)
#
# # Variables to track if a significant action occurred (leg or hand gesture)
# leg_action_detected = False
# hand_action_detected = False
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#
#     # Resize the frame for consistent section sizes
#     frame = cv2.resize(frame, (frame_width, frame_height))
#
#     # Flip the frame for better mirror effect
#     frame = cv2.flip(frame, 1)
#
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Mode-switching logic based on landmarks visibility and conditions
#     if mode == 0:  # Penalty Mode - Leg detection
#         results = pose.process(rgb_frame)
#
#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                       landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#
#             # Get required landmarks for the ankles (Leg detection)
#             left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
#             right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
#
#             # Left ankle detection
#             if left_ankle.visibility > 0.5:
#                 if prev_left_ankle_x is not None:
#                     if left_ankle.x < left_section_line / frame_width:
#                         if not clicking:
#                             pyautogui.mouseDown()
#                             clicking = True
#                             print("Left click hold - Left section")
#                             leg_action_detected = True
#                     elif left_ankle.x > right_section_line / frame_width:
#                         if not clicking:
#                             pyautogui.mouseDown()
#                             clicking = True
#                             print("Left click hold - Right section")
#                             leg_action_detected = True
#
#             # Right ankle detection
#             if right_ankle.visibility > 0.5:
#                 if prev_right_ankle_x is not None:
#                     if right_ankle.x < left_section_line / frame_width:
#                         if not clicking:
#                             pyautogui.mouseDown()
#                             clicking = True
#                             print("Left click hold - Left section (Right ankle)")
#                             leg_action_detected = True
#                     elif right_ankle.x > right_section_line / frame_width:
#                         if not clicking:
#                             pyautogui.mouseDown()
#                             clicking = True
#                             print("Left click hold - Right section (Right ankle)")
#                             leg_action_detected = True
#
#             # Release condition
#             if prev_left_ankle_x is not None and prev_right_ankle_x is not None:
#                 # if (left_ankle.x > left_section_line / frame_width and left_ankle.x < right_section_line / frame_width) and \
#                 #    (right_ankle.x > left_section_line / frame_width and right_ankle.x < right_section_line / frame_width):
#                 if (left_section_line / frame_width < left_ankle.x < right_section_line / frame_width) and \
#                    (left_section_line / frame_width < right_ankle.x < right_section_line / frame_width):
#                     if clicking:
#                         pyautogui.mouseUp()
#                         clicking = False
#                         print("Left click released - Neutral section")
#                         # playsound("sounds\\whistle.mp3")
#                         pyautogui.moveTo(screen_width // 2, screen_height // 2)
#
#             prev_left_ankle_x = left_ankle.x
#             prev_right_ankle_x = right_ankle.x
#
#         # Draw the section lines on the frame (visual reference)
#         cv2.line(frame, (left_section_line, 0), (left_section_line, frame_height), (0, 255, 0), 2)
#         cv2.line(frame, (right_section_line, 0), (right_section_line, frame_height), (0, 255, 0), 2)
#
#         # Automatically switch to Goalkeeper Mode after leg action detected for a while
#         if leg_action_detected:
#             if time.time() - mode_switch_timer > mode_switch_timeout:
#                 print("Switching to Goalkeeper Mode (Leg action detected)")
#                 mode = 1  # Switch to Goalkeeper mode
#                 mode_switch_timer = time.time()  # Reset the timer
#                 leg_action_detected = False  # Reset leg action detection
#
#     elif mode == 1:  # Goalkeeper Mode - Hand detection
#         results = hands.process(rgb_frame)
#
#         if results.multi_hand_landmarks:
#             closest_hand = None
#             closest_hand_distance = float('inf')
#
#             for landmarks in results.multi_hand_landmarks:
#                 wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
#                 distance_to_camera = wrist.z
#
#                 if distance_to_camera < closest_hand_distance:
#                     closest_hand_distance = distance_to_camera
#                     closest_hand = landmarks
#
#             if closest_hand:
#                 mp_drawing.draw_landmarks(frame, closest_hand, mp_hands.HAND_CONNECTIONS)
#
#                 wrist = closest_hand.landmark[mp_hands.HandLandmark.WRIST]
#                 thumb = closest_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
#                 index = closest_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#
#                 wrist_x = int(wrist.x * frame.shape[1])
#                 wrist_y = int(wrist.y * frame.shape[0])
#
#                 mouse_x = np.interp(wrist_x, [0, frame.shape[1]], [0, screen_width])
#                 mouse_y = np.interp(wrist_y, [0, frame.shape[0]], [0, screen_height])
#
#                 pyautogui.moveTo(mouse_x, mouse_y)
#
#                 distance = np.linalg.norm(np.array([thumb.x - index.x, thumb.y - index.y]))
#                 if distance < 0.05:
#                     pyautogui.click()
#                     print("Mouse clicked!")
#                     pyautogui.moveTo(screen_width // 2, screen_height // 2)
#                     print("Cursor moved to center.")
#                     hand_action_detected = True
#
#         # Automatically switch to Penalty Mode after hand action detected for a while
#         if hand_action_detected:
#             if time.time() - mode_switch_timer > mode_switch_timeout:
#                 print("Switching to Penalty Mode (Hand action detected)")
#                 mode = 0  # Switch to Penalty mode
#                 mode_switch_timer = time.time()  # Reset the timer
#                 hand_action_detected = False  # Reset hand action detection
#
#     # Display the resulting frame
#     cv2.imshow("Pose/Hand Detection", frame)
#
#     # Optional mode toggle via key press ('m' key)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('m'):  # Toggle between modes when 'm' is pressed
#         mode = 0 if mode == 1 else 1
#         print(f"Switched to {'Penalty Mode' if mode == 0 else 'Goalkeeper Mode'}")
#
#     # Exit on 'q'
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#
#
#
# cap.release()
# cv2.destroyAllWindows()




# MANUAL MODE
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from playsound import playsound
import time
import keyboard  # Library for global keypress detection
import pygame


# Initialize MediaPipe Pose for Penalty Mode (Legs)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands for Goalkeeper Mode (Hands)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

# Mode flag: 0 for Penalty, 1 for Goalkeeper, -1 for Paused
mode = 0
paused = False  # Paused state

# Store previous X positions for ankle detection (for Penalty mode)
prev_left_ankle_x = None
prev_right_ankle_x = None
clicking = False  # Flag to track if the mouse button is being held

# Set up the frame size
frame_width = 640
frame_height = 480

# Define the section lines for Penalty mode
left_section_line = frame_width // 3
right_section_line = 2 * frame_width // 3

# Center the mouse cursor
screen_width, screen_height = pyautogui.size()
pyautogui.moveTo(screen_width // 2, screen_height // 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame for consistent section sizes
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Flip the frame for better mirror effect
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Check for global key presses to control modes
    if keyboard.is_pressed('r'):  # Reset and start with Penalty Mode
        mode = 0  # Penalty Mode
        paused = False
        print("Reset to Penalty Mode")
        time.sleep(0.5)  # Add delay to prevent rapid switching
        # playsound("sounds\\ready.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load("sounds\\ready.mp3")
        pygame.mixer.music.play()

    elif keyboard.is_pressed('s'):  # Pause the system
        paused = True
        print("System Paused")
        time.sleep(0.5)  # Add delay to prevent rapid switching
        pygame.mixer.init()
        pygame.mixer.music.load("sounds/stop.mp3")
        pygame.mixer.music.play()

    elif keyboard.is_pressed('g') and not paused:  # Switch to Goalkeeper Mode
        mode = 1
        print("Switched to Goalkeeper Mode")
        time.sleep(0.5)
        playsound("sounds\\whistle.mp3")

    elif keyboard.is_pressed('p') and not paused:  # Switch to Penalty Mode
        mode = 0
        print("Switched to Penalty Mode")
        time.sleep(0.5)
        playsound("sounds\\whistle.mp3")

    if paused:  # Skip detection if paused
        cv2.putText(frame, "PAUSED", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow("Pose/Hand Detection", frame)
        continue

    if mode == 0:  # Penalty Mode - Leg detection
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Get required landmarks for the ankles (Leg detection)
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Left ankle detection
            if left_ankle.visibility > 0.5:
                if prev_left_ankle_x is not None:
                    if left_ankle.x < left_section_line / frame_width:
                        if not clicking:
                            pyautogui.mouseDown()
                            clicking = True
                            print("Left click hold - Left section")
                    elif left_ankle.x > right_section_line / frame_width:
                        if not clicking:
                            pyautogui.mouseDown()
                            clicking = True
                            print("Left click hold - Right section")

            # Right ankle detection
            if right_ankle.visibility > 0.5:
                if prev_right_ankle_x is not None:
                    if right_ankle.x < left_section_line / frame_width:
                        if not clicking:
                            pyautogui.mouseDown()
                            clicking = True
                            print("Left click hold - Left section (Right ankle)")
                    elif right_ankle.x > right_section_line / frame_width:
                        if not clicking:
                            pyautogui.mouseDown()
                            clicking = True
                            print("Left click hold - Right section (Right ankle)")

            # Release condition
            if prev_left_ankle_x is not None and prev_right_ankle_x is not None:
                if (left_section_line / frame_width < left_ankle.x < right_section_line / frame_width) and \
                   (left_section_line / frame_width < right_ankle.x < right_section_line / frame_width):
                    if clicking:
                        pyautogui.mouseUp()
                        clicking = False
                        print("Left click released - Neutral section")
                        pyautogui.moveTo(screen_width // 2, screen_height // 2)

            prev_left_ankle_x = left_ankle.x
            prev_right_ankle_x = right_ankle.x

        # Draw the section lines on the frame (visual reference)
        cv2.line(frame, (left_section_line, 0), (left_section_line, frame_height), (0, 255, 0), 2)
        cv2.line(frame, (right_section_line, 0), (right_section_line, frame_height), (0, 255, 0), 2)

    elif mode == 1:  # Goalkeeper Mode - Hand detection
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            closest_hand = None
            closest_hand_distance = float('inf')

            for landmarks in results.multi_hand_landmarks:
                wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
                distance_to_camera = wrist.z

                if distance_to_camera < closest_hand_distance:
                    closest_hand_distance = distance_to_camera
                    closest_hand = landmarks

            if closest_hand:
                mp_drawing.draw_landmarks(frame, closest_hand, mp_hands.HAND_CONNECTIONS)

                wrist = closest_hand.landmark[mp_hands.HandLandmark.WRIST]
                thumb = closest_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = closest_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                wrist_x = int(wrist.x * frame.shape[1])
                wrist_y = int(wrist.y * frame.shape[0])

                mouse_x = np.interp(wrist_x, [0, frame.shape[1]], [0, screen_width])
                mouse_y = np.interp(wrist_y, [0, frame.shape[0]], [0, screen_height])

                pyautogui.moveTo(mouse_x, mouse_y)

                distance = np.linalg.norm(np.array([thumb.x - index.x, thumb.y - index.y]))
                if distance < 0.05:
                    pyautogui.click()
                    print("Mouse clicked!")
                    pyautogui.moveTo(screen_width // 2, screen_height // 2)
                    print("Cursor moved to center.")

    # Display the resulting frame
    cv2.imshow("Penalty ShootOut Motion Control", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
