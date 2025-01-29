import cv2 as cv  
import numpy as np 
import mediapipe as mp  
import math 

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh 

# Define landmarks for eyes and iris tracking
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398] 
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246] 
RIGHT_IRIS = [474, 475, 476, 477] 
LEFT_IRIS = [469, 470, 471, 472] 
L_H_LEFT = [33]  # Right eye rightmost landmark 
L_H_RIGHT = [133]  # Right eye leftmost landmark 
R_H_LEFT = [362]  # Left eye rightmost landmark 
R_H_RIGHT = [263]  # Left eye leftmost landmark 

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2): 
    x1, y1 = point1.ravel()  
    x2, y2 = point2.ravel() 
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  

# Function to determine iris position
def iris_position(iris_center, right_point, left_point):  
    center_to_right_dist = euclidean_distance(iris_center, right_point) 
    total_distance = euclidean_distance(right_point, left_point) 
    
    if total_distance == 0: 
        raise ValueError("Total distance is zero, which will cause a division by zero error.") 
    
    ratio = center_to_right_dist / total_distance 
    
    # Determine iris position based on the calculated ratio
    if ratio <= 0.42: 
        return "right", ratio 
    elif 0.42 < ratio <= 0.57: 
        return "center", ratio 
    else: 
        return "left", ratio  

# Initialize webcam capture
cap = cv.VideoCapture(0) 

# Initialize Face Mesh with required parameters
with mp_face_mesh.FaceMesh( 
    max_num_faces=2, 
    refine_landmarks=True, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5 
) as face_mesh: 
    while True: 
        ret, frame = cap.read()  # Read frame from webcam
        if not ret: 
            break 
        
        frame = cv.flip(frame, 1)  # Flip frame horizontally for a mirror effect
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert frame to RGB
        img_h, img_w = frame.shape[:2]  # Get frame dimensions
        
        # Process frame with Face Mesh
        results = face_mesh.process(rgb_frame) 
        
        if results.multi_face_landmarks: 
            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ])
            
            # Compute iris center and radius
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS]) 
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS]) 
            
            center_left = np.array([l_cx, l_cy], dtype=np.int32) 
            center_right = np.array([r_cx, r_cy], dtype=np.int32) 
            
            # Draw circles around detected irises
            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA) 
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA) 
            
            # Draw landmarks for eye edges
            cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA) 
            cv.circle(frame, mesh_points[R_H_LEFT][0], 3, (0, 255, 255), -1, cv.LINE_AA) 
            
            try: 
                # Determine iris position and display it
                iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT][0], mesh_points[R_H_LEFT][0]) 
                cv.putText(frame, f"Iris Position: {iris_pos} {ratio:.2f}", (30, 30), 
                            cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA) 
            except ValueError as e: 
                cv.putText(frame, str(e), (30, 30), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1, cv.LINE_AA) 
        
        # Display the frame
        cv.imshow('Eye Tracking', frame) 
        
        # Exit when 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'): 
            break 

# Release resources
cap.release() 
cv.destroyAllWindows()
