import cv2
import numpy as np
import pywhatkit as kit
import time
import os

# Load the YOLOv3 model
net = cv2.dnn.readNet(r'C:\Users\NAMAN SHARMA\OneDrive\Desktop\intrusion\sample\yolov3.weights', 
                      r'C:\Users\NAMAN SHARMA\OneDrive\Desktop\intrusion\sample\yolov3.cfg')

# Load the COCO classes
classes = []
with open(r'C:\Users\NAMAN SHARMA\OneDrive\Desktop\intrusion\sample\coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define the weapon classes of interest
weapon_classes = ["gun", "knife", "sword"]

# Capture video from a camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    
    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    
    # Set the input for the YOLOv3 model
    net.setInput(blob)
    
    # Run the YOLOv3 model
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    
    weapon_detected = False  # Flag to check if a weapon is detected
    photo_path = 'detected_weapon.jpg'  # Path to save the photo

    # Loop through the detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Skip the first 5 elements (x, y, w, h, confidence)
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in weapon_classes:
                weapon_detected = True  # Set the flag to True
                
                # Get the bounding box center coordinates and dimensions
                center_x, center_y, w, h = detection[0:4] * np.array([width, height, width, height])
                
                # Convert to top-left corner coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Draw a bounding box around the detected weapon
                cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (0, 0, 255), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # If a weapon is detected, capture the frame and send it via WhatsApp
    if weapon_detected:
        cv2.imwrite(photo_path, frame)  # Save the frame as an image
        print("Weapon detected! Capturing photo and sending via WhatsApp...")
        
        # Send the photo via WhatsApp
        # Replace 'your_phone_number' with the recipient's phone number
        kit.sendwhatmsg("+918433045438", "Weapon detected!",0,0,0, True)
        kit.sendwhats_image("+918433045438", photo_path, "Captured image of weapon.")
        time.sleep(10)  # Wait to ensure the message is sent

    # Display the video feed with detections
    cv2.imshow('Weapon Detection', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Clean up the saved photo after sending
if os.path.exists(photo_path):
    os.remove(photo_path)
