# Import required libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def mask_detection_and_prediction(frame, face_detector, mask_classifier):
    # Obtain the dimensions of the current frame
    (height, width) = frame.shape[:2]
    
    # Create a blob from the image for input into the network
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Pass the blob through the face detection model
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # Initialize lists for detected faces, their locations, and mask predictions
    detected_faces = []
    face_locations = []
    predictions = []

    # Iterate through the detections
    for i in range(detections.shape[2]):
        # Extract confidence level for each detection
        confidence = detections[0, 0, i, 2]

        # Proceed only if the detection confidence is above the threshold
        if confidence > 0.5:
            # Calculate bounding box coordinates for the detected face
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure bounding box coordinates are within frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(width - 1, endX), min(height - 1, endY))

            # Extract the face region, convert color, resize, and preprocess
            face_region = frame[startY:endY, startX:endX]
            face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face_region = cv2.resize(face_region, (224, 224))
            face_region = img_to_array(face_region)
            face_region = preprocess_input(face_region)

            # Add the face and bounding box coordinates to their respective lists
            detected_faces.append(face_region)
            face_locations.append((startX, startY, endX, endY))

    # Make predictions if any faces were detected
    if len(detected_faces) > 0:
        detected_faces = np.array(detected_faces, dtype="float32")
        predictions = mask_classifier.predict(detected_faces, batch_size=32)

    # Return the detected face locations and their mask predictions
    return (face_locations, predictions)

# Load the face detector model from disk
prototxt_path = r"face_detector\deploy.prototxt"
weights_path = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
face_detector = cv2.dnn.readNet(prototxt_path, weights_path)

# Load the face mask detection model
mask_classifier = load_model("mask_detector.model")

# Initialize video streaming
print("[INFO] Starting video stream...")
video_stream = VideoStream(src=0).start()

# Process frames from the video stream
while True:
    # Capture frame from video stream and resize it
    frame = video_stream.read()
    frame = imutils.resize(frame, width=400)

    # Detect faces and predict mask usage
    (face_locations, predictions) = mask_detection_and_prediction(frame, face_detector, mask_classifier)

    # Loop over the detected faces and their corresponding predictions
    for (box, pred) in zip(face_locations, predictions):
        # Unpack bounding box coordinates and predictions
        (startX, startY, endX, endY) = box
        (mask_prob, no_mask_prob) = pred

        # Determine the label and color for bounding box
        label = "Mask" if mask_prob > no_mask_prob else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Format label to include the prediction probability
        label = "{}: {:.2f}%".format(label, max(mask_prob, no_mask_prob) * 100)

        # Draw the label and bounding box on the frame
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Display the processed frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit loop if 'q' key is pressed
    if key == ord("q"):
        break

# Clean up resources
cv2.destroyAllWindows()
video_stream.stop()
