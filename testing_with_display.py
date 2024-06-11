import serial
import cv2
import numpy as np
import tensorflow as tf
import time

# Load the trained model
model = tf.keras.models.load_model('sign_language_cnn_model_extended.h5')

# Update the label mapping based on your extended training
label_mapping = {
    0: 'A', 
    1: 'B', 
    2: 'C', 
    3: 'L', 
    4: 'Bye', 
    5: 'Ok', 
    6: 'Wow',
    7: 'Chaitanya',
    8: 'Blank'  # Assuming 'Blank' is a class you've trained your model with
}

# Set a confidence threshold to determine if a prediction should be considered valid
confidence_threshold = 0.5  # Adjust based on your model's performance

# Function to preprocess frames
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=0)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Open a serial connection to Arduino Uno
ser = serial.Serial('COMX', 9600)  # Replace 'COMX' with the appropriate port

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hand_rect = (100, 100, 200, 200)  # Define the region of interest for hand gestures
    cv2.rectangle(frame, (hand_rect[0], hand_rect[1]), (hand_rect[0] + hand_rect[2], hand_rect[1] + hand_rect[3]), (0, 255, 0), 2)
    roi = frame[hand_rect[1]:hand_rect[1]+hand_rect[3], hand_rect[0]:hand_rect[0]+hand_rect[2]]
    
    # Check if the hand region is valid
    if np.mean(roi) < 10:  # Adjust the threshold based on your environment
        predicted_label = 'Blank'
        confidence = 100.0
    else:
        preprocessed_frame = preprocess_frame(roi)
        predictions = model.predict(preprocessed_frame)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        # Manual check to correct predictions under specific conditions
        if confidence < confidence_threshold:
            predicted_label = 'Blank'
        else:
            predicted_label = label_mapping[predicted_class]
            confidence *= 100  # Convert to percentage

    # Display the predicted gesture and confidence
    cv2.putText(frame, '{}: {:.2f}%'.format(predicted_label, confidence), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('frame', frame)
    
    # If confidence is more than 92%, send predicted text to Arduino
    if confidence > 0.92:
        ser.write(predicted_label.encode())
        print("Predicted Gesture:", predicted_label)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
