import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2  # For camera functionality
from fer import FER  # For emotion recognition

# Load the sentiment analysis dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Define maximum sequence length
max_length = 100

# Pad sequences to ensure consistent input length
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Create a sentiment analysis model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_length))  # Embedding layer
model.add(LSTM(64, dropout=0.2))  # LSTM layer with dropout
model.add(Dense(64, activation='relu'))  # Dense layer with ReLU activation
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

# After training, open the camera and perform emotion detection
cap = cv2.VideoCapture(0)  # Open the default camera (use 0 for default camera)

# Create an emotion detector using FER
emotion_detector = FER()

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Continuously capture video frames and perform emotion detection
while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        print("Failed to grab frame")
        break

    # Perform emotion detection on the current frame
    emotions = emotion_detector.detect_emotions(frame)

    # Display emotions on the frame
    for emotion in emotions:
        bbox = emotion["box"]
        dominant_emotion = emotion["emotions"]
        
        # Draw a bounding box around the face
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
        
        # Get the emotion with the highest confidence
        emotion_text = max(dominant_emotion, key=dominant_emotion.get)
        
        # Display the emotion text on the frame
        cv2.putText(frame, emotion_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Camera Feed - Emotion Detection', frame)

    # Press 'q' to quit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
