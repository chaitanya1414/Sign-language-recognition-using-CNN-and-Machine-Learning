import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load and preprocess the dataset
data_directory = 'data/'
images = []
labels = []

# Iterate over files in the data directory
for filename in os.listdir(data_directory):
    # Load image
    image = Image.open(os.path.join(data_directory, filename))
    # Resize image to a fixed size (e.g., 64x64)
    image = image.resize((64, 64))
    # Convert image to numpy array and normalize pixel values
    image = np.array(image) / 255.0
    # Append image to list
    images.append(image)
    # Extract label from filename and append to labels list
    label = filename.split('_')[0]  # Assuming labels are before the first underscore in the filename
    labels.append(label)

# Convert lists to numpy arrays
X = np.array(images)
y = np.array(labels)

# Update label_mapping for the additional signs
label_mapping = {
    'A': 0,
    'B': 1,
    'C': 2,
    'L': 3
}

# Number of classes is now 8
num_classes = len(label_mapping)

y_one_hot = tf.keras.utils.to_categorical([label_mapping[label] for label in y], num_classes=num_classes)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Adjusted for the total number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('sign_language_cnn_model_extended.h5')
