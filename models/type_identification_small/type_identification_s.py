import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Data directory structure:
# - capital
#   - A
#     - Type 1
#       - A_29.png
#     - Type 2
#     - Type 3
#   - ... (other letters)

data_dir = "C:\\\\Users\\\\riddh\\\\OneDrive\\\\Desktop\\\\BE PROJECT ALPHABETS\\\\SMALL\\\\"  # Replace with the path to your data directory

# Create lists to store image paths and labels
image_paths = []
labels = []

# Iterate through the directory structure and gather data
for letter_folder in os.listdir(data_dir):
    letter_path = os.path.join(data_dir, letter_folder)
    if os.path.isdir(letter_path):
        for type_folder in os.listdir(letter_path):
            type_path = os.path.join(letter_path, type_folder)
            if os.path.isdir(type_path):
                label = int(type_folder.split()[-1]) - 1  # Extract the label from the folder name
                for image_file in os.listdir(type_path):
                    if image_file.endswith(".png"):  # Adjust the file extension if needed
                        image_paths.append(os.path.join(type_path, image_file))
                        labels.append(label)

# Load and preprocess the images
images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]
images = [cv2.resize(image, (28, 28)) for image in images]
images = [image / 255.0 for image in images]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

# Convert the lists to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Ensure the shape of the image data is (num_samples, 28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)

# Define and train the CNN model
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes (Type 1, Type 2, Type 3)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

# Ensure the shape of the test image data is (num_samples, 28, 28, 1)
X_test = np.array(X_test)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convert the test labels to NumPy array
y_test = np.array(y_test)

# Ensure the shape of the test image data is (num_samples, 28, 28, 1)
X_test = np.array(X_test)
X_test = X_test.reshape(-1, 28, 28, 1)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)

# Evaluate the model on the test data
# test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save("type_identification_s.h5")

# Load the sample image of the letter "o"
sample_image_path = "C:\\Users\\riddh\\OneDrive\\Desktop\\BE PROJECT ALPHABETS\\SMALL\\s\\Type 1\\S_944.png"
sample_image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)

# Preprocess the image (resize, normalize)
sample_image = cv2.resize(sample_image, (28, 28))
sample_image = sample_image / 255.0

# Ensure the shape of the image is (num_samples, 28, 28, 1)
sample_image = sample_image.reshape(1, 28, 28, 1)

# Use the trained model to make predictions
predictions = model.predict(sample_image)

# Get the predicted type
predicted_type_index = np.argmax(predictions[0])
predicted_type = f"Type {predicted_type_index + 1}"

# Print the predicted type
print(f"The sample image is predicted to be {predicted_type}")
