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
#     - Type_1
#     - Type_2
#     - Type_3
#   - ... (other letters)

data_dir = "C:\\Users\\riddh\\OneDrive\\Desktop\\BE PROJECT ALPHABETS\\CAPITAL\\"  # Replace with the path to your data directory

# Create lists to store image paths and labels
image_paths = []
labels = []

# Iterate through the directory structure and gather data
for letter_folder in os.listdir(data_dir):
    if os.path.isdir(os.path.join(data_dir, letter_folder)):
        letter_path = os.path.join(data_dir, letter_folder)

        for type_folder in os.listdir(letter_path):
            type_path = os.path.join(letter_path, type_folder)

            if type_folder in ["Type_1", "Type_2"]:
                label = ["Type_1", "Type_2"].index(type_folder)

                for image_file in os.listdir(type_path):
                    if image_file.endswith(".png"):  # Adjust the file extension if needed
                        image_paths.append(os.path.join(type_path, image_file))
                        labels.append(label)

# Load and preprocess the images
images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]
images = [cv2.resize(image, (28, 28)) for image in images]
images = [image / 255.0 for image in images]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

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
    layers.Dense(3, activation='softmax')  # 3 classes (Type_1, Type_2, Type_3)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(np.array(X_train).reshape(-1, 28, 28, 1), y_train, epochs=10)

# Ensure the shape of the test image data is (num_samples, 28, 28, 1)
X_test = np.array(X_test)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convert the test labels to NumPy array
y_test = np.array(y_test)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(np.array(X_test).reshape(-1, 28, 28, 1), y_test)
# print(f"Test accuracy: {test_acc}")

# Load the PNG image
# image_path = "C:\\Users\\riddh\\OneDrive\\Desktop\\BE PROJECT ALPHABETS\\CAPITAL\\J\\J_82818.png"  # Replace with the actual file path
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize the image to 28x28 pixels
# image = cv2.resize(image, (28, 28))

# Normalize the image
# image = image / 255.0

# Ensure the shape of the image is (28, 28, 1)
# image = image.reshape(28, 28, 1)

# Use the trained model to make predictions
# predictions = model.predict(np.array([image]))

# Get the most likely type for the image
# predicted_type = np.argmax(predictions[0])

# Define type labels
# type_labels = ["Type_1", "Type_2"]

# Print the predicted type
# print(f"The image is predicted as {type_labels[predicted_type]}")

# Save the model
model.save("type_identification_J.h5")