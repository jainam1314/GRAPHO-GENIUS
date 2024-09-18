import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display image
def display_image(image, title="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to perform word image segmentation
def segment_word_image(word_image):
    # Invert the word image
    inverted_word_image = cv2.bitwise_not(word_image)
    
    # Calculate vertical histogram projection
    vertical_hist = np.sum(inverted_word_image, axis=0) / 255
    
    # Plot the vertical histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(vertical_hist, color='blue')
    ax.set_xlabel('Column')
    ax.set_ylabel('Number of White Pixels')
    ax.set_title('Vertical Histogram Projection')
    
    # Overlay original word image
    ax.imshow(word_image, extent=[0, len(vertical_hist), 0, max(vertical_hist)], aspect='auto', alpha=0.5)
    
    # Draw vertical lines where y < 50
    # for col, count in enumerate(vertical_hist):
    #     if count < 50:
    #         ax.axvline(x=col, color='red', linestyle='--')
    
    plt.show()

# Load the word image
file_path = r"C:\Users\riddh\Downloads\category.jpg"
word_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded successfully
if word_image is None:
    print("Error: Unable to load the image.")
else:
    # Display the original word image
    display_image(word_image, "Original Word Image")
    
    # Perform word image segmentation
    segment_word_image(word_image)