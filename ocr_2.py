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
    # Display the original word image
    display_image(word_image, "Original Word Image")
    
    # Invert the word image
    inverted_word_image = cv2.bitwise_not(word_image)
    
    # Calculate vertical histogram projection
    vertical_hist = np.sum(inverted_word_image, axis=0) / 255
    
    # Plot the vertical histogram
    plt.figure(figsize=(10, 5))
    plt.plot(vertical_hist, color='blue')
    plt.xlabel('Column')
    plt.ylabel('Number of White Pixels')
    plt.title('Vertical Histogram Projection')
    plt.show()
    
    # Overlay original word image with the segmented red lines
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(word_image, cmap='gray')
    
    # Draw vertical lines where y < 20
    for col, count in enumerate(vertical_hist):
        if count < 20:
            ax.axvline(x=col, color='red', linestyle='--')
    
    # Combine red lines spanning over 75 x-coordinates into one line
    red_lines = []
    current_line = []
    for col, count in enumerate(vertical_hist):
        if count < 20:
            current_line.append(col)
        else:
            if current_line:
                if len(current_line) > 75:
                    red_lines.append((current_line[0], current_line[-1]))
                else:
                    for c in current_line:
                        ax.axvline(x=c, color='red', linestyle='--')
                current_line = []
    if current_line:
        if len(current_line) > 75:
            red_lines.append((current_line[0], current_line[-1]))
        else:
            for c in current_line:
                ax.axvline(x=c, color='red', linestyle='--')
    
    # Draw a single red line in the center of the combined red lines
    combined_red_lines = []
    if len(red_lines) > 1:
        combined_red_lines = [red_lines[0]]
        for i in range(1, len(red_lines)):
            if red_lines[i][0] - red_lines[i-1][1] > 50:
                combined_red_lines.append((red_lines[i-1][1], red_lines[i][0]))
        combined_red_lines.append(red_lines[-1])
        
        for start, end in combined_red_lines:
            center = (start + end) // 2
            ax.axvline(x=center, color='red', linestyle='-', linewidth=2)
    
    plt.title('Word Image with Segmented Red Lines')
    plt.axis('off')
    plt.show()

# Load the word image
file_path = r"C:\Users\riddh\Downloads\category.jpg"
word_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded successfully
if word_image is None:
    print("Error: Unable to load the image.")
else:
    # Perform word image segmentation
    segment_word_image(word_image)
