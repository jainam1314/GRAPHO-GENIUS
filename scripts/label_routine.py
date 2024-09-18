import os

def determine_trait_1(baseline_angle, slant_angle):
    # trait_1 = emotional stability | 1 = stable, 0 = not stable.
    if slant_angle == 0 or slant_angle == 4 or slant_angle == 6 or baseline_angle == 0:
        return 0
    else:
        return 1

def determine_trait_2(letter_size, pen_pressue):
    # trait_2 = mental energy or will power | 1 = high or average, 0 = low
    if (pen_pressue == 0 or pen_pressue == 2) or (letter_size == 1 or letter_size == 2):
        return 1
    else: 
        return 0

def determine_trait_3(top_margin, letter_size):
    # trait_3 = modesty | 1 = observed, 0 = not observed (not necessarily the opposite)
    if top_margin == 0 or letter_size == 1:
        return 1
    else:
        return 0

def determine_trait_4(line_spacing, word_spacing):
    # trait_4 = personal harmony and flexibility | 1 = harmonious, 0 = non harmonious
    if line_spacing == 2 and word_spacing == 2:
        return 1
    else: 
        return 0

def determine_trait_5(top_margin, slant_angle):
    # trait_5 = lack of discipline | 1 = observed, 0 = not observed (not necessarily the opposite)
    if top_margin == 1 and slant_angle == 6:
        return 1
    else:
        return 0

def determine_trait_6(letter_size, line_spacing):
    # trait_6 = poor concentration power | 1 = observed, 0 = not observed (not necessarily the opposite)
    if letter_size == 0 and line_spacing == 1:
        return 1
    else:
        return 0

def determine_trait_7(letter_size, word_spacing):
    # trait_7 = non communicativeness | 1 = observed, 0 = not observed (not necessarily the opposite)
    if letter_size == 1 and word_spacing == 0:
        return 1
    else:
        return 0

def determine_trait_8(line_spacing, word_spacing):
    # trait_8 = social isolation | 1 = observed, 0 = not observed (not necessarily the opposite)
    if word_spacing == 0 or line_spacing == 0:
        return 1
    else:
        return 0

# Specify the file paths
label_list = "C:\\Users\\riddh\\OneDrive\\Desktop\\BE PROJECT GIT CLONES\\Handwriting-Analysis-using-Machine-Learning\\scripts\\Extracted_Features\\label_list.txt"
feature_list_path = "C:\\Users\\riddh\\OneDrive\\Desktop\\BE PROJECT GIT CLONES\\Handwriting-Analysis-using-Machine-Learning\\scripts\\Extracted_Features\\feature_list.txt"

# Check if the feature_list file exists
if os.path.isfile(feature_list_path):
    print("Info: feature_list found.")
    
    # Open the feature_list file for reading
    with open(feature_list_path, "r") as features:
        for line in features:
            print("Line:", line)  # Print the current line to check its content
            content = line.split()
            print("Content:", content)  # Print the split content to check if it's correct
            
            # Extract the values from the content
            baseline_angle = float(content[0])
            top_margin = float(content[1])
            letter_size = float(content[2])
            line_spacing = float(content[3])
            word_spacing = float(content[4])
            slant_angle = float(content[5])
            page_id = content[6]
            
            # Debugging print statements to check the extracted values
            print("Baseline Angle:", baseline_angle)
            print("Top Margin:", top_margin)
            print("Letter Size:", letter_size)
            print("Line Spacing:", line_spacing)
            print("Word Spacing:", word_spacing)
            print("Slant Angle:", slant_angle)
            print("Page ID:", page_id)
            
    print("Done!")
    
else:
    print("Error: feature_list file not found.")
