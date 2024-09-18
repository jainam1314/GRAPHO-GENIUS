import numpy as np
import cv2
import math

ANCHOR_POINT = 6000
MIDZONE_THRESHOLD = 15000
MIN_HANDWRITING_HEIGHT_PIXEL = 20

BASELINE_ANGLE = 0.0
TOP_MARGIN = 0.0
LETTER_SIZE = 0.0
LINE_SPACING = 0.0
WORD_SPACING = 0.0
PEN_PRESSURE = 0.0
SLANT_ANGLE = 0.0

''' function for bilateral filtering '''
def bilateralFilter(image, d):
    image = cv2.bilateralFilter(image,d,50,50)
    return image

''' function for median filtering '''
def medianFilter(image, d):
    image = cv2.medianBlur(image,d)
    return image

''' function for INVERTED binary threshold '''	
def threshold(image, t):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,image = cv2.threshold(image,t,255,cv2.THRESH_BINARY_INV)
    return image

''' function for dilation of objects in the image '''
def dilate(image, kernalSize):
    kernel = np.ones(kernalSize, np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return image
    
''' function for erosion of objects in the image '''
def erode(image, kernalSize):
    kernel = np.ones(kernalSize, np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return image
    
def straighten(image):
    global BASELINE_ANGLE

    angle = 0.0
    angle_sum = 0.0
    contour_count = 0.0

    positive_angle_sum = 0.0
    negative_angle_sum = 0.0
    positive_count = 0.0
    negative_count = 0.0

    filtered = bilateralFilter(image, 3)
    thresh = threshold(filtered, 120)
    dilated = dilate(thresh, (5, 100))
    ctrs,im2 = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, ctr in enumerate(ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        if h>w or h<20:
            continue
        roi = image[y:y+h, x:x+w]
        
        if w < image.shape[1]/2:
            roi = 255
            image[y:y+h, x:x+w] = roi
            continue
        
        rect = cv2.minAreaRect(ctr)
        center = rect[0]
        angle = rect[2]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        rot = cv2.getRotationMatrix2D(((x+w)/2, (y+h)/2), angle, 1)
        extract = cv2.warpAffine(roi, rot, (w,h), borderMode = cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        
        image[y:y+h, x:x+w] = extract

        print(angle)
        angle_sum += angle
        contour_count += 1
        mean_angle = angle_sum / contour_count
        BASELINE_ANGLE = mean_angle
        return image

''' function to calculate horizontal projection of the image pixel rows and return it '''
def horizontalProjection(img):
    (h, w) = img.shape[:2]
    sumRows = []
    for j in range(h):
        row = img[j:j+1, 0:w] 
        sumRows.append(np.sum(row))
    return sumRows
    
''' function to calculate vertical projection of the image pixel columns and return it '''
def verticalProjection(img):
    (h, w) = img.shape[:2]
    sumCols = []
    for j in range(w):
        col = img[0:h, j:j+1] 
        sumCols.append(np.sum(col))
    return sumCols
    
''' function to extract lines of handwritten text from the image using horizontal projection '''
# def extractLines(img):

#     global LETTER_SIZE
#     global LINE_SPACING
#     global TOP_MARGIN
    
#     filtered = bilateralFilter(img, 5)
#     thresh = threshold(filtered, 160)
#     hpList = horizontalProjection(thresh)
#     topMarginCount = 0
    
#     for sum in hpList:
#         if(sum<=255):
#             topMarginCount += 1
#         else:
#             break
            
#     lineTop = 0
#     lineBottom = 0
#     spaceTop = 0
#     spaceBottom = 0
#     indexCount = 0
#     setLineTop = True
#     setSpaceTop = True
#     includeNextSpace = True
#     space_zero = [] 
#     lines = [] 
    
#     for i, sum in enumerate(hpList):
#         if(sum==0):
#             if(setSpaceTop):
#                 spaceTop = indexCount
#                 setSpaceTop = False 
#             indexCount += 1
#             spaceBottom = indexCount
#             if(i<len(hpList)-1): 
#                 if(hpList[i+1]==0): 
#                     continue
#             if(includeNextSpace):
#                 space_zero.append(spaceBottom-spaceTop)
#             else:
#                 if (len(space_zero)==0):
#                     previous = 0
#                 else:
#                     previous = space_zero.pop()
#                 space_zero.append(previous + spaceBottom-lineTop)
#             setSpaceTop = True 
        
#         if(sum>0):
#             if(setLineTop):
#                 lineTop = indexCount
#                 setLineTop = False 
#             indexCount += 1
#             lineBottom = indexCount
#             if(i<len(hpList)-1): 
#                 if(hpList[i+1]>0): 
#                     continue
#                 if(lineBottom-lineTop<20):
#                     includeNextSpace = False
#                     setLineTop = True 
#                     continue
#             includeNextSpace = True 
            
#             lines.append([lineTop, lineBottom])
#             setLineTop = True 
    
#         fineLines = [] 
#     for i, line in enumerate(lines):
    
#         anchor = line[0] 
#         anchorPoints = [] 
#         upHill = True 
#         downHill = False 
#         segment = hpList[line[0]:line[1]] 
        
#         for j, sum in enumerate(segment):
#             if(upHill):
#                 if(sum<ANCHOR_POINT):
#                     anchor += 1
#                     continue
#                 anchorPoints.append(anchor)
#                 upHill = False
#                 downHill = True
#             if(downHill):
#                 if(sum>ANCHOR_POINT):
#                     anchor += 1
#                     continue
#                 anchorPoints.append(anchor)
#                 downHill = False
#                 upHill = True

#         if(len(anchorPoints)<2):
#             continue
        
#         if(len(anchorPoints)<=3):
#             fineLines.append(line)
#             continue
    
#         lineTop = line[0]
#         for x in range(1, len(anchorPoints)-1, 2):

#             lineMid = (anchorPoints[x]+anchorPoints[x+1])/2
#             lineBottom = lineMid
#             if(lineBottom-lineTop < 20):
#                 continue
#             fineLines.append([lineTop, lineBottom])
#             lineTop = lineBottom
#         if(line[1]-lineTop < 20):
#             continue
#         fineLines.append([lineTop, line[1]])
        
#     space_nonzero_row_count = 0
#     midzone_row_count = 0
#     lines_having_midzone_count = 0
#     flag = False
#     for i, line in enumerate(fineLines):
#         segment = hpList[line[0]:line[1]]
#         for j, sum in enumerate(segment):
#             if(sum<MIDZONE_THRESHOLD):
#                 space_nonzero_row_count += 1
#             else:
#                 midzone_row_count += 1
#                 flag = True
                
#         if(flag):
#             lines_having_midzone_count += 1
#             flag = False
    
#     if(lines_having_midzone_count == 0): lines_having_midzone_count = 1

#     total_space_row_count = space_nonzero_row_count + np.sum(space_zero[1:-1]) 
#     average_line_spacing = float(total_space_row_count) / lines_having_midzone_count 
#     average_letter_size = float(midzone_row_count) / lines_having_midzone_count
#     LETTER_SIZE = average_letter_size
#     if(average_letter_size == 0): average_letter_size = 1
#     relative_line_spacing = average_line_spacing / average_letter_size
#     LINE_SPACING = relative_line_spacing
#     relative_top_margin = float(topMarginCount) / average_letter_size
#     TOP_MARGIN = relative_top_margin
    
#     return fineLines

def extractLines(img):
    global LETTER_SIZE
    global LINE_SPACING
    global TOP_MARGIN
    
    filtered = bilateralFilter(img, 5)
    thresh = threshold(filtered, 160)
    hpList = horizontalProjection(thresh)
    topMarginCount = 0
    
    for sum in hpList:
        if sum <= 255:
            topMarginCount += 1
        else:
            break
            
    lineTop = 0
    lineBottom = 0
    spaceTop = 0
    spaceBottom = 0
    indexCount = 0
    setLineTop = True
    setSpaceTop = True
    includeNextSpace = True
    space_zero = [] 
    lines = [] 
    
    for i, sum in enumerate(hpList):
        if sum == 0:
            if setSpaceTop:
                spaceTop = indexCount
                setSpaceTop = False 
            indexCount += 1
            spaceBottom = indexCount
            if i < len(hpList) - 1: 
                if hpList[i + 1] == 0: 
                    continue

            if (spaceBottom - spaceTop) > 0:  # Adjust this condition based on your requirement
                space_zero.append(spaceBottom - spaceTop)
                
            setSpaceTop = True 
        
        if sum > 0:
            if setLineTop:
                lineTop = indexCount
                setLineTop = False 
            indexCount += 1
            lineBottom = indexCount
            if i < len(hpList) - 1: 
                if hpList[i + 1] > 0: 
                    continue
                if lineBottom - lineTop < 20:
                    includeNextSpace = False
                    setLineTop = True 
                    continue
            includeNextSpace = True 
            
            lines.append([lineTop, lineBottom])
            setLineTop = True 
    
    fineLines = [] 
    for i, line in enumerate(lines):
        anchor = line[0] 
        anchorPoints = [] 
        upHill = True 
        downHill = False 
        segment = hpList[line[0]:line[1]] 
        
        for j, sum in enumerate(segment):
            if upHill:
                if sum < ANCHOR_POINT:
                    anchor += 1
                    continue
                anchorPoints.append(anchor)
                upHill = False
                downHill = True
            if downHill:
                if sum > ANCHOR_POINT:
                    anchor += 1
                    continue
                anchorPoints.append(anchor)
                downHill = False
                upHill = True

        if len(anchorPoints) < 2:
            continue
        
        if len(anchorPoints) <= 3:
            fineLines.append(line)
            continue
    
        lineTop = line[0]
        for x in range(1, len(anchorPoints) - 1, 2):
            lineMid = (anchorPoints[x] + anchorPoints[x + 1]) // 2
            lineBottom = lineMid
            if lineBottom - lineTop < 20:
                continue
            fineLines.append([lineTop, lineBottom])
            lineTop = lineBottom
        if line[1] - lineTop < 20:
            continue
        fineLines.append([lineTop, line[1]])
        
    space_nonzero_row_count = 0
    midzone_row_count = 0
    lines_having_midzone_count = 0
    flag = False
    for i, line in enumerate(fineLines):
        segment = hpList[line[0]:line[1]]
        for j, sum in enumerate(segment):
            if sum < MIDZONE_THRESHOLD:
                space_nonzero_row_count += 1
            else:
                midzone_row_count += 1
                flag = True
                
        if flag:
            lines_having_midzone_count += 1
            flag = False
    
    if lines_having_midzone_count == 0:
        lines_having_midzone_count = 1

    total_space_row_count = space_nonzero_row_count + np.sum(space_zero[1:-1]) 
    average_line_spacing = float(total_space_row_count) / lines_having_midzone_count 
    average_letter_size = float(midzone_row_count) / lines_having_midzone_count
    LETTER_SIZE = average_letter_size
    if average_letter_size == 0:
        average_letter_size = 1
    relative_line_spacing = average_line_spacing / average_letter_size
    LINE_SPACING = relative_line_spacing
    relative_top_margin = float(topMarginCount) / average_letter_size
    TOP_MARGIN = relative_top_margin
    
    return fineLines

    
''' function to extract words from the lines using vertical projection '''
def extractWords(image, lines):

    global LETTER_SIZE
    global WORD_SPACING
    
    filtered = bilateralFilter(image, 5)
    thresh = threshold(filtered, 180)
    width = thresh.shape[1]
    space_zero = []
    words = []
    
    for i, line in enumerate(lines):
        extract = thresh[line[0]:line[1], 0:width] 
        vp = verticalProjection(extract)
        
        wordStart = 0
        wordEnd = 0
        spaceStart = 0
        spaceEnd = 0
        indexCount = 0
        setWordStart = True
        setSpaceStart = True
        includeNextSpace = True
        spaces = []
        
        for j, sum in enumerate(vp):
            if(sum==0):
                if(setSpaceStart):
                    spaceStart = indexCount
                    setSpaceStart = False 
                indexCount += 1
                spaceEnd = indexCount
                if(j<len(vp)-1): 
                    if(vp[j+1]==0): 
                        continue

                if((spaceEnd-spaceStart) > int(LETTER_SIZE/2)):
                    spaces.append(spaceEnd-spaceStart)
                    
                setSpaceStart = True 
            
            if(sum>0):
                if(setWordStart):
                    wordStart = indexCount
                    setWordStart = False 
                indexCount += 1
                wordEnd = indexCount
                if(j<len(vp)-1): 
                    if(vp[j+1]>0): 
                        continue

                count = 0
                for k in range(line[1]-line[0]):
                    row = thresh[line[0]+k:line[0]+k+1, wordStart:wordEnd] 
                    if(np.sum(row)):
                        count += 1
                if(count > int(LETTER_SIZE/2)):
                    words.append([line[0], line[1], wordStart, wordEnd])
                    
                setWordStart = True 
        
        space_zero.extend(spaces[1:-1])
    
    space_columns = np.sum(space_zero)
    space_count = len(space_zero)
    if(space_count == 0):
        space_count = 1
    average_word_spacing = float(space_columns) / space_count
    relative_word_spacing = average_word_spacing / LETTER_SIZE
    WORD_SPACING = relative_word_spacing
    
    return words
        
''' function to determine the average slant of the handwriting '''
def extractSlant(img, words):
    
    global SLANT_ANGLE
    theta = [-0.785398, -0.523599, -0.261799, -0.0872665, 0.01, 0.0872665, 0.261799, 0.523599, 0.785398]
    s_function = [0.0] * 9
    count_ = [0]*9
    filtered = bilateralFilter(img, 5)
    thresh = threshold(filtered, 180)
    
    for i, angle in enumerate(theta):
        s_temp = 0.0 
        count = 0 
        
        for j, word in enumerate(words):
            original = thresh[word[0]:word[1], word[2]:word[3]]

            height = word[1]-word[0]
            width = word[3]-word[2]
            
            shift = (math.tan(angle) * height) / 2
            
            pad_length = abs(int(shift))
            
            blank_image = np.zeros((height,width+pad_length*2,3), np.uint8)
            new_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
            new_image[:, pad_length:width+pad_length] = original
            
            (height, width) = new_image.shape[:2]
            x1 = width/2
            y1 = 0
            x2 = width/4
            y2 = height
            x3 = 3*width/4
            y3 = height
    
            pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3]])
            pts2 = np.float32([[x1+shift,y1],[x2-shift,y2],[x3-shift,y3]])
            M = cv2.getAffineTransform(pts1,pts2)
            deslanted = cv2.warpAffine(new_image,M,(width,height))
            
            vp = verticalProjection(deslanted)
            
            for k, sum in enumerate(vp):
                if(sum == 0):
                    continue
                
                num_fgpixel = sum / 255

                if(num_fgpixel < int(height/3)):
                    continue
                
                column = deslanted[0:height, k:k+1]
                column = column.flatten()
            
                for l, pixel in enumerate(column):
                    if(pixel==0):
                        continue
                    break
                for m, pixel in enumerate(column[::-1]):
                    if(pixel==0):
                        continue
                    break
                
                delta_y = height - (l+m)
                h_sq = (float(num_fgpixel)/delta_y)**2
                h_wted = (h_sq * num_fgpixel) / height
                s_temp += h_wted
                count += 1
                
        s_function[i] = s_temp
        count_[i] = count
    
    max_value = 0.0
    max_index = 4
    for index, value in enumerate(s_function):
        if(value > max_value):
            max_value = value
            max_index = index
            
    if(max_index == 0):
        angle = 45
        result =  " : Extremely right slanted"
    elif(max_index == 1):
        angle = 30
        result = " : Above average right slanted"
    elif(max_index == 2):
        angle = 15
        result = " : Average right slanted"
    elif(max_index == 3):
        angle = 5
        result = " : A little right slanted"
    elif(max_index == 5):
        angle = -5
        result = " : A little left slanted"
    elif(max_index == 6):
        angle = -15
        result = " : Average left slanted"
    elif(max_index == 7):
        angle = -30
        result = " : Above average left slanted"
    elif(max_index == 8):
        angle = -45
        result = " : Extremely left slanted"
    elif(max_index == 4):
        p = s_function[4] / s_function[3]
        q = s_function[4] / s_function[5]
        if((p <= 1.2 and q <= 1.2) or (p > 1.4 and q > 1.4)):
            angle = 0
            result = " : No slant"
        elif((p <= 1.2 and q-p > 0.4) or (q <= 1.2 and p-q > 0.4)):
            angle = 0
            result = " : No slant"
        else:
            max_index = 9
            angle = 180
            result =  " : Irregular slant behaviour"
        
        if angle == 0:
            print("\n************************************************")
            print("Slant determined to be straight.")
        else:
            print("\n************************************************")
            print("Slant determined to be irregular.")
        cv2.imshow("Check Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        type = input("Press enter if okay, else enter c to change: ")
        if type=='c':
            if angle == 0:
                angle = 180
                result =  " : Irregular Slant"
                print("Set as"+result)
                print("************************************************\n")
            else:
                angle = 0
                result = " : Straight/No Slant"
                print("Set as"+result)
                print("************************************************\n")
        else:
            print("No Change!")
            print("************************************************\n")
        
    SLANT_ANGLE = angle
    return

''' function to extract average pen pressure of the handwriting '''
def barometer(image):

    global PEN_PRESSURE
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:]
    inverted = image
    for x in range(h):
        for y in range(w):
            inverted[x][y] = 255 - image[x][y]
            
    filtered = bilateralFilter(inverted, 3)
    ret, thresh = cv2.threshold(filtered, 100, 255, cv2.THRESH_TOZERO)
    total_intensity = 0
    pixel_count = 0
    for x in range(h):
        for y in range(w):
            if(thresh[x][y] > 0):
                total_intensity += thresh[x][y]
                pixel_count += 1
                
    average_intensity = float(total_intensity) / pixel_count
    PEN_PRESSURE = average_intensity
    return
    
''' main '''
def start(file_name):

    global BASELINE_ANGLE
    global TOP_MARGIN
    global LETTER_SIZE
    global LINE_SPACING
    global WORD_SPACING
    global PEN_PRESSURE
    global SLANT_ANGLE

    try:
        # image = cv2.imread('Test Images/'+file_name)
        image = cv2.imread(file_name)
    except:
        image = cv2.imread(file_name)

    barometer(image)
    straightened = straighten(image)
    lineIndices = extractLines(straightened)
    wordCoordinates = extractWords(straightened, lineIndices)
    extractSlant(straightened, wordCoordinates)
    
    BASELINE_ANGLE = round(BASELINE_ANGLE, 2)
    TOP_MARGIN = round(TOP_MARGIN, 2)
    LETTER_SIZE = round(LETTER_SIZE, 2)
    LINE_SPACING = round(LINE_SPACING, 2)
    WORD_SPACING = round(WORD_SPACING, 2)
    PEN_PRESSURE = round(PEN_PRESSURE, 2)
    SLANT_ANGLE = round(SLANT_ANGLE, 2)

    # print(BASELINE_ANGLE)
    # print(TOP_MARGIN)
    # print(LETTER_SIZE)
    # print(LINE_SPACING)
    # print(WORD_SPACING)
    # print(PEN_PRESSURE)
    # print(SLANT_ANGLE)
    print("Baseline Angle:", BASELINE_ANGLE)
    print("Top Margin:", TOP_MARGIN)
    print("Average Letter Size:", LETTER_SIZE)
    print("Line Spacing:", LINE_SPACING)
    print("Word Spacing:", WORD_SPACING)
    print("Pen Pressure:", PEN_PRESSURE)
    print("Slant Angle:", SLANT_ANGLE)

    
    return [BASELINE_ANGLE, TOP_MARGIN, LETTER_SIZE, LINE_SPACING, WORD_SPACING, PEN_PRESSURE, SLANT_ANGLE]
    
# start("a01-000x.png")

img = "Img (3).jpg"
img_path = r"C:\Users\riddh\OneDrive\Desktop\BE PROJECT GIT CLONES\Handwriting-Analysis-using-Machine-Learning\Test Images\\" + img
print(start(img_path))