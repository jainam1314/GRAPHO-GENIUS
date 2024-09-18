import os
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from scripts import extractor
from . import extractor
# from scripts import categorize
from . import categorize

import warnings
warnings.filterwarnings("ignore")

X_baseline_angle = []
X_top_margin = []
X_letter_size = []
X_line_spacing = []
X_word_spacing = []
X_pen_pressure = []
X_slant_angle = []

y_t1 = []
y_t2 = []
y_t3 = []
y_t4 = []
y_t5 = []
y_t6 = []
y_t7 = []
y_t8 = []
page_ids = []

# label_list = "C:\\Users\riddh\OneDrive\Desktop\BE PROJECT GIT CLONES\Handwriting-Analysis-using-Machine-Learning\scripts\Extracted_Features\label_list.txt"
label_list = r"C:\Users\riddh\OneDrive\Desktop\BE PROJECT GIT CLONES\Handwriting-Analysis-using-Machine-Learning\scripts\Extracted_Features\label_list.txt"
# model_folder_path = "models/rbf"
model_folder_path = "models/poly"

if os.path.isfile(label_list):
    print("Info: label_list found.")
    #---------x------------x-------------x---------#
    with open(label_list, "r") as labels:
        for line in labels:
            content =  line.split()

            baseline_angle = float(content[0])
            X_baseline_angle.append(baseline_angle)

            top_margin = float(content[1])
            X_top_margin.append(top_margin)

            letter_size = float(content[2])
            X_letter_size.append(letter_size)

            line_spacing = float(content[3])
            X_line_spacing.append(line_spacing)

            word_spacing = float(content[4])
            X_word_spacing.append(word_spacing)

            # pen_pressure = float(content[5])
            # X_pen_pressure.append(pen_pressure)

            slant_angle = float(content[5])
            X_slant_angle.append(slant_angle)

            trait_1 = float(content[6])
            y_t1.append(trait_1)

            # trait_2 = float(content[8])
            # y_t2.append(trait_2)

            trait_3 = float(content[7])
            y_t3.append(trait_3)

            trait_4 = float(content[8])
            y_t4.append(trait_4)

            trait_5 = float(content[9])
            y_t5.append(trait_5)

            trait_6 = float(content[10])
            y_t6.append(trait_6)

            trait_7 = float(content[11])
            y_t7.append(trait_7)

            trait_8 = float(content[12])
            y_t8.append(trait_8)

            page_id = content[13]
            page_ids.append(page_id)
    #------------x-----------x-------------x------------#

    # Emotional stability
    X_t1 = []
    for a, b in zip(X_baseline_angle, X_slant_angle):
        X_t1.append([a,b])

    # Mental energy or will power
    # X_t2 = []
    # for a, b in itertools.izip(X_letter_size, X_pen_pressure):
    #     X_t2.append([a,b])

    # Modesty
    X_t3 = []
    for a, b in zip(X_letter_size, X_top_margin):
        X_t3.append([a,b])

    # Personal harmony and flexiblity
    X_t4 = []
    for a, b in zip(X_slant_angle, X_top_margin):
        X_t4.append([a,b])

    # Lack of discipline
    X_t5 = []
    for a, b in zip(X_slant_angle, X_word_spacing):
        X_t5.append([a,b])

    # Poor concentration
    X_t6 = []
    for a, b in zip(X_letter_size, X_line_spacing):
        X_t6.append([a,b])

    # Non communicativeness
    X_t7 = []
    for a, b in zip(X_letter_size, X_word_spacing):
        X_t7.append([a,b])

    # Social isolation
    X_t8 = []
    for a, b in zip(X_line_spacing, X_word_spacing):
        X_t8.append([a,b])

    #print(X_t1)
    #print(type(X_t1))
    #print(len(X_t1))

    #---------x----------x------------x--------------x---------------x---------------x---------------#
    X_train, X_test, y_train, y_test = train_test_split(X_t1, y_t1, test_size = .30, random_state=8)
    clf1 = SVC(kernel='poly')
    clf1.fit(X_train, y_train)
    print("Classifier 1 accuracy: ",accuracy_score(clf1.predict(X_test), y_test))
    filename1 = model_folder_path + "\clf1.sav"
    pickle.dump(clf1, open(filename1, 'wb'))
 
	
    # X_train, X_test, y_train, y_test = train_test_split(X_t2, y_t2, test_size = .30, random_state=16)
    # clf2 = SVC(kernel='rbf')
    # clf2.fit(X_train, y_train)
    # print("Classifier 2 accuracy: ",accuracy_score(clf2.predict(X_test), y_test))
    # filename2 = model_folder_path + "\clf2.sav"
    # pickle.dump(clf2, open(filename2, 'wb'))
	
    X_train, X_test, y_train, y_test = train_test_split(X_t3, y_t3, test_size = .30, random_state=32)
    clf3 = SVC(kernel='poly')
    clf3.fit(X_train, y_train)
    print("Classifier 3 accuracy: ",accuracy_score(clf3.predict(X_test), y_test))
    filename3 = model_folder_path + "\clf3.sav"
    pickle.dump(clf3, open(filename3, 'wb'))
    
    X_train, X_test, y_train, y_test = train_test_split(X_t4, y_t4, test_size = .30, random_state=64)
    clf4 = SVC(kernel='poly')
    clf4.fit(X_train, y_train)
    print("Classifier 4 accuracy: ",accuracy_score(clf4.predict(X_test), y_test))
    filename4 = model_folder_path + "\clf4.sav"
    pickle.dump(clf4, open(filename4, 'wb'))
    
    X_train, X_test, y_train, y_test = train_test_split(X_t5, y_t5, test_size = .30, random_state=42)
    clf5 = SVC(kernel='poly')
    clf5.fit(X_train, y_train)
    print("Classifier 5 accuracy: ",accuracy_score(clf5.predict(X_test), y_test))
    filename5 = model_folder_path + "\clf5.sav"
    pickle.dump(clf5, open(filename5, 'wb'))
    
    X_train, X_test, y_train, y_test = train_test_split(X_t6, y_t6, test_size = .30, random_state=52)
    clf6 = SVC(kernel='poly')
    clf6.fit(X_train, y_train)
    print("Classifier 6 accuracy: ",accuracy_score(clf6.predict(X_test), y_test))
    filename6 = model_folder_path + "\clf6.sav"
    pickle.dump(clf6, open(filename6, 'wb'))
    
    X_train, X_test, y_train, y_test = train_test_split(X_t7, y_t7, test_size = .30, random_state=21)
    clf7 = SVC(kernel='poly')
    clf7.fit(X_train, y_train)
    print("Classifier 7 accuracy: ",accuracy_score(clf7.predict(X_test), y_test))
    filename7 = model_folder_path + "\clf7.sav"
    pickle.dump(clf7, open(filename7, 'wb'))
    
    X_train, X_test, y_train, y_test = train_test_split(X_t8, y_t8, test_size = .30, random_state=73)
    clf8 = SVC(kernel='poly')
    clf8.fit(X_train, y_train)
    print("Classifier 8 accuracy: ",accuracy_score(clf8.predict(X_test), y_test))
    filename8 = model_folder_path + "\clf8.sav"
    pickle.dump(clf8, open(filename8, 'wb'))
    
    #---------x-----------x-------------x----------x-------------x-------------x-------------x------------#

    while True:
        file_name = input("Enter the file name to predict or z to exit: ")
        if file_name == 'z':
            break
        
        raw_features = extractor.start(file_name)

        raw_baseline_angle = raw_features[0]
        baseline_angle, comment = categorize.determine_baseline_angle(raw_baseline_angle)
        print("Baseline Angle: "+comment)

        raw_top_margin = raw_features[1]
        top_margin, comment = categorize.determine_top_margin(raw_top_margin)
        print("Top Margin: "+comment)

        raw_letter_size = raw_features[2]
        letter_size, comment = categorize.determine_letter_size(raw_letter_size)
        print("Letter Size: "+ comment)

        raw_line_spacing = raw_features[3]
        line_spacing, comment = categorize.determine_line_spacing(raw_line_spacing)
        print("Line Spacing: "+ comment)

        raw_word_spacing = raw_features[4]
        word_spacing, comment = categorize.determine_word_spacing(raw_word_spacing)
        print("Word Spacing: "+ comment)

        # raw_pen_pressure = raw_features[5]
        # pen_pressure, comment = categorize.determine_pen_pressure(raw_pen_pressure)
        # print("Pen Pressure: "+ comment)

        raw_slant_angle = raw_features[5]
        slant_angle, comment = categorize.determine_slant_angle(raw_slant_angle)
        print("Slant Angle: "+ comment)

        print()
        print("Emotional Stability: ", clf1.predict([[baseline_angle, slant_angle]]))
        # print("Mental Energy or Will Power: ", clf2.predict([[letter_size, pen_pressure]]))
        print("Modesty: ", clf3.predict([[letter_size, top_margin]]))
        print("Personal Harmony and Flexibility: ", clf4.predict([[line_spacing, word_spacing]]))
        print("Lack of Discipline: ", clf5.predict([[slant_angle, top_margin]]))
        print("Poor Concentration: ", clf6.predict([[letter_size, line_spacing]]))
        print("Non Communicativeness: ", clf7.predict([[letter_size, word_spacing]]))
        print("Social Isolation: ", clf8.predict([[line_spacing, word_spacing]]))

        print("#----------------x-----------x------------x---------x-------------x-----------x------------#")
else:
    print("Error: label_list file not found.")
