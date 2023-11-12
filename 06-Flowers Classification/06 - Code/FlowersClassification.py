import pandas as pd
import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Setup folder
main_folder = 'C:\\Users\\val\\OneDrive\\Documents\\GitHub\\Experiment-Portfolio\\06-Flowers Classification\\06 - Code\\flower_images'

def load_content(main_folder): # Load function
    photos = []
    labels = []
    cname = os.listdir(main_folder)

    for cn in cname:
        cp = os.path.join(main_folder, cn) # Flower folders

        for fname in os.listdir(cp):
            imgp = os.path.join(cp, fname) # Flower folder photos
            img = cv2.imread(imgp) # Read photos
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale
            img = cv2.resize(img, (100, 100))  # Resize all equally
            photos.append(img.flatten())  # Flatten the image
            labels.append(cn)

    return np.array(photos), np.array(labels)

photos, labels = load_content(main_folder) # Load function

x_train, x_test, y_train, y_test = train_test_split(photos, labels, test_size=0.2, random_state=42) # Train Test Split

# SVM Classification Algorithm
svm = SVC()
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test) # Prediction
accuracy = accuracy_score(y_test, y_pred) # Accuracy
print("Accuracy: ", (accuracy*100)) # Print result