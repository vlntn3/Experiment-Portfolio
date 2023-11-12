import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Set up file
file_path = r"C:\Users\val\OneDrive\Documents\GitHub\Experiment-Portfolio\03-Heart Failure Prediction\03 - Code\HeartData.csv"
data = pd.read_csv(file_path)

# Strings to numerical
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Chest Pain Type'] = label_encoder.fit_transform(data['Chest Pain Type'])
data['Resting ECG'] = label_encoder.fit_transform(data['Resting ECG'])
data['Exercise Angina'] = label_encoder.fit_transform(data['Exercise Angina'])
data['ST Slope'] = label_encoder.fit_transform(data['ST Slope'])

print("Data Info:\n", data.info()) # INFO
print("\nFirst 5 Data:\n", data.head()) # HEAD
print("\nLast 5 Data:\n", data.tail()) # TAIL
print("\nDescribe:\n", data.describe()) # DESCRIBE
print("\nData Shape:\n", data.shape) # SHAPE

# Set X and y data
X = data[['Age','Sex','Chest Pain Type','Resting BP','Cholesterol','Fasting BS','Resting ECG','Max HR','Exercise Angina','Old peak','ST Slope']]
y = data['Heart Disease']

# Create Plot
plt.figure(figsize=(12,8)) # Size
X.hist(bins=20, color='red', edgecolor='black') # Plot
plt.tight_layout() # Adjust
plt.show() # Show

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Train Test Split

model = LogisticRegression() # Algorithm
model.fit(x_train, y_train) # Fits the data

prediction = model.predict(x_test) # Predict
accuracy = accuracy_score (y_test, prediction) # Accuracy
print("Model Accuracy: ", (accuracy*100)) # Print accuracy score