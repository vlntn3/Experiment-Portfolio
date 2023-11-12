import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Set up file
file_path = r"C:\Users\val\OneDrive\Documents\GitHub\Experiment-Portfolio\04-Drug Classification\04 - Code\Drugs.csv"
data = pd.read_csv(file_path)

# Strings
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
label_encoder = LabelEncoder()
data['BP'] = label_encoder.fit_transform(data['BP'])
label_encoder = LabelEncoder()
data['Cholesterol'] = label_encoder.fit_transform(data['Cholesterol'])
label_encoder = LabelEncoder()

print("Data Info:\n", data.info()) # INFO
print("\nFirst 5 Data:\n", data.head()) # HEAD
print("\nLast 5 Data:\n", data.tail()) # TAIL
print("\nDescribe:\n", data.describe()) # DESCRIBE
print("\nData Shape:\n", data.shape) # SHAPE

# Set X and y data
X = data[['Age','Sex','BP','Cholesterol','Na_to_K']]
y = data['Drug']

# Create Plot
plt.figure(figsize=(12,8)) # Size
X.hist(bins=20, color='red', edgecolor='black') # Plot
plt.tight_layout() # Adjust
plt.show() # Show

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Train Test Split

model = LogisticRegression() # Algorithm
model.fit(x_train, y_train) # Fits data

prediction = model.predict(x_test) # Predict
accuracy = accuracy_score (y_test, prediction) # Accuracy
print("Model Accuracy: ", (accuracy*100)) # Print accuracy score