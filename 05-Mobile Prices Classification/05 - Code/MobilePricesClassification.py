import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set up file
file_path_train = r"C:\Users\val\OneDrive\Documents\GitHub\Experiment-Portfolio\05-Mobile Prices Classification\05 - Code\train.csv"
file_path_test = r"C:\Users\val\OneDrive\Documents\GitHub\Experiment-Portfolio\05-Mobile Prices Classification\05 - Code\test.csv"
data_train = pd.read_csv(file_path_train)
data_test = pd.read_csv(file_path_test)

print(data_train) # Train data
print(data_test) # Test data

# Set X and y data
X = data_train.drop('price_range', axis=1)
y = data_train['price_range']

# Create Plot
plt.figure(figsize=(12,8)) # Size
y.hist(bins=20, color='red', edgecolor='black') # Plot
plt.tight_layout() # Adjust
plt.show() # Show

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Train Test Split

model = LogisticRegression() # Classification algorithm
model.fit(x_train, y_train) # Fits data

ytrp = model.predict(x_train) # Predict
accuracy = accuracy_score (ytrp, y_train) # Accuracy
print("Training Model Accuracy: ", (accuracy*100)) # Print accuracy score

ytsp = model.predict(x_test) # Predict
accuracy2 = accuracy_score (ytsp, y_test) # Accuracy
print("Testing Model Accuracy: ", (accuracy2*100)) # Print accuracy score