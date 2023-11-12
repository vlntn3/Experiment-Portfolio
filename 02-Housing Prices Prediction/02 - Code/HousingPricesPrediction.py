import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set up file
file_path = r"C:\Users\val\OneDrive\Documents\GitHub\Experiment-Portfolio\02-Housing Prices Prediction\02 - Code\RealEstatePrices.csv"
data = pd.read_csv(file_path)

# Set X and y data
X = data[['transaction date','house age','distance to the nearest MRT station','number of convenience stores','latitude','longitude']]
y = data['house price of unit area']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Train Test Split

model = LinearRegression() # Algorithm
model.fit(x_train, y_train) # Fits data

prediction = model.predict(x_test) # Model Prediction
mse = mean_squared_error(y_test, prediction, squared=False) # RMSE
print("Root Mean Squared Error: ", mse) # Print RMSE