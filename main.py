import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (assuming the file is downloaded and placed in the same directory)
df = pd.read_csv('sample_submission.csv')

# Extract relevant features
features = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
target = df['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-Squared: {r2}')

# Make a prediction for a new house
new_house = [[2000, 3, 2]]  # Example input: 2000 sq ft, 3 bedrooms, 2 bathrooms
predicted_price = model.predict(new_house)
print(f'Predicted Price: ${predicted_price[0]:.2f}')
