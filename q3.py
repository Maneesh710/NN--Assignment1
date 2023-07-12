import pandas as pd

# Import the dataset
data = pd.read_csv(r"C:\Users\User\Downloads\NNDL_Code and Data\NNDL_Code and Data\Salary_Data.csv")

# Separate the features (X) and the target variable (y)
X = data["YearsExperience"].values.reshape(-1, 1)
y = data["Salary"].values.reshape(-1, 1)

from sklearn.model_selection import train_test_split

# Split the data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LinearRegression

# Create the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the salaries for the test data
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

import matplotlib.pyplot as plt

# Plot the training data
plt.scatter(X_train, y_train, color='blue', label='Training Data')

# Plot the test data
plt.scatter(X_test, y_test, color='red', label='Test Data')

# Plot the predicted values
plt.plot(X_test, y_pred, color='green', linewidth=2, label='Predicted Data')

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression - Salary Prediction")
plt.legend()
plt.show()

