# Just a throw away attempt at using lin reg on weather data

import pandas as pd

# Load the dataset
file_path = './twin-falls-all-data-farm-oiko-1950-2023.csv'
data = pd.read_csv(file_path)

# Check the first few rows of the dataframe
#print(data.head())

# Check for missing values
#print(data.isnull().sum())

# Fill missing values if necessary, for example with the mean of the column
#data.fillna(data.mean(), inplace=True)


# Convert date to datetime

# Convert date to datetime
data['date'] = pd.to_datetime(data.iloc[:, 0])

# Extract year, month, day, hour as new features
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour


print(data.columns)

# # Use time, soil temp, and surface rad to predict temp
features = data[['year', 'month', 'day', 'hour', 'soil_temperature_level_1 (degC)', 'surface_solar_radiation (W/m^2)']]
target = data['temperature (degC)']


# split data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# basic implementation using a linear regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the testing set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Extrapolate future trends
# Example: Predicting the next year's data
import numpy as np

# Generate data for the next year
future_data = pd.DataFrame({
    'year': 2024,
    'month': np.tile(np.arange(1, 13), 31),
    'day': np.repeat(np.arange(1, 32), 12),
    'hour': 12,
    'soil_temperature_level_1 (degC)': 10,  # Corrected feature name
    'surface_solar_radiation (W/m^2)': 150  # Corrected feature name
})

# Predict future temperatures
future_temperatures = model.predict(future_data)
print(future_temperatures)
