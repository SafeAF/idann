import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Read the CSV file into a DataFrame
df = pd.read_csv('twin-falls-all-data-farm-oiko-1950-2023-corrected.csv')

# Ensure that 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Filter data for fall months (August to December)
df_fall = df[df['date'].dt.month >= 8]

# Initialize a list to store first frost dates
first_frost_dates = []

# Group data by year
for year, group in df_fall.groupby(df_fall['date'].dt.year):
    # Sort by date
    group = group.sort_values('date')
    # Find the first date when temperature drops below 0°C
    frost_days = group[group['temperature (degC)'] <= 0]
    if not frost_days.empty:
        first_frost_date = frost_days.iloc[0]['date']
        first_frost_dates.append({'year': year, 'date': first_frost_date})
    else:
        # If no frost found in fall months, skip the year
        continue

# Create a DataFrame from the list
frost_df = pd.DataFrame(first_frost_dates)

# Convert dates to day of year
frost_df['day_of_year'] = frost_df['date'].dt.dayofyear

# Set 'year' as the index
frost_df.set_index('year', inplace=True)

# Prepare the time series data
time_series = frost_df['day_of_year']

# Check for missing years and handle accordingly
# For simplicity, we'll fill missing years by linear interpolation
time_series = time_series.reindex(range(time_series.index.min(), time_series.index.max() + 1))
time_series.interpolate(method='linear', inplace=True)

# Fit the ARIMA model
# Determine the order (p, d, q)
# For simplicity, we'll use (1,1,1)
model = ARIMA(time_series, order=(1,1,1))
model_fit = model.fit()

# Forecast the first frost date for 2024
forecast = model_fit.forecast(steps=1)
predicted_day_of_year = forecast.iloc[0]
predicted_date = datetime(2024, 1, 1) + timedelta(days=int(predicted_day_of_year) - 1)

print(f"Predicted first frost date for fall 2024: {predicted_date.date()}")

# Optional: Plot the historical and predicted first frost dates
plt.figure(figsize=(12, 6))
plt.plot(time_series.index, time_series.values, label='Historical Data')
plt.scatter(2024, predicted_day_of_year, color='red', label='2024 Prediction')
plt.xlabel('Year')
plt.ylabel('Day of Year')
plt.title('First Frost Date Trend and Prediction using ARIMA')
plt.legend()
plt.show()

from pmdarima import auto_arima

# Fit auto_arima to find the best model
auto_model = auto_arima(time_series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
auto_model.fit(time_series)

# Forecast
forecast = auto_model.predict(n_periods=1)
predicted_day_of_year = forecast[0]
predicted_date = datetime(2024, 1, 1) + timedelta(days=int(predicted_day_of_year) - 1)

print(f"Predicted first frost date for fall 2024 using auto_arima: {predicted_date.date()}")





# import sqlite3
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset

# file_path = '/home/sam/source/idannn/twin-falls-all-data-farm-oiko-1950-2023-corrected.csv'
# data = pd.read_csv(file_path)

# # Print all column names to understand what variables are available
# print("All Variable Names:")
# print(data.columns.tolist())

# # Convert date to datetime object and extract components
# data['date'] = pd.to_datetime(data.iloc[:, 0])
# data['year'] = data['date'].dt.year
# data['month'] = data['date'].dt.month
# data['day'] = data['date'].dt.day
# data['hour'] = data['date'].dt.hour

# import matplotlib.dates as mdates
# import pandas as pd
# import matplotlib.pyplot as plt

# # Convert 'date' column to datetime
# data['date'] = pd.to_datetime(data['date'])

# # Filter for frost conditions in fall months
# frost_data = data[(data['temperature (degC)'] < 0) & data['month'].isin([9, 10, 11])]

# # Find the date of first frost for each year
# first_frost = frost_data.groupby(frost_data['year'])['date'].min()

# # Reset index to make 'year' a column again
# first_frost = first_frost.reset_index()

# # Convert the date of first frost to day of year
# first_frost['day_of_year'] = first_frost['date'].dt.dayofyear

# # Plotting the histogram
# plt.figure(figsize=(10, 6))
# plt.hist(first_frost['day_of_year'], bins=30, edgecolor='black')

# # Customize x-axis to show months
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# # Labeling axes
# plt.xlabel('Month of the first frost')
# plt.ylabel('Frequency')

# # Display the plot
# plt.show()
# # Features and target
# X = data[['hour', 'day', 'month']]
# y = data['temperature (degC)']

# import matplotlib.pyplot as plt

# # Plot temperature over time
# data.plot(x='date', y='temperature (degC)')
# plt.show()

# # Or for a simple histogram of temperatures
# data['temperature (degC)'].hist()
# plt.show()

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Evaluate the model
# y_pred = model.predict(X_test)
# from sklearn.metrics import mean_squared_error, r2_score

# print("Mean squared error:", mean_squared_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))

# # Example prediction for a specific time
# new_data = [[7, 21, 12]]  # Example for 12 PM, day 15, month 7
# predicted_temp = model.predict(new_data)
# print(f"Predicted temperature {predicted_temp[0]}")



# def expected_temp(hour, day, month):
#     subset = data[(data['hour'] == hour) & (data['day'] == day) & (data['month'] == month)]
#     if not subset.empty:
#         return subset['avg_temp'].mean()
#     else:
#         return "No data available for this specific time"

	#print(expected_temp(12, 15, 7))  # Expected average temperature at 12 PM on July 15th

	# def predict
	# # Prepare your data for regression
	# X = data[['hour', 'day', 'month']]  # Features
	# y = data['avg_temp']  # Target variable

	# # Split the data into training and testing sets
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# # Create and train the model
	# model = LinearRegression()
	# model.fit(X_train, y_train)

	# # Predict and evaluate
	# y_pred = model.predict(X_test)
	# print("Model Coefficient:", model.coef_)
	# print("Model Intercept:", model.intercept_)
	# print("Mean squared error:", mean_squared_error(y_test, y_pred))
	# print("R2 Score:", r2_score(y_test, y_pred))

	# # Example prediction
	# new_data = np.array([[12, 15, 7]])  # Hour 12, Day 15, Month 7
	# predicted_temp = model.predict(new_data)
	# print("Predicted temperature:", predicted_temp[0])