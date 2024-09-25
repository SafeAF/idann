# linear regression to determine date of first frost in the fall
# Ans: October 15

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm

# Read the CSV file into a DataFrame
df = pd.read_csv('twin-falls-all-data-farm-oiko-1950-2023-corrected.csv')

# Ensure that 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Add a 'day_of_year' column for easier processing
df['day_of_year'] = df['date'].dt.dayofyear

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

# Convert dates to day of year for regression analysis
frost_df['day_of_year'] = frost_df['date'].dt.dayofyear

# Prepare data for linear regression
X = frost_df['year']
y = frost_df['day_of_year']
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Fit the regression model
model = sm.OLS(y, X).fit()

# Predict the first frost date for 2024
predicted_day_of_year = model.predict([1, 2024])[0]
predicted_date = datetime(2024, 1, 1) + timedelta(days=predicted_day_of_year - 1)

print(f"Predicted first frost date for fall 2024: {predicted_date.date()}")

# Optional: Plot the historical and predicted first frost dates
plt.scatter(frost_df['year'], frost_df['day_of_year'], label='Historical Data')
plt.plot(frost_df['year'], model.predict(X), color='red', label='Trend Line')
plt.scatter(2024, predicted_day_of_year, color='green', label='2024 Prediction')
plt.xlabel('Year')
plt.ylabel('Day of Year')
plt.title('First Frost Date Trend and Prediction')
plt.legend()
plt.show()