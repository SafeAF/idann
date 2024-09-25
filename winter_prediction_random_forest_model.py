import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats

# Load data
df = pd.read_csv('twin-falls-all-data-farm-oiko-1950-2023-corrected.csv')
df['date'] = pd.to_datetime(df['date'])

# Extract time features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Adjust year for December to include in the next year's winter
df['winter_year'] = df['year']
df.loc[df['month'] == 12, 'winter_year'] += 1

# Filter data for winter months
df_winter = df[df['month'].isin([12, 1, 2])]

# Aggregate winter statistics
winter_stats = df_winter.groupby('winter_year').agg({
    'temperature (degC)': ['mean', 'min', 'max'],
    'total_precipitation (mm of water equivalent)': 'sum'
}).reset_index()
winter_stats.columns = ['winter_year', 'avg_temp', 'min_temp', 'max_temp', 'total_precipitation']

# Create lag features
winter_stats = winter_stats.sort_values('winter_year').reset_index(drop=True)
winter_stats['prev_avg_temp'] = winter_stats['avg_temp'].shift(1)
winter_stats['prev_total_precipitation'] = winter_stats['total_precipitation'].shift(1)
winter_stats = winter_stats.dropna().reset_index(drop=True)

# Features and targets
X = winter_stats[['prev_avg_temp', 'prev_total_precipitation']]
y_temp = winter_stats['avg_temp']
y_precip = winter_stats['total_precipitation']

# Split data into training and testing sets
train_years = winter_stats[winter_stats['winter_year'] < 2020]
test_years = winter_stats[winter_stats['winter_year'] >= 2020]

X_train = train_years[['prev_avg_temp', 'prev_total_precipitation']]
y_train_temp = train_years['avg_temp']
y_train_precip = train_years['total_precipitation']

X_test = test_years[['prev_avg_temp', 'prev_total_precipitation']]
y_test_temp = test_years['avg_temp']
y_test_precip = test_years['total_precipitation']

# Handle missing values
X_train = X_train.fillna(method='ffill')
X_test = X_test.fillna(method='ffill')

# Train Random Forest models
rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_precip = RandomForestRegressor(n_estimators=100, random_state=42)

rf_temp.fit(X_train, y_train_temp)
rf_precip.fit(X_train, y_train_precip)

# Evaluate Temperature Model
y_pred_temp = rf_temp.predict(X_test)
mae_temp = mean_absolute_error(y_test_temp, y_pred_temp)
r2_temp = r2_score(y_test_temp, y_pred_temp)
print(f"Temperature Prediction - MAE: {mae_temp:.2f}°C, R²: {r2_temp:.2f}")

# Evaluate Precipitation Model
y_pred_precip = rf_precip.predict(X_test)
mae_precip = mean_absolute_error(y_test_precip, y_pred_precip)
r2_precip = r2_score(y_test_precip, y_pred_precip)
print(f"Precipitation Prediction - MAE: {mae_precip:.2f} mm, R²: {r2_precip:.2f}")

# Predict for Upcoming Winter
last_avg_temp = winter_stats['avg_temp'].iloc[-1]
last_total_precipitation = winter_stats['total_precipitation'].iloc[-1]

X_new = pd.DataFrame({
    'prev_avg_temp': [last_avg_temp],
    'prev_total_precipitation': [last_total_precipitation]
})

predicted_avg_temp = rf_temp.predict(X_new)[0]
predicted_total_precip = rf_precip.predict(X_new)[0]

print(f"Predicted Average Winter Temperature for Winter 2024-2025: {predicted_avg_temp:.2f}°C")
print(f"Predicted Total Winter Precipitation for Winter 2024-2025: {predicted_total_precip:.2f} mm")

# Calculate historical statistics
historical_avg_temp_mean = winter_stats['avg_temp'].mean()
historical_avg_temp_std = winter_stats['avg_temp'].std()
historical_precip_mean = winter_stats['total_precipitation'].mean()
historical_precip_std = winter_stats['total_precipitation'].std()

# Z-scores
z_score_temp = (predicted_avg_temp - historical_avg_temp_mean) / historical_avg_temp_std
z_score_precip = (predicted_total_precip - historical_precip_mean) / historical_precip_std

# Percentile ranks
percentile_rank_temp = stats.percentileofscore(winter_stats['avg_temp'], predicted_avg_temp)
percentile_rank_precip = stats.percentileofscore(winter_stats['total_precipitation'], predicted_total_precip)

# Classification
if predicted_avg_temp < historical_avg_temp_mean - historical_avg_temp_std:
    temp_classification = 'colder than average'
elif predicted_avg_temp > historical_avg_temp_mean + historical_avg_temp_std:
    temp_classification = 'warmer than average'
else:
    temp_classification = 'about average'

if predicted_total_precip < historical_precip_mean - historical_precip_std:
    precip_classification = 'drier than average'
elif predicted_total_precip > historical_precip_mean + historical_precip_std:
    precip_classification = 'wetter than average'
else:
    precip_classification = 'about average'

print(f"The predicted average temperature is {predicted_avg_temp:.2f}°C, which is {z_score_temp:.2f} standard deviations from the historical mean ({historical_avg_temp_mean:.2f}°C).")
print(f"This suggests the upcoming winter will be {temp_classification}.")

print(f"The predicted total precipitation is {predicted_total_precip:.2f} mm, which is {z_score_precip:.2f} standard deviations from the historical mean ({historical_precip_mean:.2f} mm).")
print(f"This suggests the upcoming winter will be {precip_classification}.")

print(f"The predicted average temperature is at the {percentile_rank_temp:.1f}th percentile of historical data.")
print(f"The predicted total precipitation is at the {percentile_rank_precip:.1f}th percentile of historical data.")
