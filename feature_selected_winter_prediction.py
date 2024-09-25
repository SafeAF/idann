# winter prediction with feature selection

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats

# Load the data
df = pd.read_csv('twin-falls-all-data-farm-oiko-1950-2023-corrected.csv', parse_dates=['date'])

# Extract time features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour

# Adjust year for December to include in the next year's winter
df['winter_year'] = df['year']
df.loc[df['month'] == 12, 'winter_year'] += 1

# Filter data for winter months
df_winter = df[df['month'].isin([12, 1, 2])]

# Select relevant columns
columns_of_interest = [
    'winter_year',
    'temperature (degC)',
    'soil_temperature_level_1 (degC)',
    'soil_temperature_level_2 (degC)',
    'soil_temperature_level_3 (degC)',
    'surface_thermal_radiation (W/m^2)',
    'surface_solar_radiation (W/m^2)',
    'relative_humidity (0-1)',
    'surface_pressure (Pa)',
    'mean_sea_level_pressure (Pa)',
    'total_cloud_cover (0-1)',
    'wind_speed (m/s)',
    'wind_direction (deg)',
    'total_precipitation (mm of water equivalent)',
    'snowfall (mm of water equivalent)',
    'snow_depth (mm of water equivalent)',
    'snow_density (kg/m^3)',
    'volumetric_soil_water_layer_1 (0-1)',
    'volumetric_soil_water_layer_2 (0-1)',
    'volumetric_soil_water_layer_3 (0-1)',
    'volumetric_soil_water_layer_4 (0-1)'
]

# Create a DataFrame with the selected columns
winter_features = df_winter[columns_of_interest].groupby('winter_year').mean().reset_index()

# Handle missing values
winter_features.fillna(winter_features.mean(), inplace=True)

# Define targets
y_temp = winter_features['temperature (degC)']
y_precip = winter_features['total_precipitation (mm of water equivalent)']

# Prepare feature matrix
X = winter_features.drop(['winter_year', 'temperature (degC)', 'total_precipitation (mm of water equivalent)'], axis=1)

# Feature importance using Random Forest
X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
X_train, X_test, y_precip_train, y_precip_test = train_test_split(X, y_precip, test_size=0.2, random_state=42)

rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_precip = RandomForestRegressor(n_estimators=100, random_state=42)

rf_temp.fit(X_train, y_temp_train)
rf_precip.fit(X_train, y_precip_train)

# Get feature importances
importance_temp = rf_temp.feature_importances_
importance_precip = rf_precip.feature_importances_

feature_names = X.columns
temp_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_temp})
precip_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_precip})

temp_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
precip_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Select most significant features
features_temp = temp_importance_df['Feature'].head(5).tolist()
features_precip = precip_importance_df['Feature'].head(5).tolist()

# Prepare final feature sets
X_temp = winter_features[features_temp]
X_precip = winter_features[features_precip]

# Split data into training and testing sets
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
X_precip_train, X_precip_test, y_precip_train, y_precip_test = train_test_split(X_precip, y_precip, test_size=0.2, random_state=42)

# Train the models
rf_temp_final = RandomForestRegressor(n_estimators=100, random_state=42)
rf_precip_final = RandomForestRegressor(n_estimators=100, random_state=42)

rf_temp_final.fit(X_temp_train, y_temp_train)
rf_precip_final.fit(X_precip_train, y_precip_train)

# Evaluate the models
y_temp_pred = rf_temp_final.predict(X_temp_test)
mae_temp = mean_absolute_error(y_temp_test, y_temp_pred)
r2_temp = r2_score(y_temp_test, y_temp_pred)
print(f"Temperature Prediction - MAE: {mae_temp:.2f}°C, R²: {r2_temp:.2f}")

y_precip_pred = rf_precip_final.predict(X_precip_test)
mae_precip = mean_absolute_error(y_precip_test, y_precip_pred)
r2_precip = r2_score(y_precip_test, y_precip_pred)
print(f"Precipitation Prediction - MAE: {mae_precip:.2f} mm, R²: {r2_precip:.2f}")

# Prepare input data for prediction
recent_years = winter_features.tail(3)
X_temp_new = recent_years[features_temp].mean().to_frame().T
X_precip_new = recent_years[features_precip].mean().to_frame().T

# Make predictions
predicted_avg_temp = rf_temp_final.predict(X_temp_new)[0]
predicted_total_precip = rf_precip_final.predict(X_precip_new)[0]

print(f"Predicted Average Winter Temperature: {predicted_avg_temp:.2f}°C")
print(f"Predicted Total Winter Precipitation: {predicted_total_precip:.2f} mm")

# Compare predictions with historical data
historical_avg_temp_mean = y_temp.mean()
historical_avg_temp_std = y_temp.std()
historical_precip_mean = y_precip.mean()
historical_precip_std = y_precip.std()

# Z-scores
z_score_temp = (predicted_avg_temp - historical_avg_temp_mean) / historical_avg_temp_std
z_score_precip = (predicted_total_precip - historical_precip_mean) / historical_precip_std

# Percentile ranks
percentile_rank_temp = stats.percentileofscore(y_temp, predicted_avg_temp)
percentile_rank_precip = stats.percentileofscore(y_precip, predicted_total_precip)

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
