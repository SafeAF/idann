# random forest model
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import shap

# Read the CSV file into a DataFrame
df = pd.read_csv('twin-falls-all-data-farm-oiko-1950-2023-corrected.csv')

# Ensure that 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Extract the year from the 'date' column
df['year'] = df['date'].dt.year

# Add a 'day_of_year' column
df['day_of_year'] = df['date'].dt.dayofyear

# Filter data for fall months (August to December)
df_fall = df[df['date'].dt.month >= 8]

# Extract First Frost Dates
first_frost_dates = []

for year, group in df_fall.groupby('year'):
    # Sort by date
    group = group.sort_values('date')
    # Find the first date when temperature drops below 0°C
    frost_days = group[group['temperature (degC)'] <= 0]
    if not frost_days.empty:
        first_frost_date = frost_days.iloc[0]['date']
        first_frost_dates.append({'year': year, 'first_frost_date': first_frost_date})
    else:
        # If no frost found in fall months, skip the year
        continue

# Create a DataFrame from the list
frost_df = pd.DataFrame(first_frost_dates)
frost_df['first_frost_day_of_year'] = frost_df['first_frost_date'].dt.dayofyear

# Feature Engineering
pre_frost_months = [6, 7, 8, 9]
df_pre_frost = df[df['date'].dt.month.isin(pre_frost_months)]
features_list = []

for year in frost_df['year']:
    # Get the first frost date for the year
    first_frost_date = frost_df.loc[frost_df['year'] == year, 'first_frost_date'].values[0]
    # Filter data up to the first frost date
    df_year = df_pre_frost[(df_pre_frost['year'] == year) & (df_pre_frost['date'] < first_frost_date)]
    if df_year.empty:
        continue
    # Compute aggregate features
    feature_row = {'year': year}
    # Temperature features
    feature_row['mean_temp'] = df_year['temperature (degC)'].mean()
    feature_row['max_temp'] = df_year['temperature (degC)'].max()
    feature_row['min_temp'] = df_year['temperature (degC)'].min()
    # Soil temperatures
    feature_row['mean_soil_temp_1'] = df_year['soil_temperature_level_1 (degC)'].mean()
    feature_row['mean_soil_temp_2'] = df_year['soil_temperature_level_2 (degC)'].mean()
    feature_row['mean_soil_temp_3'] = df_year['soil_temperature_level_3 (degC)'].mean()
    # Radiation features
    feature_row['mean_solar_radiation'] = df_year['surface_solar_radiation (W/m^2)'].mean()
    feature_row['mean_thermal_radiation'] = df_year['surface_thermal_radiation (W/m^2)'].mean()
    # Atmospheric features
    feature_row['mean_cloud_cover'] = df_year['total_cloud_cover (0-1)'].mean()
    feature_row['mean_relative_humidity'] = df_year['relative_humidity (0-1)'].mean()
    feature_row['mean_wind_speed'] = df_year['wind_speed (m/s)'].mean()
    feature_row['mean_surface_pressure'] = df_year['surface_pressure (Pa)'].mean()
    # Append the features
    features_list.append(feature_row)

# Create a DataFrame from the features list
features_df = pd.DataFrame(features_list)

# Merge with the frost_df to get the target variable
data = pd.merge(frost_df[['year', 'first_frost_day_of_year']], features_df, on='year')

# Prepare the data
feature_cols = [col for col in data.columns if col not in ['year', 'first_frost_day_of_year']]
X = data[['year'] + feature_cols]
y = data['first_frost_day_of_year']

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Split the data
X_train = X[X['year'] < 2020].copy()
X_test = X[X['year'] >= 2020].copy()
y_train = y[X['year'] < 2020]
y_test = y[X['year'] >= 2020]
X_train = X_train.drop('year', axis=1)
X_test = X_test.drop('year', axis=1)

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error on Test Set: {mae:.2f} days")

# Results
results = pd.DataFrame({'Year': X[X['year'] >= 2020]['year'], 'Actual DOY': y_test, 'Predicted DOY': y_pred})
print(results)

# Feature Importance
importances = rf_model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances from Random Forest Model')
plt.show()

# SHAP Values
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")

# Predict for 2024
mean_features = X_train.mean()
X_2024 = mean_features.to_frame().T
X_2024['year'] = 2024
X_2024 = X_2024[X_train.columns]
predicted_day_of_year_2024 = rf_model.predict(X_2024)[0]
predicted_date_2024 = datetime(2024, 1, 1) + timedelta(days=int(predicted_day_of_year_2024) - 1)
print(f"Predicted first frost date for fall 2024: {predicted_date_2024.date()}")
