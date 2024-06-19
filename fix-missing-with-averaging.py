import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = './twin-falls-all-data-farm-oiko-1950-2023.csv'
data = pd.read_csv(file_path)

# Convert date to datetime object and extract components
data['date'] = pd.to_datetime(data.iloc[:, 0])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour

key_variables = ['temperature (degC)', 'soil_temperature_level_1 (degC)', 'soil_temperature_level_2 (degC)',
                 'soil_temperature_level_3 (degC)', 'surface_thermal_radiation (W/m^2)', 'surface_solar_radiation (W/m^2)',
                 'direct_normal_solar_radiation (W/m^2)', 'surface_diffuse_solar_radiation (W/m^2)',
                 'relative_humidity (0-1)', 'surface_pressure (Pa)', 'mean_sea_level_pressure (Pa)',
                 'total_cloud_cover (0-1)', 'total_precipitation (mm of water equivalent)',
                 'snowfall (mm of water equivalent)', 'snow_depth (mm of water equivalent)',
                 'snow_density (kg/m^3)', 'volumetric_soil_water_layer_1 (0-1)',
                 'volumetric_soil_water_layer_2 (0-1)', 'volumetric_soil_water_layer_3 (0-1)',
                 'volumetric_soil_water_layer_4 (0-1)', 'wind_speed (m/s)', 'wind_direction (deg)',
                 '10m_wind_gust (m/s)', '100m_wind_speed (m/s)']

# Identify missing data points
missing_data = data.isnull().sum()
print("Missing Data Points Per Column:")
print(missing_data)

# Fill missing values with the mean of each column
data_filled_mean = data.copy()
for column in key_variables:
    data_filled_mean[column].fillna(data[column].mean(), inplace=True)

# Save the corrected dataset
corrected_file_path = './twin-falls-all-data-farm-oiko-1950-2023-corrected.csv'
data_filled_mean.to_csv(corrected_file_path, index=False)
print(f"Corrected data saved to {corrected_file_path}")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = './twin-falls-all-data-farm-oiko-1950-2023.csv'
data = pd.read_csv(file_path)

# Convert date to datetime object and extract components
data['date'] = pd.to_datetime(data.iloc[:, 0])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour

key_variables = ['temperature (degC)', 'soil_temperature_level_1 (degC)', 'soil_temperature_level_2 (degC)',
                 'soil_temperature_level_3 (degC)', 'surface_thermal_radiation (W/m^2)', 'surface_solar_radiation (W/m^2)',
                 'direct_normal_solar_radiation (W/m^2)', 'surface_diffuse_solar_radiation (W/m^2)',
                 'relative_humidity (0-1)', 'surface_pressure (Pa)', 'mean_sea_level_pressure (Pa)',
                 'total_cloud_cover (0-1)', 'total_precipitation (mm of water equivalent)',
                 'snowfall (mm of water equivalent)', 'snow_depth (mm of water equivalent)',
                 'snow_density (kg/m^3)', 'volumetric_soil_water_layer_1 (0-1)',
                 'volumetric_soil_water_layer_2 (0-1)', 'volumetric_soil_water_layer_3 (0-1)',
                 'volumetric_soil_water_layer_4 (0-1)', 'wind_speed (m/s)', 'wind_direction (deg)',
                 '10m_wind_gust (m/s)', '100m_wind_speed (m/s)']

# Identify missing data points
missing_data = data.isnull().sum()
print("Missing Data Points Per Column:")
print(missing_data)

# Fill missing values with the mean of each column
data_filled_mean = data.copy()
for column in key_variables:
    data_filled_mean[column].fillna(data[column].mean(), inplace=True)

# Save the corrected dataset
corrected_file_path = './twin-falls-all-data-farm-oiko-1950-2023-corrected.csv'
data_filled_mean.to_csv(corrected_file_path, index=False)
print(f"Corrected data saved to {corrected_file_path}")
