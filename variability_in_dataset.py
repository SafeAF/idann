# another throw away script to examine variability
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = './twin-falls-all-data-farm-oiko-1950-2023.csv'
data = pd.read_csv(file_path)

# Print all column names to understand what variables are available
print("All Variable Names:")
print(data.columns.tolist())

# Convert date to datetime object and extract components
data['date'] = pd.to_datetime(data.iloc[:, 0])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour

# Print statistical summary of key variables
# key_variables = ['temperature (degC)', 'soil_temperature_level_1 (degC)', 
#                  'soil_temperature_level_2 (degC)', 'soil_temperature_level_3 (degC)', 
#                  'surface_solar_radiation (W/m^2)']

key_variables = ['total_precipitation (mm of water equivalent)', 'temperature (degC)',
     'snowfall (mm of water equivalent)', 'snow_density (kg/m^3)']




# Identify missing data points
missing_data = data.isnull().sum()
print("Missing Data Points Per Column:")
print(missing_data)

# Visualize missing data
plt.figure(figsize=(12, 8))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()


# print(data[key_variables].describe())

# # Histograms for each key variable
# data[key_variables].hist(bins=30, figsize=(15, 10))
# plt.suptitle('Distribution of Key Variables')
# plt.show()

# # Correlation heatmap
# corr = data[key_variables].corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()


# ['Unnamed: 0', 'coordinates (lat,lon)', 'model (name)', 'model elevation (surface)', 'utc_offset (hrs)',
#  'temperature (degC)', 'soil_temperature_level_1 (degC)', 'soil_temperature_level_2 (degC)',
#   'soil_temperature_level_3 (degC)', 'surface_thermal_radiation (W/m^2)', 'surface_solar_radiation (W/m^2)',
#    'direct_normal_solar_radiation (W/m^2)', 'surface_diffuse_solar_radiation (W/m^2)',
#     'relative_humidity (0-1)', 'surface_pressure (Pa)', 'mean_sea_level_pressure (Pa)',
#      'total_cloud_cover (0-1)', 'total_precipitation (mm of water equivalent)',
#       'snowfall (mm of water equivalent)', 'snow_depth (mm of water equivalent)',
#        'snow_density (kg/m^3)', 'volumetric_soil_water_layer_1 (0-1)',
#         'volumetric_soil_water_layer_2 (0-1)', 'volumetric_soil_water_layer_3 (0-1)',
#          'volumetric_soil_water_layer_4 (0-1)', 'wind_speed (m/s)', 'wind_direction (deg)',
#           '10m_wind_gust (m/s)', '100m_wind_speed (m/s)']
