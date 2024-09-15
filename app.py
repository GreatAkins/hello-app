# Import necessary libraries
import streamlit as st  # For creating interactive web applications
import joblib  # For saving and loading Python objects like machine learning models
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations and handling arrays
import datetime  # For working with date and time
import matplotlib.pyplot as plt  # For creating visualizations
from scipy import stats  # For statistical functions and tests
import statsmodels.api as sm  # For statistical modeling and hypothesis testing
from geopy.geocoders import Nominatim  # For geocoding and location-based queries
import geopandas as gpd  # For geospatial data analysis
from shapely.geometry import Point  # For geometric operations on points
from sklearn.neighbors import NearestNeighbors  # For nearest neighbors algorithms

# Load the scaler and model once
if 'scaler' not in st.session_state:
    st.session_state.scaler = joblib.load('scaler.pkl')
if 'model' not in st.session_state:
    st.session_state.model = joblib.load('best_random_forest_model.joblib')

# Initialize session state to track form submission
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

# Title of the application
st.title('Property Price Valuation Tool')

# Description
st.write("_Welcome to the Property Price Valuation Tool!_")
st.write("_Here you can predict property prices based on various factors and view detailed property data on an interactive map._")

# Function to convert postcode to latitude and longitude
def get_lat_long(postcode):
    """
    Retrieve the latitude and longitude for a given postcode using the Nominatim geocoding service.

    Parameters:
    postcode (str): The postcode for which latitude and longitude are to be retrieved.

    Returns:
    tuple: A tuple containing latitude and longitude if successful, (None, None) if the geocoding fails.
    """
    geolocator = Nominatim(user_agent="property_price_app")
    
    try:
        # Attempt to geocode the postcode
        location = geolocator.geocode(postcode)
        
        if location:
            # Return latitude and longitude if the location was found
            return location.latitude, location.longitude
        else:
            # Return (None, None) if no location was found
            return None, None

    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred: {e}")
        return None, None

# Load the shapefile
flood_zones = gpd.read_file('merseyside flood zones.shp')

# Function to check if a point is in a flood zone
def get_flood_risk(latitude, longitude, flood_zones):
    """
    Determine if a given latitude and longitude falls within any flood risk zones.

    Parameters:
    latitude (float): Latitude of the location to check.
    longitude (float): Longitude of the location to check.
    flood_zones (GeoDataFrame): A GeoDataFrame containing flood risk zones as geometries.

    Returns:
    bool: True if the point is within any flood risk zone, False otherwise.
    """
    try:
        # Create a Point object for the given latitude and longitude
        point = Point(longitude, latitude)

        # Check if the point is within any of the flood risk zones
        for zone in flood_zones.geometry:
            if zone.contains(point):
                return True
        return False

    except AttributeError:
        # Handle cases where flood_zones might not have the 'geometry' attribute
        print("Error: The flood_zones GeoDataFrame must have a 'geometry' column.")
        return False

    except Exception as e:
        # Handle other exceptions that may occur
        print(f"An error occurred: {e}")
        return False

# Load the deprivation data
imd_data = pd.read_csv('final_property_with_deprivation.csv')
df = pd.read_csv('comprehensive_property_data.csv')
df2 = df

# Ensure the DataFrame contains only the necessary columns for mapping
map_data = df[['latitude', 'longitude']]

# Display the map with markers for each property
st.write("### 🏠Property Locations in Merseyside")
st.map(map_data)
X = imd_data[['latitude', 'longitude']].values

# Initialize the Nearest Neighbors model
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(X)

# Function to get socioeconomic factors
def get_socioeconomic_factors(latitude, longitude, year, imd_data, neigh):
    """
    Retrieve socioeconomic factors for a given latitude and longitude based on the nearest LSOA (Lower Layer Super Output Area).

    Parameters:
    latitude (float): Latitude of the location to analyze.
    longitude (float): Longitude of the location to analyze.
    year (int): The year for which socioeconomic factors are requested.
    imd_data (pd.DataFrame): DataFrame containing IMD (Index of Multiple Deprivation) data with columns for socio-economic factors.
    neigh (NearestNeighbors): A fitted NearestNeighbors model to find the nearest LSOA.

    Returns:
    dict: A dictionary containing the socioeconomic factors (IMD score, crime rate, education score, health score, living environment score, income score).
    """
    try:
        # Find the nearest LSOA to the given latitude and longitude
        _, indices = neigh.kneighbors([[latitude, longitude]])
        nearest_lsoa = imd_data.iloc[indices[0][0]]
        
        # Determine the relevant year for the IMD data
        relevant_year = 2015 if year <= 2016 else 2019

        # Filter the IMD data to get the relevant information for the nearest LSOA and year
        relevant_imd_data = imd_data[
            (imd_data['geo_code'] == nearest_lsoa['geo_code']) & 
            (imd_data['year'] == relevant_year)
        ]
        
        if not relevant_imd_data.empty:
            # Extract and return the socioeconomic factors
            return {
                "imd_score": relevant_imd_data.iloc[0]['Index of Multiple Deprivation (IMD) Score'],
                "crime_rate": relevant_imd_data.iloc[0]['Crime Score'],
                "education_score": relevant_imd_data.iloc[0]['Education, Skills and Training Score'],
                "health_score": relevant_imd_data.iloc[0]['Health Deprivation and Disability Score'],
                "living_environment_score": relevant_imd_data.iloc[0]['Living Environment Score'],
                "income_score": relevant_imd_data.iloc[0]['Income Score (rate)']
            }
        
        # Return default values if no relevant IMD data is found
        return {
            "imd_score": None,
            "crime_rate": None,
            "education_score": None,
            "health_score": None,
            "living_environment_score": None,
            "income_score": None
        }

    except KeyError as e:
        # Handle cases where expected columns are not found in the DataFrame
        print(f"Error: Missing column in IMD data - {e}")
        return {
            "imd_score": None,
            "crime_rate": None,
            "education_score": None,
            "health_score": None,
            "living_environment_score": None,
            "income_score": None
        }

    except Exception as e:
        # Handle other unexpected errors
        print(f"An unexpected error occurred: {e}")
        return {
            "imd_score": None,
            "crime_rate": None,
            "education_score": None,
            "health_score": None,
            "living_environment_score": None,
            "income_score": None
        }

# Form for property details input
with st.form(key='property_form'):
    postcode = st.text_input("Postcode")
    duration = st.selectbox("Lease Type", ['Freehold', 'Leasehold'])
    property_type = st.selectbox("Property Type", ['Detached', 'Semi-detached', 'Flat', 'Terraced'])
    condition = st.selectbox("Property Condition", ['New', 'Old'])
    year = st.number_input("Year of Transfer", min_value=2000, max_value=2030, value=datetime.datetime.now().year)
    submit_button = st.form_submit_button(label='Submit')

# Single button for submission and prediction
if submit_button:
    # Process the form submission
    latitude, longitude = get_lat_long(postcode)
    flood_risk = get_flood_risk(latitude, longitude, flood_zones)
    
    # Fetch socioeconomic factors based on location and year
    socioeconomic_factors = get_socioeconomic_factors(latitude, longitude, year, imd_data, neigh)
    
    # Store the results in session state
    st.session_state.latitude = latitude
    st.session_state.longitude = longitude
    st.session_state.flood_risk = flood_risk
    st.session_state.socioeconomic_factors = socioeconomic_factors
    st.session_state.form_submitted = True  # Mark form as submitted

if st.session_state.form_submitted:
    st.write(f"Flood Risk: {'Yes' if st.session_state.flood_risk else 'No'}")
    
    # Retrieve socioeconomic factors from session state
    socioeconomic_factors = st.session_state.socioeconomic_factors

    # Editable fields for socioeconomic factors
    imd_score = st.number_input("IMD Score", value=socioeconomic_factors['imd_score'] if socioeconomic_factors['imd_score'] is not None else 45.0)
    crime_rate = st.number_input("Crime Rate", value=socioeconomic_factors['crime_rate'] if socioeconomic_factors['crime_rate'] is not None else 2.0)
    education_score = st.number_input("Education Score", value=socioeconomic_factors['education_score'] if socioeconomic_factors['education_score'] is not None else 50.0)
    health_score = st.number_input("Health Score", value=socioeconomic_factors['health_score'] if socioeconomic_factors['health_score'] is not None else 3.0)
    living_environment_score = st.number_input("Living Environment Score", value=socioeconomic_factors['living_environment_score'] if socioeconomic_factors['living_environment_score'] is not None else 0.0)
    income_score = st.number_input("Income Score", value=socioeconomic_factors['income_score'] if socioeconomic_factors['income_score'] is not None else 40.0)

    if st.button("Predict Property Price"):

       # Feature engineering or preprocessing
        property_type_dict = {'Detached': 0, 'Semi-detached': 1, 'Terraced': 2, 'Flat': 3}
        duration_dict = {'Freehold': 0, 'Leasehold': 1}
        duration_encoded = duration_dict[duration]
        property_type_encoded = property_type_dict[property_type]
        old_new_encoded = 1 if condition == 'New' else 0
        flood_risk_encoded = 1 if st.session_state.flood_risk else 0

        # Interaction Terms
        imd_propertytype = imd_score * property_type_encoded
        floodzone_latitude = flood_risk_encoded * st.session_state.latitude
        
        # Combine all features into an array
        features = np.array([[property_type_encoded, old_new_encoded, duration_encoded, st.session_state.latitude, st.session_state.longitude, year, 
                              flood_risk_encoded, imd_score, income_score, education_score, health_score, 
                              crime_rate, living_environment_score, imd_propertytype, floodzone_latitude]])

        # Scaling numerical features
        numerical_indices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        features[:, numerical_indices] = st.session_state.scaler.transform(features[:, numerical_indices])

        # Predict the normalized log price
        normalized_log_prediction = st.session_state.model.predict(features)[0]

        # Access the scaler from st.session_state
        scaler = st.session_state.scaler

        # Identify the index of the 'log_price' feature in the scaler's feature set
        log_price_index = list(scaler.feature_names_in_).index('log_price')

        # Reverse the normalization
        original_log_prediction = (normalized_log_prediction * scaler.scale_[log_price_index]) + scaler.mean_[log_price_index]

        # Reverse the log transformation to get the actual price
        predicted_price = np.exp(original_log_prediction)

        # Display the estimated property price
        st.write(f"The estimated property price is: £{predicted_price:,.2f}")
        
        # Optionally reset session state to allow new predictions
        st.session_state.form_submitted = False

# Visualization section
st.subheader('Model Insights & Data Visualizations')

# plots or additional visual insights
if st.checkbox('Show Feature Importance'):
    # Load or calculate feature importances
    importances = st.session_state.model.feature_importances_
    feature_names = ['property_type_encoded', 'old_new_encoded', 'duration_encoded', 'latitude', 'longitude', 'year', 
                     'flood_risk_encoded', 'imd_score', 'income_score', 'education_score', 'health_score', 
                     'crime_rate', 'living_environment_score', 'imd_propertytype', 'floodzone_latitude']
    
    # Convert to a DataFrame for easier plotting
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    st.bar_chart(importance_df.set_index('Feature'))

# Average Price Over Time
st.subheader("Average Property Price Over Time")
average_price_by_year = df.groupby('year')['PRICE'].mean()

# Plotting the average price over time
fig, ax = plt.subplots()
ax.plot(average_price_by_year.index, average_price_by_year.values)
ax.set_xlabel('Year')
ax.set_ylabel('Average Price (£)')
ax.set_title('Average Property Price Over Time')
st.pyplot(fig)

# Temporal Trends
st.subheader(" Temporal Trends")
df['DATE OF TRANSFER'] = pd.to_datetime(df['DATE OF TRANSFER'], format="%d/%m/%Y")
df.set_index('DATE OF TRANSFER', inplace=True)

# Filter out the outliers
df = df[(np.abs(stats.zscore(df['PRICE'])) < 3)]
df = df.sort_index()
df = df['PRICE'].ffill()

# Decompose the time series
decomposition = sm.tsa.seasonal_decompose(df, model='multiplicative', period=12)

# Plotting the decomposed components
# Set up the figure and axes
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Plot the Trend
decomposition.trend.plot(ax=ax1, color='blue')
ax1.set_title('Trend')
ax1.set_xlabel('')

# Plot the Seasonality without x-axis labels
decomposition.seasonal.plot(ax=ax2, color='green')
ax2.set_title('Seasonality')
ax2.set_xlabel('')

# Plot the Residuals
decomposition.resid.plot(ax=ax3, color='red')
ax3.set_title('Residuals')
ax3.set_xlabel('')

# Improve spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Show the plot
st.pyplot(fig)

# View Property Data
st.subheader("View Property Data")

# Display the DataFrame as a table 
st.write("### Property Data Table")
st.write(df2)

st.header("About")
st.write("""
This tool is developed to help users predict property prices based on various factors such as location, crime rates, flood risks, and more. 

The project is conducted as part of an academic initiative to explore the integration of spatial data and machine learning for property price prediction.
""")