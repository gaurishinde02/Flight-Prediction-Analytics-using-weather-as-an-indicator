import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# Load the Random Forest Classifier model for initial delay prediction
rfc_model_filename = 'rfc_smote.pkl'  # Update this path
with open(rfc_model_filename, 'rb') as file:
    rfc_model = pickle.load(file)


# Load the trained model
model_filename = 'xgb_new.pkl'  # Update this path if your model is saved in a different location
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

encoders_filename = 'label_encoders (1).pkl'
with open(encoders_filename, 'rb') as file:
    label_encoders = pickle.load(file)

# Define a function to safely encode using LabelEncoder, handling unseen labels
def safe_encode(le, categories):
    le_classes = le.classes_.tolist()
    process_unknown = lambda x: le_classes.index(x) if x in le_classes else -1
    return [process_unknown(x) for x in categories]

# Creating a simple form for input
st.title('Flight Delay Prediction App')
# Add input fields for the features your model uses. Here's an example:
# Add input fields for the features your model uses. Here's an example:
day_of_month = st.number_input('Day of Month', min_value=1, max_value=31, value=13)
day_of_week = st.number_input('Day of Week', min_value=1, max_value=7, value=1)
marketing_airline_network = st.text_input('Marketing Airline Network', value='B6')
dot_id_marketing_airline = st.number_input('DOT ID Marketing Airline', value=20409)
operating_airline = st.text_input('Operating Airline', value='B6')
origin_airport_id = st.number_input('Origin Airport ID', value=12953)
origin = st.text_input('Origin', value='LGA')
dest = st.text_input('Destination', value='PBI')
dest_state = st.text_input('Destination State', value='WA')
crs_dep_time = st.number_input('CRS Departure Time', value=1600)
dep_time_blk = st.text_input('Departure Time Block', value='1800-1859')
air_time = st.number_input('Air Time', value=45)
distance = st.number_input('Distance', value=182)
dep_count = st.number_input('Departure Count', value=51)
congestion = st.number_input('Congestion', value=69)
snow_depth = st.number_input('Snow Depth (m)', value=0)
surface_pressure = st.number_input('Surface Pressure (hPa)', value=0.5)
weather_code = st.number_input('Weather Code (WMO code)', value=0)
cloud_cover_mid = st.number_input('Cloud Cover Mid (%)', value=0)
relative_humidity_2m = st.number_input('Relative Humidity 2m (%)', value=0)


# Add more inputs as needed...

# When the user submits the inputs
if st.button('Predict Delay'):
    # Collect inputs into a DataFrame
    input_data = pd.DataFrame({
        'DayofMonth': [day_of_month],
        'DayOfWeek': [day_of_week],
        'Marketing_Airline_Network': [marketing_airline_network],
        'DOT_ID_Marketing_Airline': [dot_id_marketing_airline],
        'Operating_Airline ': [operating_airline],
        'OriginAirportID': [origin_airport_id],
        'Origin': [origin],
        'Dest': [dest],
        'DestState': [dest_state],
        'CRSDepTime': [crs_dep_time],
        'DepTimeBlk': [dep_time_blk],
        'AirTime': [air_time],
        'Distance': [distance],
        'DepCount': [dep_count],
        'Congestion': [congestion],
        'snow_depth (m)': [snow_depth],
        'surface_pressure (hPa)': [surface_pressure],
        'weather_code (wmo code)': [weather_code],
        'cloud_cover_mid (%)': [cloud_cover_mid],
        'relative_humidity_2m (%)': [relative_humidity_2m]
    })

    # Encode using the loaded label encoders, handling unseen categories
    categorical_columns = ['Marketing_Airline_Network', 'Operating_Airline ', 'Origin', 'Dest', 'DestState',
                           'DepTimeBlk']

    for column in categorical_columns:
        if column in input_data.columns:
            input_data[column] = safe_encode(label_encoders[column], input_data[column])

    # First, predict if there is a delay using the RFC model
    rfc_prediction = rfc_model.predict(input_data)[0]

    if rfc_prediction == 1:  # If RFC predicts a delay, use the XGBoost model to predict delay duration
        delay_duration = model.predict(input_data)[0]
        delay_duration_formatted = "{:.2f}".format(delay_duration)  # Format to 2 decimal places

        if delay_duration > 0:
            st.markdown(f"## ✈️ Predicted Delay Duration: **{delay_duration_formatted} minutes**")
            st.error("Sadly, your flight could be delayed. 😢 Please check with your airline for more information.")
            st.snow()  # Display snow animation to indicate delay
        else:
            st.markdown("## ✈️ Your flight is on time! **No delay expected.**")
            st.success("Wonderful! Your flight is expected to depart as scheduled! 🎉")
            st.balloons()  # Show balloons animation for a positive outcome
    else:
        st.markdown("## ✈️ Your flight is on time! **No delay expected.**")
        st.success("Fantastic! It looks like you'll be taking off on time! ✅")
        st.balloons()  # Show balloons animation for a positive outcome



