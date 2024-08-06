import numpy as np
import joblib
import pandas as pd
import streamlit as st
from PIL import Image

# Load the model using joblib
model = joblib.load("gb_model.joblib")

def crash_prediction(latitude, longitude, time_of_day):
    data = {
        'LATITUDE':  latitude,
        'LONGITUDE': longitude,
        'TIME OF DAY': time_of_day
    }

    # Create a DataFrame
    input_data = pd.DataFrame([data])
    print(input_data)
    prediction = model.predict(input_data)
    print(prediction)
    return prediction

# crash_prediction(40.63,-74.09,"Morning")

def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">CRASH PREDICTION</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    latitude = st.number_input("Enter Latitude")
    longitude = st.number_input("Enter Longitude")
    time_of_day = st.text_input("Enter Time of Day")

    # Convert inputs to appropriate numeric types
    latitude = float(latitude)
    longitude = int(longitude)
    

    labels = [
        "NUMBER OF PERSONS INJURED",
        "NUMBER OF PERSONS KILLED",
        "NUMBER OF PEDESTRIANS INJURED",
        "NUMBER OF PEDESTRIANS KILLED",
        "NUMBER OF CYCLIST INJURED",
        "NUMBER OF CYCLIST KILLED",
        "NUMBER OF MOTORIST INJURED",
        "NUMBER OF MOTORIST KILLED"
    ]



    result = ""
    if st.button("Predict"):
        result = crash_prediction(latitude, longitude, time_of_day)
        if isinstance(result, np.ndarray) and len(result) > 0:
            formatted_result = {label: round(float(value), 2) for label, value in zip(labels, result[0])}
            st.success(f'The predicted values are:')
            for label, value in formatted_result.items():
                st.write(f'{label}: {value}')
        else:
            st.error('Prediction failed. Please check the input values or model.')

if __name__ == '__main__':
    main()


