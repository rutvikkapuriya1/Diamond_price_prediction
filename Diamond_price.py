

import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the unique values for 'cut', 'color', and 'clarity'
cut_values = ['Ideal', 'Premium', 'Good', 'Very Good', 'Fair']
color_values = ['E', 'I', 'J', 'H', 'F', 'G', 'D']
clarity_values = ['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF']

# Function to perform one-hot encoding
def one_hot_encode(df, column, values):
    for value in values:
        df[column + '_' + value] = (df[column] == value).astype(int)
    df.drop(column, axis=1, inplace=True)

# Function to get user input for prediction
def get_user_input():
    carat = st.slider('Carat', 0.2, 5.01, 0.8)
    cut = st.selectbox('Cut', cut_values)
    color = st.selectbox('Color', color_values)
    clarity = st.selectbox('Clarity', clarity_values)
    depth = st.slider('Depth', 43, 79, 61)
    table = st.slider('Table', 43, 95, 57)
    x = st.slider('x', 0, 10, 5)
    y = st.slider('y', 0, 60, 30)
    z = st.slider('z', 0, 32, 16)
    
    # Create a dictionary with the user input
    data = {
        'carat': carat,
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z
    }
    
    # Convert the dictionary to a DataFrame
    features = pd.DataFrame(data, index=[0])
    
    # Perform one-hot encoding with remove_first option for specific features
    one_hot_encode(features, 'cut', cut_values)
    one_hot_encode(features, 'color', color_values)
    one_hot_encode(features, 'clarity', clarity_values)
    
    # Reorder the columns to match the order used during model training
    features = features[model.feature_names_in_]
    
    return features

# Set the title of the app
st.title('Diamond Price Prediction App')

# Get user input
user_input = get_user_input()

# Make prediction
prediction = model.predict(user_input)
predicted_price = round(prediction[0], 2)

# Display the prediction
st.subheader('Prediction')
st.write('The predicted price of the diamond is $', predicted_price)

# Additional information and data display
st.subheader('User Input Features')
st.write(user_input)


# In[ ]:




