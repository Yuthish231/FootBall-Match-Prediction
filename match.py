import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load trained models and scaler
rf_model = joblib.load('random_forest.pkl')
svm_model = joblib.load('svm_model.pkl')
nn_model = load_model('neural_model.h5')
scaler = joblib.load('scaler.pkl')  # Load the scaler

# Load the dataset
data = pd.read_csv('matches.csv')

# Title and Description
st.title("Team Match Result Prediction")
st.write("Select two teams and choose your team to predict the match result.")

# Model choice
model_choice = st.selectbox("Choose the model", ["Random Forest", "SVM", "Neural Network"])

# Team selection
team_1 = st.selectbox("Choose Team 1", data['team'].unique())
team_2 = st.selectbox("Choose Team 2", data['team'].unique())

# Ensure different teams are selected
if team_1 == team_2:
    st.error("Please select two different teams.")
else:
    # Choose the team for result prediction
    chosen_team = st.radio("Select your team to predict the result", (team_1, team_2))

    # Display selected team’s features
    team_data = data[data['team'] == chosen_team][['gf', 'ga', 'xg', 'xga', 'poss', 'sh', 'sot', 'dist']].iloc[0]
    st.write(f"Selected Team ({chosen_team}) Stats:")
    st.write(team_data)

    # Extract the features for the chosen team
    input_data = team_data.values.reshape(1, -1)

    # Make prediction
    if st.button("Predict Result"):
        if model_choice == "Random Forest":
            result = rf_model.predict(input_data)
        elif model_choice == "SVM":
            result = svm_model.predict(input_data)
        else:  # Neural Network
            input_data_scaled = scaler.transform(input_data)  # Scale input for neural network
            result = nn_model.predict(input_data_scaled)
            result = np.argmax(result, axis=1)
        
        # Display result
        result_map = {1: "Win", 0: "Loss", 2: "Draw"}
        st.write(f"Predicted Result for {chosen_team}: {result_map.get(result[0], 'Unknown')}")
