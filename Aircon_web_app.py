import streamlit as st

# Load the pickled models and scaler
model = pickle.load(open('air_conditioner_prices_dataset.pkl', 'rb'))
ss = pickle.load(open('air_conditioner_prices_dataset_ss.pkl', 'rb'))
rf = pickle.load(open('air_conditioner_prices_dataset_rf.pkl', 'rb'))
dt = pickle.load(open('air_conditioner_prices_dataset_dt.pkl', 'rb'))
kn = pickle.load(open('air_conditioner_prices_dataset_kn.pkl', 'rb'))

# Set the title of the web app
st.title('Air Conditioner Price Prediction')

# Get the user input
brand = st.selectbox('Brand', df['Brand'].unique())
type_ = st.selectbox('Type', df['Type'].unique())
features = st.selectbox('Features', df['Features'].unique())
location = st.selectbox('Location', df['Location'].unique())
capacity = st.number_input('Capacity (BTUs)')
eer = st.number_input('EER/SEER')

# Encode the categorical features
brand_encoded = le.transform([brand])[0]
type_encoded = le.transform([type_])[0]
features_encoded = le.transform([features])[0]
location_encoded = le.transform([location])[0]

# Preprocess the user input
input_data = [[brand_encoded, type_encoded, features_encoded, location_encoded, capacity, eer]]
input_data_scaled = ss.transform(input_data)

# Predict the price using different models
lr_price = model.predict(input_data_scaled)[0]
rf_price = rf.predict(input_data_scaled)[0]
dt_price = dt.predict(input_data_scaled)[0]
kn_price = kn.predict(input_data_scaled)[0]

# Display the predicted prices
st.write('Predicted Price (Linear Regression):', lr_price)
st.write('Predicted Price (Random Forest):', rf_price)
st.write('Predicted Price (Decision Tree):', dt_price)
st.write('Predicted Price (K-Neighbors Regressor):', kn_price)