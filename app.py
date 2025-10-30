# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

# Updated app.py for Streamlit Deployment

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================================
# 1. STREAMLIT SETUP & TITLE
# ===============================================
st.title("House Price Prediction ML App")
st.subheader("Linear Regression Model")

# ===============================================
# 2. DATA GENERATION (As in your original code)
# ===============================================

# Generate a fake but realistic housing dataset
data = {
    'area': np.random.randint(500, 5000, 200),
    'bedrooms': np.random.randint(1, 5, 200),
    'bathrooms': np.random.randint(1, 3, 200),
    'stories': np.random.randint(1, 4, 200),
    'parking': np.random.randint(0, 3, 200),
    'location_score': np.random.randint(1, 10, 200)
}
df = pd.DataFrame(data)
df['price'] = (
    df['area'] * 120 +  # Area contributes heavily
    df['bedrooms'] * 10000 +
    df['bathrooms'] * 15000 +
    df['stories'] * 5000 +
    df['parking'] * 2000 +
    df['location_score'] * 3000 +
    np.random.randint(-50000, 50000, 200) # Random noise
)

# ===============================================
# 3. DISPLAY DATA (Using st.write)
# ===============================================

st.header("1. Sample Data and Overview")
st.text("First 5 rows of the generated dataset:")
st.dataframe(df.head()) # Use st.dataframe for tables

# Show basic info - only feasible if using a separate function, 
# st.write is used here for descriptions instead of df.info() output
# df.info() 
st.text(f"Dataset Shape: {df.shape}")
st.text(f"Price Statistics:\n{df['price'].describe()}")


# ===============================================
# 4. MODEL TRAINING
# ===============================================

# Plot Price Distribution (using st.pyplot)
st.header("2. Price Distribution")
fig_hist, ax_hist = plt.subplots() # Use subplots for Streamlit
sns.histplot(df['price'], kde=True, ax=ax_hist)
ax_hist.set_title("Price Distribution")
st.pyplot(fig_hist) # Display the plot

# Define features (X) and target (y)
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'location_score']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

st.header("3. Model Training & Evaluation")
st.text(f"Training Samples: {X_train.shape}")
st.text(f"Testing Samples: {X_test.shape}")

# Initialize and Train Model
model = LinearRegression()
model.fit(X_train, y_train)

st.success("Model trained successfully! ✅")

# Predict
y_pred = model.predict(X_test)

# ===============================================
# 5. EVALUATION METRICS (Using st.write)
# ===============================================

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Evaluation Metrics:")
st.write(f"**R2 Score:** `{r2:.4f}`")
st.write(f"**Mean Absolute Error (MAE):** `{mae:,.2f}`")
st.write(f"**Mean Squared Error (MSE):** `{mse:,.2f}`")

# ===============================================
# 6. SCATTER PLOT (Using st.pyplot)
# ===============================================

st.header("4. Actual vs Predicted Prices")

# Create figure and axes explicitly for the scatter plot
fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5)) 

ax_scatter.scatter(y_test, y_pred, color='purple')
ax_scatter.set_xlabel("Actual Prices")
ax_scatter.set_ylabel("Predicted Prices")
ax_scatter.set_title("Actual vs Predicted House Prices")

# Display the figure in Streamlit
st.pyplot(fig_scatter) 

# ===============================================
# 7. INTERACTIVE PREDICTION EXAMPLE
# ===============================================

st.header("5. Try a Prediction")

# Use Streamlit widgets for user input
area = st.slider("Area (sqft)", 500, 5000, 2000)
bedrooms = st.slider("Bedrooms", 1, 5, 2)
bathrooms = st.slider("Bathrooms", 1, 4, 2)
stories = st.slider("Stories", 1, 5, 1)
parking = st.slider("Parking Spaces", 0, 3, 1)
location_score = st.slider("Location Score (1-10)", 1, 10, 5)

# Create the input array
sample = np.array([
    area, bedrooms, bathrooms, stories, parking, location_score
]).reshape(1, -1)

# Predict the price
predicted_price = model.predict(sample)[0]

st.info(f"**Predicted Price for given house:** **`${predicted_price:,.2f}`**")

# ===============================================
# END OF APP
# ===============================================
