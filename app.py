import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Set Page Config
st.set_page_config(page_title="Real Estate ROI Predictor", layout="wide", page_icon="🏡")

# 1. Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Artifacts
@st.cache_resource
def load_models():
    reg = pickle.load(open('reg_model.pkl', 'rb'))
    clf = pickle.load(open('clf_model.pkl', 'rb'))
    enc = pickle.load(open('encoders.pkl', 'rb'))
    feats = pickle.load(open('model_features.pkl', 'rb'))
    return reg, clf, enc, feats

try:
    reg_model, clf_model, encoders, model_features = load_models()
except FileNotFoundError:
    st.error("Model files not found! Please run the Jupyter Notebook first to generate .pkl files.")
    st.stop()

# 3. Sidebar - User Inputs
st.sidebar.header("📍 Property Location & Type")
state = st.sidebar.selectbox("State", encoders['State'].classes_)
city = st.sidebar.selectbox("City", encoders['City'].classes_)
prop_type = st.sidebar.selectbox("Property Type", encoders['Property_Type'].classes_)

st.sidebar.header("🏗️ Property Specifications")
sqft = st.sidebar.number_input("Size (SqFt)", min_value=100, max_value=10000, value=1500)
bhk = st.sidebar.slider("BHK", 1, 5, 3)
price = st.sidebar.number_input("Current Price (Lakhs)", min_value=1.0, value=75.0)
year_built = st.sidebar.number_input("Year Built", min_value=1990, max_value=2024, value=2015)

st.sidebar.header("🚍 Amenities & Infrastructure")
transport = st.sidebar.selectbox("Public Transport Access", encoders['Public_Transport_Accessibility'].classes_)
security = st.sidebar.selectbox("Security Features", encoders['Security'].classes_)
furnishing = st.sidebar.selectbox("Furnished Status", encoders['Furnished_Status'].classes_)

# 4. Main Dashboard UI
st.title("🏡 Real Estate Investment Analysis Dashboard")
st.markdown("Enter property details in the sidebar to evaluate investment potential and 5-year returns.")

if st.sidebar.button("📊 Analyze Investment"):
    # Create Input DataFrame
    input_dict = {f: 0 for f in model_features} # Initialize with zeros
    
    # Map inputs to features
    input_dict['State'] = encoders['State'].transform([state])[0]
    input_dict['City'] = encoders['City'].transform([city])[0]
    input_dict['Property_Type'] = encoders['Property_Type'].transform([prop_type])[0]
    input_dict['Public_Transport_Accessibility'] = encoders['Public_Transport_Accessibility'].transform([transport])[0]
    input_dict['Security'] = encoders['Security'].transform([security])[0]
    input_dict['Furnished_Status'] = encoders['Furnished_Status'].transform([furnishing])[0]
    
    # Numerical features
    input_dict['Size_in_SqFt'] = sqft
    input_dict['BHK'] = bhk
    input_dict['Price_in_Lakhs'] = price
    input_dict['Price_per_SqFt'] = price / sqft
    input_dict['Year_Built'] = year_built
    input_dict['Age_of_Property'] = 2024 - year_built
    
    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([input_dict])[model_features]
    
    # 5. Predictions
    future_price = reg_model.predict(input_df)[0]
    is_good_inv = clf_model.predict(input_df)[0]
    roi = ((future_price - price) / price) * 100

    # 6. Display Results
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Estimated Price (2029)", f"₹{future_price:.2f} L")
    with col2:
        st.metric("Total ROI %", f"{roi:.1f}%", delta=f"{roi/5:.1f}% Annually")
    with col3:
        if is_good_inv == 1:
            st.success("Verdict: GOOD INVESTMENT ✅")
        else:
            st.warning("Verdict: CAUTION ⚠️")

    # 7. Visualization
    st.subheader("📈 5-Year Appreciation Forecast")
    years = np.array([2024, 2025, 2026, 2027, 2028, 2029])
    # Linear projection for chart
    prices = np.linspace(price, future_price, 6)
    fig = px.line(x=years, y=prices, labels={'x': 'Year', 'y': 'Price (Lakhs)'}, markers=True)
    fig.update_traces(line_color='#2ecc71')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Please adjust property details and click 'Analyze Investment' to see the forecast.")

# --- ADD THIS TO YOUR DATA LOADING SECTION ---
@st.cache_data
def load_market_data():
    # Load only necessary columns to save memory
    return pd.read_csv('https://docs.google.com/spreadsheets/d/1UW4YOCGKDjbLddzl4vAT6Zn9PcnqHbpV/edit?usp=drive_link&ouid=102993098082308732946&rtpof=true&sd=true', usecols=['City', 'Price_in_Lakhs', 'Property_Type'])
             
market_df = load_market_data()

# --- COMPLETED EDA SECTION ---
if st.checkbox("Show Market Context"):
    st.subheader(f"📊 Market Analysis for {city}")
    
    # Filter data for the selected city
    city_data = market_df[market_df['City'] == city]
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.write(f"**Price Distribution in {city}**")
        # Create a histogram of prices in the selected city
        fig_hist = px.histogram(
            city_data, 
            x="Price_in_Lakhs", 
            nbins=30,
            color_discrete_sequence=['#3498db'],
            labels={'Price_in_Lakhs': 'Price (Lakhs)'}
        )
        fig_hist.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_chart2:
        st.write(f"**Average Price by Property Type**")
        # Average price comparison
        avg_price = city_data.groupby('Property_Type')['Price_in_Lakhs'].mean().reset_index()
        fig_bar = px.bar(
            avg_price, 
            x='Property_Type', 
            y='Price_in_Lakhs',
            color='Property_Type',
            text_auto='.2f'
        )
        fig_bar.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Show summary statistics
    st.write(f"**Quick Stats for {city}:**")
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    stat_col1.metric("Avg Price", f"₹{city_data['Price_in_Lakhs'].mean():.2f} L")
    stat_col2.metric("Median Price", f"₹{city_data['Price_in_Lakhs'].median():.2f} L")

    stat_col3.metric("Listings Found", f"{len(city_data)}")
