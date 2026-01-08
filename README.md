🏡 Real Estate Investment Advisor: Predicting Property Profitability & Future Value
This project is a machine learning-powered tool designed to help investors and homebuyers evaluate real estate opportunities across various Indian cities. It provides two primary insights: a regression analysis to forecast property value over a 5-year horizon and a classification model to determine if a property qualifies as a "Good Investment" based on market metrics and infrastructure.

🚀 Features
Future Value Prediction: Forecasts the estimated property price in 5 years using a Random Forest Regressor.

Investment Rating: Classifies properties as "Good" or "Bad" investments based on a multi-factor score including transport accessibility, security, and price relative to the city median.

Interactive Dashboard: A user-friendly Streamlit interface allowing users to input specific property details (location, size, amenities) and receive instant analysis.

Market Context: Provides visual data on price distributions and average prices by property type for the selected city.


🛠️ Tech Stack

Language: Python

Machine Learning: Scikit-learn (Random Forest Regressor & Classifier)

Web Framework: Streamlit

Data Manipulation: Pandas, NumPy

Visualization: Plotly Express, Matplotlib, Seaborn

📂 Project Structure

app.py: The main Streamlit application script containing the UI logic and visualization code.


Untitled.ipynb: Jupyter Notebook used for data cleaning, feature engineering, and model training.


reg_model.pkl: Trained Random Forest model for price prediction.


clf_model.pkl: Trained Random Forest model for investment classification.


encoders.pkl: Serialized LabelEncoders for categorical feature processing.


requirements.txt: List of necessary Python dependencies.

⚙️ Installation & Usage
1. Clone the Repository

Bash

git clone https://github.com/Harshal1720/Real-Estate-Investment-Advisor-Predicting-Property-Profitability-Future-Value.git

cd Real-Estate-Investment-Advisor

2. Install Dependencies

Ensure you have Python installed, then run:

Bash

pip install -r requirements.txt

3. Run the Application
Bash

streamlit run app.py


📊 How it Works

Data Processing: The model uses a dataset of approximately 250,000 property records.

Target Logic:

Future Price: Calculated using an 8% Compound Annual Growth Rate (CAGR) over 5 years.

Investment Score: properties gain points for being priced below the city median, having high public transport accessibility, and featuring security.

Inference: User inputs from the sidebar are encoded and fed into the .pkl models to generate real-time predictions.
