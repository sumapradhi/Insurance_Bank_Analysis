import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load your data
df = pd.read_csv(r'C:\GUVI\NewVM\Comp_Banking_Analytics\credit_finaldata.csv')

# Load pickled models and scalers
model_credit_score = joblib.load(r'C:\GUVI\NewVM\Comp_Banking_Analytics\model_joblib')
imputer_kmeans = joblib.load(r'C:\GUVI\NewVM\Comp_Banking_Analytics\imputer.pkl')
scaler_kmeans = joblib.load(r'C:\GUVI\NewVM\Comp_Banking_Analytics\scaler.pkl')
kmeans_model = joblib.load(r'C:\GUVI\NewVM\Comp_Banking_Analytics\kmeans_model.pkl')
model_credit_risk = joblib.load(r'C:\GUVI\NewVM\Comp_Banking_Analytics\rf_model.pkl')
scaler_credit_risk = joblib.load(r'C:\GUVI\NewVM\Comp_Banking_Analytics\scaler1.pkl')
model_investment = joblib.load(r'C:\GUVI\NewVM\Comp_Banking_Analytics\lr_model.pkl')
scaler_investment = joblib.load(r'C:\GUVI\NewVM\Comp_Banking_Analytics\scaler2.pkl')

columns_for_clustering = ['Age', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
                           'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month',
                           'Monthly_Balance', 'Credit_Score', 'Log_Annual_Income']

# Function for Credit Score Prediction
def predict_credit_score(input_values):
    features = np.array([input_values])
    return model_credit_score.predict(features)[0]

# Function for Clustering Prediction
def predict_clustering(input_values):
    # Reshape input_values to (1, n_features)
    features = np.array(input_values).reshape(1, -1)

    # Create a DataFrame from the input values
    sample_input = pd.DataFrame(features, columns=columns_for_clustering)

    # Impute missing values for the sample input
    sample_input[columns_for_clustering] = imputer_kmeans.transform(sample_input[columns_for_clustering])

    # Standardize the sample input
    sample_input_scaled = scaler_kmeans.transform(sample_input[columns_for_clustering])

    # Predict the cluster for the user input
    predicted_cluster = kmeans_model.predict(sample_input_scaled)[0]
    return predicted_cluster

# Function for Credit Risk Prediction
def predict_credit_risk(input_values):
    features = np.array([input_values])
    sample_input_scaled = scaler_credit_risk.transform(features)
    predicted_credit_risk = model_credit_risk.predict(sample_input_scaled)[0]
    return predicted_credit_risk

# Function for Investment Prediction
def predict_investment(input_values):
    features = np.array(input_values).reshape(1, -1)  # Reshape to (1, n_features)
    sample_input_scaled = scaler_investment.transform(features)
    predicted_amount = model_investment.predict(sample_input_scaled)[0]
    return predicted_amount

# Sidebar option menu
selected_option = st.sidebar.selectbox(
    "Main Menu",
    ["About Project", "Predictions"],
    format_func=lambda x: "🏠 About Project" if x == "About Project" else "🔮 Predictions"
)

# Display content based on the selected option
if selected_option == "About Project":
    st.title("About Project")
    st.markdown("""
        ## Overview:
        This prediction app utilizes machine learning models for various financial predictions. The app is designed to provide insights into credit scores, clustering, credit risk, and investment amounts.

        ## Technologies Used:
        - Python
        - Streamlit
        - Scikit-learn
        - Pandas

        ## Features:
        - **Credit Score Prediction:**
          Users can input financial data, and the app predicts their credit score using a RandomForestClassifier model.

        - **Clustering Prediction:**
          Utilizes KMeans clustering to categorize users into clusters based on their financial attributes.

        - **Credit Risk Prediction:**
          Predicts whether a user is at credit risk or not using a RandomForestClassifier model.

        - **Investment Prediction:**
          Predicts the monthly amount invested based on financial features using a Linear Regression model.

        ## How to Use:
        - Select "Predictions" in the sidebar to access the prediction functionalities.
        - Input relevant financial information.
        - Click the "Predict" button for the desired prediction.

       
    """)
    st.balloons()


elif selected_option == "Predictions":
    st.title("Predictions")
    st.balloons()

    # Submenu for prediction tabs
    prediction_tab = st.sidebar.selectbox(
        "Select Prediction",
        ["Credit Score", "Clustering", "Credit Risk", "Investment"]
    )

    # Function to get user input based on the selected tab
    def get_user_input(features):
        input_values = {}
        for feature in features:
            input_values[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)
        return input_values

    # Display prediction based on the selected tab
    if prediction_tab == 'Credit Score':
        st.header('Credit Score Prediction')
        input_values_credit_score = get_user_input(['Log_Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                                                    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                                                    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_Mix',
                                                    'Outstanding_Debt', 'Credit_History_Age', 'Monthly_Balance'])
        if st.button('Predict Credit Score'):
            predicted_credit_score = predict_credit_score(list(input_values_credit_score.values()))
            st.success(f'Predicted Credit Score: {predicted_credit_score}')

    elif prediction_tab == 'Clustering':
        st.header('Clustering Prediction')
        input_values_clustering = get_user_input(columns_for_clustering)
        if st.button('Predict Clustering'):
            predicted_cluster = predict_clustering(list(input_values_clustering.values()))
            st.success(f'Predicted Cluster: {predicted_cluster}')

    elif prediction_tab == 'Credit Risk':
        st.header('Credit Risk Prediction')
        input_values_credit_risk = get_user_input(['Age', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                                                  'Num_Credit_Card', 'Credit_Utilization_Ratio', 'Credit_History_Age',
                                                  'Total_EMI_per_month', 'Monthly_Balance', 'Credit_Score',
                                                  'Log_Annual_Income'])
        if st.button('Predict Credit Risk'):
            predicted_credit_risk = predict_credit_risk(list(input_values_credit_risk.values()))
            st.success(f'Predicted Credit Risk: {predicted_credit_risk}')

    elif prediction_tab == 'Investment':
        st.header('Investment Prediction')
        input_values_investment = get_user_input(['Age', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                                                 'Num_Credit_Card', 'Credit_History_Age', 'Payment_of_Min_Amount',
                                                 'Total_EMI_per_month', 'Monthly_Balance', 'Credit_Score',
                                                 'Log_Annual_Income'])
        if st.button('Predict Investment'):
            predicted_investment = predict_investment(list(input_values_investment.values()))
            st.success(f'Predicted Investment: {predicted_investment}')