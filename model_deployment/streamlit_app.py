# adding necessary libraries
import streamlit as st
import pandas as pd
import lifetimes
import math
import numpy as np
import datetime
np.random.seed(42)
import altair as alt
import time
import warnings
warnings.filterwarnings("ignore")
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from lifetimes.plotting import plot_frequency_recency_matrix
from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes.plotting import plot_period_transactions
from lifetimes.utils import calibration_and_holdout_data
from lifetimes import ParetoNBDFitter
from lifetimes.plotting import plot_history_alive
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================================
# Page Configuration
# =====================================================================================
st.set_page_config(
    page_title="CLV Prediction App",
    page_icon="🚀",
    layout="wide"
)


# =====================================================================================
# Sidebar Configuration
# =====================================================================================
st.sidebar.image("https://www.celebaltech.com/assets/img/logo.png", width=200)
st.sidebar.title("Configuration Panel")
st.sidebar.markdown("---")

st.sidebar.header("⚙️ Input Parameters")
st.sidebar.markdown("Adjust the model parameters for the CLV calculation.")

days = st.sidebar.slider("Select Prediction Days (T)", min_value=1, max_value=365, step=1, value=30,
                         help="The number of days in the future to predict customer purchases.")
profit = st.sidebar.slider("Select Profit Margin", min_value=0.01, max_value=0.25, step=0.01, value=0.05, format="%f",
                           help="The estimated profit margin on each sale (e.g., 0.05 for 5%).")

# Display selected features in an expander
with st.sidebar.expander("📊 Selected Input Features", expanded=False):
    slider_data = {
        "Prediction Days (T)": days,
        "Profit Margin": profit
    }
    features_df = pd.DataFrame(slider_data, index=[0])
    st.dataframe(features_df)

st.sidebar.markdown("---")
st.sidebar.header("📝 File Upload Instructions")
st.sidebar.markdown("""
- Please upload your RFM data as a **CSV file**.
- Ensure the columns are correctly named: `frequency`, `recency`, `T`, `monetary_value`.
- You can download a sample file for reference.
""")
st.sidebar.markdown("[Example CSV Input File](https://raw.githubusercontent.com/mukulsinghal001/customer-lifetime-prediction-using-python/main/model_deployment/sample_file.csv)")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**App by:** Md Aftab  
**Company:** Celebal Technologies
""")


# =====================================================================================
# Main Application Body
# =====================================================================================

# Main title and introduction
st.title("🚀 AI-Powered Customer Lifetime Value (CLV) Prediction")
st.markdown("""
Welcome to the CLV Prediction App! This tool leverages the **Pareto/NBD** and **Gamma-Gamma** models to forecast future customer value.  
Simply upload your data to segment your customers and identify your most valuable ones.
""")

st.image("https://sarasanalytics.com/wp-content/uploads/2019/11/Customer-Lifetime-value-new-1.jpg", use_column_width=True)

# File Uploader
data = st.file_uploader("📂 Upload Your RFM Data Here", type="csv")


# Conditional display: Show instructions if no file is uploaded, or results if a file is uploaded.
if data is None:
    st.info("💡 Please upload a CSV file using the uploader above to begin the analysis.")
    
    with st.expander("📘 How It Works & Column Definitions"):
        st.markdown("""
        This application analyzes customer behavior based on RFM metrics to predict their lifetime value.
        
        **1. Data Upload:** You provide a CSV file with the following columns:
        - **`frequency`**: The number of repeat purchases the customer has made.
        - **`recency`**: The time between the customer's first and last purchase (in days).
        - **`T`**: The time since the customer's first purchase (in days), representing their age.
        - **`monetary_value`**: The average value of a customer's purchases.
        
        **2. Prediction Models:**
        - The **Pareto/NBD model** predicts the number of future purchases a customer is likely to make.
        - The **Gamma-Gamma model** estimates the average monetary value of those future purchases.
        
        **3. CLV Calculation:** Combining these predictions gives us the Customer Lifetime Value (CLV), discounted for time.
        
        **4. Customer Segmentation:**
        - **K-Means Clustering** is used on the prediction outputs to group customers into four value segments: **Low, Medium, High, and Very High**.
        """)

else:
    # This is the original function, with only cosmetic changes and an improved download button.
    # THE CORE LOGIC IS UNALTERED as per the request.
    def process_and_display_data(data_file, t_days, profit_m):
        """
        Loads data, runs CLV models, performs clustering, and displays results in Streamlit.
        """
        input_data = pd.read_csv(data_file)
        if "Unnamed: 0" in input_data.columns:
            input_data = pd.DataFrame(input_data.iloc[:, 1:])

        # --- Pareto/NBD Model ---
        pareto_model = lifetimes.ParetoNBDFitter(penalizer_coef=0.1)
        pareto_model.fit(input_data["frequency"], input_data["recency"], input_data["T"])
        
        t = t_days
        input_data["predicted_purchases"] = pareto_model.conditional_expected_number_of_purchases_up_to_time(
            t, input_data["frequency"], input_data["recency"], input_data["T"])
        input_data["p_alive"] = pareto_model.conditional_probability_alive(
            input_data["frequency"], input_data["recency"], input_data["T"])

        # --- Gamma-Gamma Model ---
        # Filter out non-positive frequency and monetary_value customers
        idx = input_data[(input_data["frequency"] <= 0.0) | (input_data["monetary_value"] <= 0.0)].index
        input_data = input_data.drop(idx, axis=0)
        input_data.reset_index(drop=True, inplace=True)

        ggf_model = lifetimes.GammaGammaFitter(penalizer_coef=0.1)
        ggf_model.fit(input_data["frequency"], input_data["monetary_value"])

        input_data["expected_avg_sales_"] = ggf_model.conditional_expected_average_profit(
            input_data["frequency"], input_data["monetary_value"])
        
        input_data["predicted_clv"] = ggf_model.customer_lifetime_value(
            pareto_model,
            input_data["frequency"],
            input_data["recency"],
            input_data["T"],
            input_data["monetary_value"],
            time=1,  # Predict CLV for 1 month (30 days)
            freq='D',
            discount_rate=0.01
        )
        input_data["profit_margin"] = input_data["predicted_clv"] * profit_m
        input_data = input_data.reset_index(drop=True)

        # --- K-Means Clustering for Segmentation ---
        col = ["predicted_purchases", "expected_avg_sales_", "predicted_clv", "profit_margin"]
        new_df = input_data[col]
        
        # Ensure there are enough samples for 4 clusters
        n_clusters = min(4, len(new_df))
        if n_clusters < 2:
            st.warning("Not enough unique data points to perform customer segmentation.")
            return input_data

        k_model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=1000, random_state=42).fit(new_df)
        labels = k_model.labels_
        input_data["Labels"] = labels

        # Dynamically map cluster labels to value segments
        cluster_centers = pd.DataFrame(k_model.cluster_centers_, columns=col)
        cluster_centers['clv_rank'] = cluster_centers['predicted_clv'].rank(method='first').astype(int)
        label_mapper = {
            cluster[0]: segment for cluster, segment in 
            zip(cluster_centers['clv_rank'].sort_values().iteritems(), ["Low", "Medium", "High", "V_High"])}
        
        input_data["Labels"] = input_data["Labels"].map(label_mapper)
        
        return input_data

    # --- Display Results ---
    st.header("📊 Analysis Results", divider='rainbow')

    with st.spinner('Analyzing your data... This might take a moment ⏳'):
        result_df = process_and_display_data(data, days, profit)

        st.subheader("Customer Data with CLV Predictions & Segments")
        st.dataframe(result_df)

        # --- Altair Chart for Segmentation ---
        st.subheader("Customer Segmentation Distribution")
        
        # Adding a count bar chart
        fig = alt.Chart(result_df).mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5
        ).encode(
            x=alt.X('count(Labels):Q', title='Number of Customers'),
            y=alt.Y('Labels:N', title='Customer Segment', sort='-x'),
            color=alt.Color('Labels:N', legend=None)
        ).properties(
            title='Distribution of Customers Across Segments'
        )

        # Adding annotation text to the chart
        text = fig.mark_text(
            align="left",
            baseline="middle",
            dx=3  # Nudges text to right so it doesn't overlap with the bar
        ).encode(
            text="count(Labels):Q"
        )
        
        chart = (fig + text).configure_axis(
            grid=False
        ).configure_view(
            strokeWidth=0
        )
        
        st.altair_chart(chart, use_container_width=True)

        # --- Download Button ---
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(result_df)
        
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="customer_lifetime_prediction_result.csv",
            mime="text/csv",
            use_container_width=True
        )