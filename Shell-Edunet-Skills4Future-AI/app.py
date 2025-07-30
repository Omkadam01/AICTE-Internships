import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Set Streamlit page config first thing
st.set_page_config(page_title="EV Forecast", layout="wide", initial_sidebar_state="collapsed")

# === Load Model and Data ===
@st.cache_resource
def load_model():
    """Load the pre-trained forecasting model."""
    try:
        return joblib.load('forecasting_ev_model.pkl')
    except FileNotFoundError:
        st.error("Forecasting model 'forecasting_ev_model.pkl' not found. Please ensure the model file is in the correct directory.")
        st.stop()

model = load_model()

@st.cache_data
def load_data():
    """Load and preprocess historical EV data."""
    try:
        csv_path = "preElectric_Vehicle_Population_Size_History_By_County.csv"
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Electric Vehicle (EV) Total'] = pd.to_numeric(df['Electric Vehicle (EV) Total'], errors='coerce').fillna(0)
        
        # Feature Engineering
        df['county_encoded'] = df['County'].astype('category').cat.codes
        min_date = df['Date'].min()
        df['months_since_start'] = (df['Date'].dt.year - min_date.year) * 12 + (df['Date'].dt.month - min_date.month)
        
        return df.sort_values("Date")
    except FileNotFoundError:
        st.error(f"CSV file '{csv_path}' not found. Please check the file location and name.")
        st.stop()

df = load_data()

# === Styling ===
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #28313B, #485461);
        }
        .st-emotion-cache-16txtl3 {
            padding-top: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #485461;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(38, 39, 48, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #FFFFFF; font-weight: bold;'>EV Adoption Forecaster</h1>
        <p style='color: #E0E0E0; font-size: 18px;'>
            Forecasting Electric Vehicle Adoption for Counties in Washington State
        </p>
    </div>
""", unsafe_allow_html=True)
st.image("ev-car-factory.jpg", use_container_width=True)

# === Core Forecasting Function ===
@st.cache_data
def generate_forecast(_model, county_df, forecast_horizon=36):
    """Generate a 36-month forecast for a given county."""
    
    historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()
    county_code = county_df['county_encoded'].iloc[0]

    future_predictions = []
    for i in range(1, forecast_horizon + 1):
        # Feature calculation for the next step
        lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        recent_cumulative = cumulative_ev[-6:]
        ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) > 1 else 0

        features = {
            'months_since_start': months_since_start + i,
            'county_encoded': county_code,
            'ev_total_lag1': lag1, 'ev_total_lag2': lag2, 'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }
        
        pred = _model.predict(pd.DataFrame([features]))[0]
        pred = max(0, round(pred)) # Ensure prediction is non-negative
        
        future_predictions.append(pred)
        
        # Update history for the next iteration
        historical_ev.append(pred)
        historical_ev.pop(0)
        cumulative_ev.append(cumulative_ev[-1] + pred)
        cumulative_ev.pop(0)
        
    forecast_dates = pd.date_range(start=latest_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')
    return pd.DataFrame({'Date': forecast_dates, 'Predicted EV Total': future_predictions})

# === UI Tabs ===
tab1, tab2 = st.tabs(["Single County Forecast", "Compare Counties"])

with tab1:
    st.header("Single County Deep Dive")
    county_list = sorted(df['County'].dropna().unique().tolist())
    selected_county = st.selectbox("Select a County", county_list, key="single_county_select")

    if selected_county:
        county_df = df[df['County'] == selected_county]
        
        # Generate Forecast
        forecast_df = generate_forecast(model, county_df)
        
        # Prepare data for plotting
        historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        historical_cum.rename(columns={'Electric Vehicle (EV) Total': 'EV Count'}, inplace=True)
        historical_cum['Type'] = 'Historical'
        
        forecast_cum = forecast_df.copy()
        forecast_cum.rename(columns={'Predicted EV Total': 'EV Count'}, inplace=True)
        forecast_cum['Type'] = 'Forecast'

        combined_df = pd.concat([historical_cum, forecast_cum], ignore_index=True)
        combined_df['Cumulative EV'] = combined_df['EV Count'].cumsum()

        # Metrics
        latest_historical_total = int(county_df['Electric Vehicle (EV) Total'].sum())
        total_forecasted = int(forecast_df['Predicted EV Total'].sum())
        total_cumulative_forecast = latest_historical_total + total_forecasted
        
        growth_pct = (total_forecasted / latest_historical_total) * 100 if latest_historical_total > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Historical EVs", value=f"{latest_historical_total:,}")
        with col2:
            st.metric(label="Forecasted New EVs (Next 3 Years)", value=f"{total_forecasted:,}")
        with col3:
            st.metric(label="Projected Growth", value=f"{growth_pct:.1f}%")
        
        # Plot
        fig = px.line(
            combined_df, 
            x='Date', 
            y='Cumulative EV', 
            color='Type',
            title=f"Cumulative EV Adoption Forecast for {selected_county} County",
            labels={'Cumulative EV': 'Total Electric Vehicles', 'Date': 'Year'},
            template='plotly_dark'
        )
        fig.update_layout(
            legend_title_text='',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Compare County Trends")
    multi_counties = st.multiselect(
        "Select up to 3 counties to compare", 
        county_list, 
        max_selections=3,
        key="multi_county_select"
    )

    if multi_counties:
        comparison_data = []
        for cty in multi_counties:
            cty_df = df[df['County'] == cty]
            
            # Historical part
            hist_cty = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
            hist_cty['County'] = cty
            
            # Forecast part
            forecast_cty = generate_forecast(model, cty_df)
            forecast_cty['County'] = cty
            forecast_cty.rename(columns={'Predicted EV Total': 'Electric Vehicle (EV) Total'}, inplace=True)
            
            # Combine
            combined_cty = pd.concat([hist_cty, forecast_cty], ignore_index=True)
            combined_cty['Cumulative EV'] = combined_cty['Electric Vehicle (EV) Total'].cumsum()
            comparison_data.append(combined_cty)

        if comparison_data:
            comp_df = pd.concat(comparison_data, ignore_index=True)
            
            fig = px.line(
                comp_df,
                x='Date',
                y='Cumulative EV',
                color='County',
                title='Comparison of Cumulative EV Adoption Trends',
                labels={'Cumulative EV': 'Total Electric Vehicles', 'Date': 'Year', 'County': 'County'},
                template='plotly_dark'
            )
            fig.update_layout(legend_title_text='County')
            st.plotly_chart(fig, use_container_width=True)

# === Footer ===
st.markdown("---")
st.markdown("<p style='text-align: center; color: #E0E0E0;'>Prepared By Om Kadam</p>", unsafe_allow_html=True)