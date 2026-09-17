
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="StockVision",
    page_icon="📈",
    layout="wide"
)

st.title("📈 StockVision")
st.subheader("AI-Powered Stock Price Prediction")

st.write(
    "Analyze historical stock data and estimate the next trading day's closing price."
)


# =========================
# STOCK LIST
# =========================

stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "ITC": "ITC.NS"
}


stock_name = st.selectbox(
    "🏦 Select a Stock",
    list(stocks.keys())
)

symbol = stocks[stock_name]


# =========================
# DOWNLOAD DATA
# =========================

@st.cache_data
def load_data(symbol):

    data = yf.Ticker(symbol).history(
        period="5y",
        interval="1d",
        auto_adjust=False
    )

    return data


# =========================
# ANALYZE BUTTON
# =========================

if st.button("🚀 Analyze Stock"):

    with st.spinner("Fetching stock data and training model..."):

        data = load_data(symbol)

    if data.empty:

        st.error("❌ Stock data could not be downloaded. Please try again.")

    else:

        data = data.dropna()

        # =========================
        # FEATURE ENGINEERING
        # =========================

        data["MA_7"] = data["Close"].rolling(7).mean()
        data["MA_30"] = data["Close"].rolling(30).mean()

        data["Previous_Close"] = data["Close"].shift(1)

        data["Daily_Return"] = data["Close"].pct_change()

        # NEXT DAY CLOSE = TARGET
        data["Next_Close"] = data["Close"].shift(-1)

        data = data.dropna()

        features = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "MA_7",
            "MA_30",
            "Previous_Close",
            "Daily_Return"
        ]

        X = data[features]
        y = data["Next_Close"]


        # =========================
        # TRAIN / TEST SPLIT
        # =========================

        split = int(len(data) * 0.8)

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]


        # =========================
        # RANDOM FOREST MODEL
        # =========================

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)


        # =========================
        # TEST PREDICTIONS
        # =========================

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        r2 = r2_score(y_test, predictions)


        # =========================
        # NEXT DAY PREDICTION
        # =========================

        latest_data = data[features].iloc[-1:]

        predicted_price = float(
            model.predict(latest_data)[0]
        )

        current_price = float(
            data["Close"].iloc[-1]
        )

        expected_change = (
            (predicted_price - current_price)
            / current_price
        ) * 100


        # =========================
        # RESULTS
        # =========================

        st.success(
            f"Analysis completed for {stock_name}"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current Price",
                f"₹{current_price:,.2f}"
            )

        with col2:
            st.metric(
                "Predicted Next Close",
                f"₹{predicted_price:,.2f}"
            )

        with col3:
            st.metric(
                "Expected Change",
                f"{expected_change:.2f}%"
            )

        with col4:
            st.metric(
                "R² Score",
                f"{r2:.4f}"
            )


        # =========================
        # PRICE CHART
        # =========================

        st.subheader("📊 Historical Stock Price")

        chart_data = data[["Close"]].copy()

        st.line_chart(chart_data)


        # =========================
        # MODEL PERFORMANCE
        # =========================

        st.subheader("🤖 Model Performance")

        col1, col2 = st.columns(2)

        with col1:

            st.write(f"**MAE:** {mae:.2f}")

            st.write(f"**RMSE:** {rmse:.2f}")

        with col2:

            st.write(f"**R² Score:** {r2:.4f}")

            st.write(
                f"**Training Data:** {len(X_train):,} records"
            )


        # =========================
        # DISCLAIMER
        # =========================

        st.warning(
            "⚠️ This prediction is generated by a machine learning model "
            "using historical data. It is an estimate and should not be "
            "treated as financial advice."
        )
