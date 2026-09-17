
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="StockVision",
    page_icon="📈",
    layout="wide"
)


# ---------------- HEADER ----------------

st.title("📈 StockVision")
st.subheader("AI-Powered Stock Market Analysis & Prediction")

st.write(
    "Analyze historical stock data, identify market trends "
    "and generate machine-learning based price predictions."
)


# ---------------- STOCK LIST ----------------

stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS"
}


# ---------------- SIDEBAR ----------------

st.sidebar.header("⚙️ Stock Settings")

stock_name = st.sidebar.selectbox(
    "Select Stock",
    list(stocks.keys())
)

period = st.sidebar.selectbox(
    "Historical Data",
    ["1y", "2y", "5y"]
)

symbol = stocks[stock_name]


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data(symbol, period):

    data = yf.Ticker(symbol).history(
        period=period,
        interval="1d",
        auto_adjust=False
    )

    return data


# ---------------- ANALYZE BUTTON ----------------

if st.button("🚀 Analyze Stock"):

    with st.spinner("Downloading market data and training model..."):

        data = load_data(symbol, period)

    if data.empty:

        st.error(
            "Stock data could not be downloaded. Please try again."
        )

    else:

        data = data.dropna()

        # ---------------- TECHNICAL INDICATORS ----------------

        data["MA_7"] = data["Close"].rolling(7).mean()
        data["MA_30"] = data["Close"].rolling(30).mean()

        data["Daily_Return"] = data["Close"].pct_change()

        data["Volatility"] = (
            data["Daily_Return"]
            .rolling(7)
            .std()
        )

        data["Previous_Close"] = data["Close"].shift(1)

        # Next day's closing price = prediction target

        data["Next_Close"] = data["Close"].shift(-1)

        data = data.dropna()


        # ---------------- FEATURES ----------------

        features = [
            "Open",
            "High",
            "Low",
            "Volume",
            "MA_7",
            "MA_30",
            "Daily_Return",
            "Volatility",
            "Previous_Close"
        ]

        X = data[features]
        y = data["Next_Close"]


        # ---------------- TRAIN TEST SPLIT ----------------

        split = int(len(data) * 0.8)

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]


        # ---------------- MODEL ----------------

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)


        # ---------------- TEST PREDICTIONS ----------------

        predictions = model.predict(X_test)


        # ---------------- MODEL PERFORMANCE ----------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )


        # ---------------- NEXT PRICE PREDICTION ----------------

        latest_features = data[features].iloc[-1:].values

        predicted_price = float(
            model.predict(latest_features)[0]
        )

        current_price = float(
            data["Close"].iloc[-1]
        )

        expected_change = (
            (predicted_price - current_price)
            / current_price
        ) * 100


        # ---------------- PREDICTION SIGNAL ----------------

        if expected_change > 1:

            signal = "📈 Potential Upward Movement"

        elif expected_change < -1:

            signal = "📉 Potential Downward Movement"

        else:

            signal = "➡️ Relatively Stable"


        # ---------------- SUCCESS MESSAGE ----------------

        st.success(
            f"Analysis completed for {stock_name}"
        )


        # ---------------- MAIN METRICS ----------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Current Price",
                f"₹{current_price:,.2f}"
            )

        with col2:

            st.metric(
                "Predicted Next Price",
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


        # ---------------- PREDICTION SIGNAL ----------------

        st.subheader("🔮 Prediction Signal")

        st.info(
            f"Model-based indication: **{signal}**"
        )


        # ---------------- HISTORICAL PRICE ----------------

        st.subheader("📈 Historical Stock Price")

        st.line_chart(
            data["Close"]
        )


        # ---------------- MOVING AVERAGES ----------------

        st.subheader("📊 Moving Average Analysis")

        chart_data = data[
            ["Close", "MA_7", "MA_30"]
        ]

        st.line_chart(
            chart_data
        )


        # ---------------- MARKET STATISTICS ----------------

        st.subheader("📌 Market Statistics")

        stat1, stat2, stat3, stat4 = st.columns(4)

        with stat1:

            st.metric(
                "Highest Price",
                f"₹{data['High'].max():,.2f}"
            )

        with stat2:

            st.metric(
                "Lowest Price",
                f"₹{data['Low'].min():,.2f}"
            )

        with stat3:

            st.metric(
                "Average Price",
                f"₹{data['Close'].mean():,.2f}"
            )

        with stat4:

            st.metric(
                "Average Volume",
                f"{data['Volume'].mean():,.0f}"
            )


        # ---------------- MODEL PERFORMANCE ----------------

        st.subheader("🤖 Model Performance")

        performance1, performance2 = st.columns(2)

        with performance1:

            st.write(
                f"**MAE:** {mae:.2f}"
            )

            st.write(
                f"**RMSE:** {rmse:.2f}"
            )

            st.write(
                f"**R² Score:** {r2:.4f}"
            )

        with performance2:

            comparison = pd.DataFrame({
                "Actual": y_test.values,
                "Predicted": predictions
            })

            st.line_chart(
                comparison
            )


        # ---------------- HISTORICAL DATA TABLE ----------------

        with st.expander("📋 View Historical Data"):

            st.dataframe(
                data.tail(100),
                use_container_width=True
            )


# ---------------- DISCLAIMER ----------------

st.markdown("---")

st.info(
    "⚠️ StockVision provides machine-learning based estimates "
    "for educational and analytical purposes only. "
    "Predictions should not be treated as financial advice."
)
