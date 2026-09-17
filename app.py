import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="StockVision",
    page_icon="📈",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1f2937);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 10px;
}

.signal-box {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #374151;
    background-color: #111827;
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    font-size: 13px;
    padding-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="hero">

<h1>📈 StockVision</h1>

<p>
AI-Powered Stock Market Analysis & Prediction Dashboard
</p>

</div>
""", unsafe_allow_html=True)


# =====================================================
# STOCK LIST
# =====================================================

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


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Stock Settings")

stock_name = st.sidebar.selectbox(
    "Select Stock",
    list(stocks.keys())
)

period = st.sidebar.selectbox(
    "Historical Period",
    ["1y", "2y", "5y"]
)

symbol = stocks[stock_name]

st.sidebar.markdown("---")

st.sidebar.info(
    "StockVision uses historical market data "
    "and a Random Forest machine-learning model "
    "for educational analysis."
)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data(symbol, period):

    data = yf.Ticker(symbol).history(
        period=period,
        interval="1d",
        auto_adjust=False
    )

    return data


# =====================================================
# ANALYZE
# =====================================================

if st.button("🚀 Analyze Stock", use_container_width=True):

    with st.spinner(
        "Fetching market data and training AI model..."
    ):

        data = load_data(symbol, period)


    if data.empty:

        st.error(
            "Unable to retrieve stock data. Please try again later."
        )

        st.stop()


    # =================================================
    # DATA PREPARATION
    # =================================================

    data = data.dropna()

    data["MA_7"] = data["Close"].rolling(7).mean()

    data["MA_30"] = data["Close"].rolling(30).mean()

    data["Daily_Return"] = data["Close"].pct_change()

    data["Volatility"] = (
        data["Daily_Return"].rolling(7).std()
    )

    data["Previous_Close"] = data["Close"].shift(1)

    data["Next_Close"] = data["Close"].shift(-1)

    data = data.dropna()


    # =================================================
    # FEATURES
    # =================================================

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


    # =================================================
    # TRAIN TEST SPLIT
    # =================================================

    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]


    # =================================================
    # RANDOM FOREST MODEL
    # =================================================

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)


    # =================================================
    # PREDICTION
    # =================================================

    predictions = model.predict(X_test)


    # =================================================
    # MODEL PERFORMANCE
    # =================================================

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


    # =================================================
    # NEXT PRICE
    # =================================================

    latest_features = (
        data[features].iloc[-1:].values
    )

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


    # =================================================
    # SIGNAL
    # =================================================

    if expected_change > 1:

        signal = "📈 Potential Upward Movement"

    elif expected_change < -1:

        signal = "📉 Potential Downward Movement"

    else:

        signal = "➡️ Relatively Stable"


    # =================================================
    # SUCCESS
    # =================================================

    st.success(
        f"Analysis completed for {stock_name}"
    )


    # =================================================
    # KEY METRICS
    # =================================================

    st.markdown(
        '<div class="section-title">📊 Key Market Insights</div>',
        unsafe_allow_html=True
    )

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
            "Model R²",
            f"{r2:.4f}"
        )


    # =================================================
    # PREDICTION SIGNAL
    # =================================================

    st.markdown(
        '<div class="section-title">🔮 Prediction Signal</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="signal-box">

        <h3>{signal}</h3>

        <p>
        This indication is based on the machine-learning
        model's predicted next closing price.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # =================================================
    # TABS
    # =================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price Analysis",
        "📊 Market Statistics",
        "🤖 Model Performance",
        "📋 Historical Data"
    ])


    # =================================================
    # PRICE ANALYSIS
    # =================================================

    with tab1:

        st.subheader("📈 Historical Stock Price")

        st.line_chart(
            data["Close"],
            use_container_width=True
        )

        st.subheader("📊 Moving Average Analysis")

        moving_average_data = data[
            ["Close", "MA_7", "MA_30"]
        ]

        st.line_chart(
            moving_average_data,
            use_container_width=True
        )


    # =================================================
    # MARKET STATISTICS
    # =================================================

    with tab2:

        st.subheader("📌 Market Statistics")

        stat1, stat2 = st.columns(2)

        with stat1:

            st.metric(
                "Highest Price",
                f"₹{data['High'].max():,.2f}"
            )

            st.metric(
                "Lowest Price",
                f"₹{data['Low'].min():,.2f}"
            )

        with stat2:

            st.metric(
                "Average Price",
                f"₹{data['Close'].mean():,.2f}"
            )

            st.metric(
                "Average Volume",
                f"{data['Volume'].mean():,.0f}"
            )

        st.subheader("📉 Daily Return")

        st.line_chart(
            data["Daily_Return"],
            use_container_width=True
        )


    # =================================================
    # MODEL PERFORMANCE
    # =================================================

    with tab3:

        st.subheader(
            "🤖 Random Forest Model Performance"
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.metric(
                "MAE",
                f"{mae:.2f}"
            )

        with metric2:

            st.metric(
                "RMSE",
                f"{rmse:.2f}"
            )

        with metric3:

            st.metric(
                "R² Score",
                f"{r2:.4f}"
            )

        st.subheader(
            "Actual vs Predicted Price"
        )

        comparison = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": predictions
        })

        st.line_chart(
            comparison,
            use_container_width=True
        )


    # =================================================
    # HISTORICAL DATA
    # =================================================

    with tab4:

        st.subheader(
            "📋 Historical Stock Data"
        )

        st.dataframe(
            data.tail(100),
            use_container_width=True
        )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    StockVision • AI & Machine Learning Based Stock Analysis

    <br><br>

    ⚠️ Predictions are generated for educational
    and analytical purposes only and should not
    be considered financial advice.

    </div>
    """,
    unsafe_allow_html=True
)
