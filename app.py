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
# HEADER
# =====================================================

st.title("📈 StockVision")

st.markdown(
    """
    ### Stock Market Analysis & Prediction Dashboard

    Explore historical stock prices, market statistics,
    machine learning predictions and compare different stocks.
    """
)


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


selected_stock = st.sidebar.selectbox(
    "Select Stock",
    list(stocks.keys())
)


period = st.sidebar.selectbox(
    "Historical Period",
    ["1y", "2y", "5y"]
)


symbol = stocks[selected_stock]


# =====================================================
# LOAD DATA FUNCTION
# =====================================================

@st.cache_data
def load_data(symbol, period):

    try:

        data = yf.Ticker(symbol).history(
            period=period,
            interval="1d",
            auto_adjust=False,
            repair=True
        )

        if data.empty:
            return pd.DataFrame()

        # Remove rows where essential price data is missing
        data = data.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close"
            ]
        )

        return data

    except Exception:

        return pd.DataFrame()


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def prepare_features(data):

    data = data.copy()

    data["MA_7"] = (
        data["Close"]
        .rolling(window=7)
        .mean()
    )

    data["MA_30"] = (
        data["Close"]
        .rolling(window=30)
        .mean()
    )

    data["Daily_Return"] = (
        data["Close"]
        .pct_change()
    )

    data["Volatility"] = (
        data["Daily_Return"]
        .rolling(window=7)
        .std()
    )

    data["Previous_Close"] = (
        data["Close"]
        .shift(1)
    )

    data["Next_Close"] = (
        data["Close"]
        .shift(-1)
    )

    data = data.dropna()

    return data


# =====================================================
# LOAD MAIN STOCK DATA
# =====================================================

with st.spinner("Loading stock data..."):

    data = load_data(
        symbol,
        period
    )


if data.empty:

    st.error(
        "Unable to retrieve stock data. Please try again."
    )

    st.stop()


# =====================================================
# PREPARE DATA
# =====================================================

data = prepare_features(data)


# =====================================================
# MACHINE LEARNING FEATURES
# =====================================================

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


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

split_index = int(
    len(data) * 0.80
)


X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]


y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]


# =====================================================
# RANDOM FOREST MODEL
# =====================================================

model = RandomForestRegressor(

    n_estimators=200,

    max_depth=12,

    random_state=42,

    n_jobs=-1

)


model.fit(
    X_train,
    y_train
)


# =====================================================
# PREDICTIONS
# =====================================================

predictions = model.predict(
    X_test
)


# =====================================================
# MODEL METRICS
# =====================================================

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


# =====================================================
# NEXT PRICE PREDICTION
# =====================================================

latest_features = (
    X.iloc[-1:]
)


next_predicted_price = model.predict(
    latest_features
)[0]


latest_actual_price = (
    data["Close"]
    .dropna()
    .iloc[-1]
)


expected_change = (

    (
        next_predicted_price
        -
        latest_actual_price
    )
    /
    latest_actual_price
) * 100


# =====================================================
# PREDICTION SIGNAL
# =====================================================

if expected_change > 1:

    signal = "📈 Potential Upward Movement"

elif expected_change < -1:

    signal = "📉 Potential Downward Movement"

else:

    signal = "➡️ Relatively Stable"


# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(

    [

        "📈 Price Analysis",

        "📊 Market Statistics",

        "🤖 Model Performance",

        "📋 Historical Data",

        "🆚 Stock Comparison"

    ]

)


# =====================================================
# TAB 1 - PRICE ANALYSIS
# =====================================================

with tab1:

    st.subheader(
        f"📈 {selected_stock} Price Analysis"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Current Price",
            f"₹{latest_actual_price:,.2f}"
        )


    with col2:

        st.metric(
            "Predicted Next Price",
            f"₹{next_predicted_price:,.2f}"
        )


    with col3:

        st.metric(
            "Expected Change",
            f"{expected_change:.2f}%"
        )


    st.info(
        f"Prediction Signal: {signal}"
    )


    # -------------------------------------------------
    # CLOSING PRICE
    # -------------------------------------------------

    st.subheader(
        "📊 Historical Closing Price"
    )


    close_chart = data[
        ["Close"]
    ].rename(
        columns={
            "Close": "Closing Price"
        }
    )


    st.line_chart(
        close_chart,
        use_container_width=True
    )


    # -------------------------------------------------
    # MOVING AVERAGES
    # -------------------------------------------------

    st.subheader(
        "📉 Moving Average Analysis"
    )


    moving_average_data = data[

        [
            "Close",
            "MA_7",
            "MA_30"
        ]

    ].rename(

        columns={

            "Close": "Closing Price",

            "MA_7": "7-Day MA",

            "MA_30": "30-Day MA"

        }

    )


    st.line_chart(
        moving_average_data,
        use_container_width=True
    )


# =====================================================
# TAB 2 - MARKET STATISTICS
# =====================================================

with tab2:

    st.subheader(
        "📊 Market Statistics"
    )


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


    st.subheader(
        "📈 Daily Returns"
    )


    return_chart = data[
        ["Daily_Return"]
    ].rename(

        columns={
            "Daily_Return": "Daily Return"
        }

    )


    st.line_chart(
        return_chart,
        use_container_width=True
    )


# =====================================================
# TAB 3 - MODEL PERFORMANCE
# =====================================================

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
            f"{r2:.2f}"
        )


    # -------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------

    st.subheader(
        "🔍 Feature Importance"
    )


    importance_data = pd.DataFrame(

        {

            "Feature": features,

            "Importance": model.feature_importances_

        }

    ).sort_values(

        "Importance",
        ascending=False

    )


    st.bar_chart(

        importance_data.set_index(
            "Feature"
        ),

        use_container_width=True

    )


    # -------------------------------------------------
    # ACTUAL VS PREDICTED
    # -------------------------------------------------

    st.subheader(
        "🎯 Actual vs Predicted Prices"
    )


    prediction_data = pd.DataFrame(

        {

            "Actual Price": y_test.values,

            "Predicted Price": predictions

        },

        index=y_test.index

    )


    st.line_chart(

        prediction_data,

        use_container_width=True

    )


# =====================================================
# TAB 4 - HISTORICAL DATA
# =====================================================

with tab4:

    st.subheader(
        "📋 Historical Stock Data"
    )


    display_data = data.tail(100)


    st.dataframe(

        display_data,

        use_container_width=True

    )


    csv_data = data.to_csv()


    st.download_button(

        label="⬇️ Download Historical Data",

        data=csv_data,

        file_name=(
            selected_stock
            .replace(" ", "_")
            +
            "_historical_data.csv"
        ),

        mime="text/csv"

    )


# =====================================================
# TAB 5 - STOCK COMPARISON
# =====================================================

with tab5:

    st.subheader(
        "🆚 Compare Two Stocks"
    )


    comparison_col1, comparison_col2 = st.columns(2)


    stock1 = comparison_col1.selectbox(

        "Select First Stock",

        list(stocks.keys()),

        index=0,

        key="stock1"

    )


    stock2 = comparison_col2.selectbox(

        "Select Second Stock",

        list(stocks.keys()),

        index=1,

        key="stock2"

    )


    if stock1 == stock2:

        st.warning(
            "Please select two different stocks."
        )


    else:

        with st.spinner(
            "Loading comparison data..."
        ):

            data1 = load_data(

                stocks[stock1],

                period

            )

            data2 = load_data(

                stocks[stock2],

                period

            )


        if data1.empty or data2.empty:

            st.error(
                "Unable to retrieve comparison data."
            )


        else:

            # -----------------------------------------
            # CLEAN CLOSE DATA
            # -----------------------------------------

            close1 = data1[
                "Close"
            ].dropna()

            close2 = data2[
                "Close"
            ].dropna()


            if close1.empty or close2.empty:

                st.error(
                    "Current price data is unavailable."
                )

            else:

                comparison_data = pd.DataFrame(

                    {

                        stock1: close1,

                        stock2: close2

                    }

                ).dropna()


                # -------------------------------------
                # HISTORICAL COMPARISON
                # -------------------------------------

                st.subheader(
                    "📈 Historical Price Comparison"
                )


                st.line_chart(

                    comparison_data,

                    use_container_width=True

                )


                # -------------------------------------
                # CURRENT PRICE COMPARISON
                # -------------------------------------

                st.subheader(
                    "📊 Current Price Comparison"
                )


                compare1, compare2 = st.columns(2)


                latest_price1 = close1.iloc[-1]

                latest_price2 = close2.iloc[-1]


                with compare1:

                    st.metric(

                        stock1,

                        f"₹{latest_price1:,.2f}"

                    )


                with compare2:

                    st.metric(

                        stock2,

                        f"₹{latest_price2:,.2f}"

                    )


                st.info(

                    "The comparison chart displays the "
                    "historical closing prices of the selected stocks."

                )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")


st.caption(

    "StockVision | Educational stock analysis "
    "and machine learning prediction project. "
    "Predictions are for educational purposes only "
    "and are not financial advice."

)
