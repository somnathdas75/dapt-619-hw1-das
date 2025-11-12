"""
Stock Price Analysis and Prediction Dashboard
A Streamlit app for historical data analysis, visualization, and 12-month price prediction
By Somnath Das
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Analysis & Prediction Dashboard By Somnath Das - 12 Month Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def download_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        data.reset_index(inplace=True)
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

def extract_stock_name_from_filename(filename):
    """Extract stock name from filename (last 4 letters before extension)"""
    try:
        # Remove file extension and get the last 4 characters
        name_without_ext = os.path.splitext(filename)[0]
        if len(name_without_ext) >= 4:
            return name_without_ext[-4:].upper()
        else:
            return name_without_ext.upper()
    except:
        return "STOCK"

def load_uploaded_data(uploaded_file):
    """Load data from uploaded Excel/CSV file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def prepare_features(df):
    """Create technical indicators and features"""
    data = df.copy()
    
    # Ensure we have required columns
    if 'close' not in data.columns:
        st.error("Data must have a 'close' column")
        return None
    
    # Moving averages
    data['ma_7'] = data['close'].rolling(window=7).mean()
    data['ma_21'] = data['close'].rolling(window=21).mean()
    data['ma_50'] = data['close'].rolling(window=50).mean()
    data['ma_200'] = data['close'].rolling(window=200).mean()
    
    # Returns
    data['returns'] = data['close'].pct_change()
    
    # Volatility
    data['volatility'] = data['returns'].rolling(window=21).std()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['signal_line'] = data['macd'].ewm(span=9, adjust=False).mean()
    
    # Lag features
    for i in [1, 2, 3, 5, 10]:
        data[f'close_lag_{i}'] = data['close'].shift(i)
    
    # Volume features if available
    if 'volume' in data.columns:
        data['volume_ma_7'] = data['volume'].rolling(window=7).mean()
        data['volume_change'] = data['volume'].pct_change()
    
    return data

def train_models(data):
    """Train multiple ML models for prediction"""
    # Prepare features
    feature_cols = ['ma_7', 'ma_21', 'ma_50', 'ma_200', 'volatility', 
                    'rsi', 'macd', 'signal_line', 'close_lag_1', 
                    'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_10']
    
    # Add volume features if available
    if 'volume' in data.columns:
        feature_cols.extend(['volume_ma_7', 'volume_change'])
    
    # Remove rows with NaN
    model_data = data.dropna()
    
    X = model_data[feature_cols]
    y = model_data['close']
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train models
    models = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_scaled, y)
    models['Linear Regression'] = lr
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    models['Random Forest'] = rf
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_scaled, y)
    models['Gradient Boosting'] = gb
    
    return models, scaler, feature_cols

def predict_future(data, models, scaler, feature_cols, months=12):
    """Predict future prices for specified months"""
    future_data = data.copy()
    predictions = {name: [] for name in models.keys()}
    predictions['Ensemble'] = []
    dates = []
    
    last_date = pd.to_datetime(data['date'].iloc[-1])
    days_to_predict = months * 30  # Approximate 30 days per month

    print("days_to_predict:", days_to_predict)    
    for i in range(days_to_predict):
        # Get features from last row
        future_data = prepare_features(future_data)
        last_features = future_data[feature_cols].iloc[-1:].values
        
        # Handle NaN values
        if np.isnan(last_features).any():
            last_features = np.nan_to_num(last_features, nan=np.nanmean(last_features))
        
        last_features_scaled = scaler.transform(last_features)
        
        # Make predictions with each model
        model_preds = []
        for name, model in models.items():
            pred = model.predict(last_features_scaled)[0]
            predictions[name].append(pred)
            model_preds.append(pred)
        
        # Ensemble average
        ensemble_pred = np.mean(model_preds)
        predictions['Ensemble'].append(ensemble_pred)
        
        # Create next date
        next_date = last_date + timedelta(days=i+1)
        dates.append(next_date)
        
        # Add predicted row
        new_row = future_data.iloc[-1:].copy()
        new_row['date'] = next_date
        new_row['close'] = ensemble_pred
        
        future_data = pd.concat([future_data, new_row], ignore_index=True)
    
    # Create predictions dataframe
    pred_df = pd.DataFrame(predictions)
    pred_df['date'] = dates
    
    return pred_df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_closing_prices(data):
    """Plot historical closing prices"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['date'], 
        y=data['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#2E86AB', width=2)
    ))
    
    # Add moving averages if available
    if 'ma_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['ma_50'],
            mode='lines',
            name='MA 50',
            line=dict(color='#A23B72', width=1, dash='dash')
        ))
    
    if 'ma_200' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['ma_200'],
            mode='lines',
            name='MA 200',
            line=dict(color='#F18F01', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title='Historical Closing Prices',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_volume(data):
    """Plot volume over time"""
    if 'volume' not in data.columns:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data['date'], 
        y=data['volume'],
        name='Volume',
        marker_color='#06A77D'
    ))
    
    fig.update_layout(
        title='Trading Volume Over Time',
        xaxis_title='Date',
        yaxis_title='Volume',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_yearly_performance(data):
    """Plot yearly aggregated data"""
    yearly_data = data.copy()
    yearly_data['year'] = pd.to_datetime(yearly_data['date']).dt.year
    
    yearly_stats = yearly_data.groupby('year').agg({
        'close': ['mean', 'min', 'max', 'first', 'last'],
        'volume': 'sum' if 'volume' in yearly_data.columns else 'count'
    }).reset_index()
    
    yearly_stats.columns = ['year', 'avg_close', 'min_close', 'max_close', 
                            'open_close', 'close_close', 'total_volume']
    yearly_stats['yearly_return'] = ((yearly_stats['close_close'] - yearly_stats['open_close']) 
                                      / yearly_stats['open_close'] * 100)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Average Yearly Close Price', 'Yearly Returns (%)'),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    # Average close price
    fig.add_trace(
        go.Bar(x=yearly_stats['year'], y=yearly_stats['avg_close'],
               name='Avg Close', marker_color='#2E86AB'),
        row=1, col=1
    )
    
    # Yearly returns
    colors = ['#06A77D' if x >= 0 else '#D62828' for x in yearly_stats['yearly_return']]
    fig.add_trace(
        go.Bar(x=yearly_stats['year'], y=yearly_stats['yearly_return'],
               name='Return %', marker_color=colors),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig, yearly_stats

def plot_predictions(historical_data, predictions, prediction_months):
    """Plot historical data with predictions"""
    fig = go.Figure()
    
    # Historical data (last year for better visualization)
    recent_data = historical_data.tail(252)  # Approximately 1 year of trading days
    fig.add_trace(go.Scatter(
        x=recent_data['date'], 
        y=recent_data['close'],
        mode='lines',
        name='Historical',
        line=dict(color='#2E86AB', width=2)
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=predictions['date'], 
        y=predictions['Ensemble'],
        mode='lines',
        name='Predicted (Ensemble)',
        line=dict(color='#D62828', width=2, dash='dash')
    ))
    
    # Prediction range
    if 'Random Forest' in predictions.columns and 'Linear Regression' in predictions.columns:
        fig.add_trace(go.Scatter(
            x=predictions['date'],
            y=predictions['Random Forest'],
            mode='lines',
            name='RF Prediction',
            line=dict(color='lightblue', width=1),
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=predictions['date'],
            y=predictions['Linear Regression'],
            mode='lines',
            name='LR Prediction',
            line=dict(color='lightcoral', width=1),
            opacity=0.5
        ))
    
    fig.update_layout(
        title=f'Historical Data + {prediction_months}-Month Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("📈 Stock Analysis & Prediction Dashboard by Somnath Das Nov 2025")
    st.markdown("---")
    
    # Initialize stock_name variable
    stock_name = "Give Your Stock Name"  # Default value
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        data_source = st.radio(
            "Select Data Source:",
            ["Upload File (CSV/Excel)", "Download from Yahoo Finance"]
        )
        
        if data_source == "Upload File (CSV/Excel)":
            uploaded_file = st.file_uploader(
                "Upload your stock data file",
                type=['csv', 'xlsx', 'xls']
            )
            if uploaded_file is not None:
                stock_name = extract_stock_name_from_filename(uploaded_file.name)
            ticker_display = st.text_input("Stock Ticker (for display)", value=stock_name)
            ticker_symbol = ticker_display
        else:
            ticker_symbol = st.text_input("Enter Stock Ticker", "NVDA")
            stock_name = ticker_symbol
            start_date = st.date_input("Start Date", datetime(2018, 1, 1))
            end_date = st.date_input("End Date", datetime.now())
            uploaded_file = None
        
        prediction_months = st.slider("Prediction Period (Months)", 1, 24, 12)
        print(" prediction_months : ", prediction_months)
        analyze_button = st.button("🚀 Analyze & Predict", type="primary")
    
    # Main content
    if analyze_button:
        with st.spinner("Loading and processing data..."):
            # Load data
            if data_source == "Upload File (CSV/Excel)":
                if uploaded_file is not None:
                    data = load_uploaded_data(uploaded_file)
                    stock_name = extract_stock_name_from_filename(uploaded_file.name)
                else:
                    st.warning("Please upload a file")
                    return
            else:
                data = download_stock_data(ticker_symbol, start_date, end_date)
                stock_name = ticker_symbol
            
            if data is None or len(data) == 0:
                st.error("No data available")
                return
            
            # Display stock name
            st.subheader(f"Analyzing: {stock_name}")
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trading Days", len(data))
            with col2:
                st.metric("Current Price", f"${data['close'].iloc[-1]:.2f}")
            with col3:
                st.metric("Highest Price", f"${data['close'].max():.2f}")
            with col4:
                st.metric("Lowest Price", f"${data['close'].min():.2f}")
            
            st.markdown("---")
            
            # Prepare features
            with st.spinner("Preparing technical indicators..."):
                processed_data = prepare_features(data)
            
            # Tab layout
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Price Chart", 
                "📈 Volume Analysis", 
                "📅 Yearly Performance",
                "🔮 Predictions",
                "💾 Download Data"
            ])
            
            with tab1:
                st.subheader("Historical Closing Prices")
                fig = plot_closing_prices(processed_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Price statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_return = ((data['close'].iloc[-1] - data['close'].iloc[0]) 
                                   / data['close'].iloc[0] * 100)
                    st.metric("Total Return", f"{total_return:.2f}%")
                with col2:
                    avg_daily_return = processed_data['returns'].mean() * 100
                    st.metric("Avg Daily Return", f"{avg_daily_return:.4f}%")
                with col3:
                    volatility = processed_data['returns'].std() * 100
                    st.metric("Volatility (Std Dev)", f"{volatility:.4f}%")
            
            with tab2:
                st.subheader("Trading Volume Analysis")
                if 'volume' in data.columns:
                    fig_vol = plot_volume(data)
                    if fig_vol:
                        st.plotly_chart(fig_vol, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg Daily Volume", f"{data['volume'].mean():,.0f}")
                    with col2:
                        st.metric("Total Volume", f"{data['volume'].sum():,.0f}")
                else:
                    st.info("Volume data not available in dataset")
            
            with tab3:
                st.subheader("Yearly Performance Overview")
                fig_yearly, yearly_stats = plot_yearly_performance(data)
                st.plotly_chart(fig_yearly, use_container_width=True)
                
                st.subheader("Yearly Statistics Table")
                st.dataframe(
                    yearly_stats.style.format({
                        'avg_close': '${:.2f}',
                        'min_close': '${:.2f}',
                        'max_close': '${:.2f}',
                        'yearly_return': '{:.2f}%'
                    }),
                    use_container_width=True
                )
            
            with tab4:
                st.subheader(f"Price Predictions for Next {prediction_months} Months")
                
                with st.spinner("Training prediction models..."):
                    models, scaler, feature_cols = train_models(processed_data)
                    predictions = predict_future(
                        data, models, scaler, feature_cols, months=prediction_months
                    )
                
                print("Prediction time in months:", prediction_months)
                fig_pred = plot_predictions(data, predictions, prediction_months)
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Prediction summary
                st.subheader("Prediction Summary")
                col1, col2, col3 = st.columns(3)
                
                current_price = data['close'].iloc[-1]
                predicted_price = predictions['Ensemble'].iloc[-1]
                predicted_change = ((predicted_price - current_price) / current_price) * 100
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric(
                        f"Predicted Price ({prediction_months}M)", 
                        f"${predicted_price:.2f}",
                        f"{predicted_change:+.2f}%"
                    )
                with col3:
                    monthly_return = (predicted_change / prediction_months)
                    st.metric("Avg Monthly Return", f"{monthly_return:.2f}%")
                
                # Show prediction range
                st.info(f"""
                **Prediction Range:**
                - High (Random Forest): ${predictions['Random Forest'].iloc[-1]:.2f}
                - Low (Linear Regression): ${predictions['Linear Regression'].iloc[-1]:.2f}
                - Mid (Gradient Boosting): ${predictions['Gradient Boosting'].iloc[-1]:.2f}
                """)
            
            with tab5:
                st.subheader("Download Data")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Historical Data**")
                    csv_hist = data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Historical Data (CSV)",
                        data=csv_hist,
                        file_name=f"{stock_name}_historical_data.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.write("**Prediction Data**")
                    csv_pred = predictions.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv_pred,
                        file_name=f"{stock_name}_predictions_{prediction_months}m.csv",
                        mime="text/csv"
                    )
                
                st.write("**Preview of Data**")
                st.dataframe(data.tail(10), use_container_width=True)
    
    else:
        # Welcome message
        st.info("""
        👋 Welcome to the Stock Analysis & Prediction Dashboard!
        
        **Features:**
        - 📊 Comprehensive historical price analysis
        - 📈 Volume and trading pattern visualization
        - 📅 Yearly performance breakdown
        - 🔮 AI-powered price predictions up to 24 months
        - 💾 Download data in CSV format
        
        **Getting Started:**
        1. Choose your data source from the sidebar
        2. Upload your file OR enter a stock ticker
        3. Click "Analyze & Predict" to begin
        
        **Note:** Predictions are based on historical patterns and should not be considered financial advice.
        """)
        
        st.markdown("---")
        st.subheader("📋 Sample Data Format")
        st.write("Your uploaded file should have these columns:")
        sample_df = pd.DataFrame({
            'date': ['12/12/1980', '12/15/1980', '12/16/1980'],
            'open': [0.0984, 0.0937, 0.0869],
            'high': [0.0989, 0.0937, 0.0869],
            'low': [0.0984, 0.0933, 0.0864],
            'close': [0.0984, 0.0933, 0.0864],
            'volume': [1605922, 691965, 361853]
        })
        st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    main()