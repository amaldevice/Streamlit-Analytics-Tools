import pandas as pd
import numpy as np
import streamlit as st
import math
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor, train
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from neuralprophet import NeuralProphet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import math


# --------------------- Model-Specific Functions ---------------------

def train_arima(train, test):
    st.subheader("Pengaturan ARIMA")
    p = st.number_input("Masukkan nilai p:", min_value=0, value=1, step=1)
    d = st.number_input("Masukkan nilai d:", min_value=0, value=1, step=1)
    q = st.number_input("Masukkan nilai q:", min_value=0, value=1, step=1)
    
    try:
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        y_pred = model_fit.forecast(steps=len(test))
        y_pred = pd.Series(y_pred, index=test.index)
        return y_pred
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam pelatihan model ARIMA: {e}")
        st.stop()   

def train_sarima(train, test):
    st.subheader("Pengaturan SARIMA")
    p = st.number_input("Masukkan nilai p (SARIMA):", min_value=0, value=1, step=1)
    d = st.number_input("Masukkan nilai d (SARIMA):", min_value=0, value=1, step=1)
    q = st.number_input("Masukkan nilai q (SARIMA):", min_value=0, value=1, step=1)
    P = st.number_input("Masukkan nilai P (SARIMA):", min_value=0, value=1, step=1)
    D = st.number_input("Masukkan nilai D (SARIMA):", min_value=0, value=1, step=1)
    Q = st.number_input("Masukkan nilai Q (SARIMA):", min_value=0, value=1, step=1)
    s = st.number_input("Masukkan nilai s (SARIMA):", min_value=1, value=12, step=1)  # seasonal_period
    
    try:
        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit(disp=False)
        y_pred = model_fit.forecast(steps=len(test))
        y_pred = pd.Series(y_pred, index=test.index)
        return y_pred
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam pelatihan model SARIMA: {e}")
        st.stop()

def train_exponential_smoothing(train, test):
    st.subheader("Pengaturan Exponential Smoothing")
    trend_option = st.radio("Pilih jenis trend:", ('add', 'mul', None), index=0)
    seasonal_option = st.radio("Pilih jenis seasonal:", ('add', 'mul', None), index=0)
    seasonal_periods = st.number_input("Masukkan jumlah seasonal periods:", min_value=1, value=12, step=1)
    
    try:
        model = ExponentialSmoothing(
            train,
            trend=trend_option,
            seasonal=seasonal_option,
            seasonal_periods=seasonal_periods
        )
        model_fit = model.fit()
        y_pred = model_fit.forecast(steps=len(test))
        y_pred = pd.Series(y_pred, index=test.index)
        return y_pred
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam pelatihan model ExponentialSmoothing: {e}")
        st.stop()

def train_prophet_model(train, test):
    st.subheader("Pengaturan Prophet")
    # Preprocessing
    if isinstance(train, pd.DataFrame):
        if train.shape[1] != 1:
            st.warning("Pilih satu kolom target untuk peramalan.")
            target_column = st.selectbox("Pilih kolom target:", train.columns)
            train_prophet = train.reset_index().rename(columns={train.index.name: 'ds', target_column: 'y'})
            test_prophet = test.reset_index().rename(columns={test.index.name: 'ds', target_column: 'y'})
        else:
            target_column = train.columns[0]
            train_prophet = train.reset_index().rename(columns={train.index.name: 'ds', target_column: 'y'})
            test_prophet = test.reset_index().rename(columns={test.index.name: 'ds', target_column: 'y'})
    elif isinstance(train, pd.Series):
        train_prophet = train.reset_index().rename(columns={train.index.name: 'ds', train.name: 'y'})
        test_prophet = test.reset_index().rename(columns={test.index.name: 'ds', test.name: 'y'})
    else:
        st.error("Format data tidak dikenali untuk Prophet.")
        st.stop()
    
    try:
        model = Prophet()
        model.fit(train_prophet)
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)
        y_pred = forecast['yhat'].iloc[-len(test):].values
        y_pred = pd.Series(y_pred, index=test.index)
        return y_pred
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam pelatihan model Prophet: {e}")
        st.stop()

def train_neuralprophet_model(train, test):
    st.subheader("Pengaturan NeuralProphet")
    # Preprocessing
    if isinstance(train, pd.DataFrame):
        if train.shape[1] != 1:
            st.warning("Pilih satu kolom target untuk peramalan.")
            target_column = st.selectbox("Pilih kolom target:", train.columns)
            train_neuralprophet = train.reset_index().rename(columns={train.index.name: 'ds', target_column: 'y'})
            test_neuralprophet = test.reset_index().rename(columns={test.index.name: 'ds', target_column: 'y'})
        else:
            target_column = train.columns[0]
            train_neuralprophet = train.reset_index().rename(columns={train.index.name: 'ds', target_column: 'y'})
            test_neuralprophet = test.reset_index().rename(columns={test.index.name: 'ds', target_column: 'y'})
    elif isinstance(train, pd.Series):
        train_neuralprophet = train.reset_index().rename(columns={train.index.name: 'ds', train.name: 'y'})
        test_neuralprophet = test.reset_index().rename(columns={test.index.name: 'ds', test.name: 'y'})
    else:
        st.error("Format data tidak dikenali untuk NeuralProphet.")
        st.stop()
    
    try:
        model = NeuralProphet()
        freq = st.selectbox("Pilih frekuensi data (freq):", ['D', 'W', 'M', 'Q', 'Y'], index=None)
        model.fit(train_neuralprophet, freq=freq)  # Adjust 'freq' as per user selection
        future = model.make_future_dataframe(train_neuralprophet, periods=len(test))
        forecast = model.predict(future)
        y_pred = forecast['yhat1'].iloc[-len(test):].values
        y_pred = pd.Series(y_pred, index=test.index)
        return y_pred
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam pelatihan model NeuralProphet: {e}")
        st.stop()

def train_lstm_model(train, test):
    st.subheader("Pengaturan LSTM")
    
    # Step 1: Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    try:
        train_values = train.values.reshape(-1, 1) if isinstance(train, (pd.DataFrame, pd.Series)) else np.array(train).reshape(-1, 1)
        test_values = test.values.reshape(-1, 1) if isinstance(test, (pd.DataFrame, pd.Series)) else np.array(test).reshape(-1, 1)
        train_scaled = scaler.fit_transform(train_values)
        test_scaled = scaler.transform(test_values)
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam scaling data: {e}")
        st.stop()
    
    # Step 2: Sequence creation
    sequence_length = st.number_input("Masukkan panjang sequence:", min_value=1, step=1, help="Panjang sequence diperlukan untuk LSTM.", value=None)
    if sequence_length is None:
        st.warning("Masukkan panjang sequence untuk melanjutkan.")
        st.stop()

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_test_seq, y_test = create_sequences(test_scaled, sequence_length)
    
    if len(X_train) == 0 or len(X_test_seq) == 0:
        st.error("Panjang sequence terlalu besar untuk data yang diberikan.")
        st.stop()
    
    # Reshape for LSTM: [samples, time_steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))
    
    # Step 3: Modeling
    try:
        model = Sequential()
        model.add(LSTM(150, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        with st.spinner("Melatih model LSTM..."):
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test_seq, y_test),
                verbose=0
            )
        st.success("Pelatihan model LSTM selesai.")
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam pelatihan model LSTM: {e}")
        st.stop()
    
    # Step 4: Predict
    try:
        y_pred_scaled = model.predict(X_test_seq)
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        y_pred = pd.Series(y_pred, index=test.index[sequence_length:])
        return y_pred, sequence_length
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam prediksi dan inverse scaling: {e}")
        st.stop()

# --------------------- Main Training Function ---------------------

def train_model_timeseries(train, test):
    """
    Train time series model based on user selection.
    """
    st.header("Training Model Peramalan")
    model_option = st.radio(
        "Pilih Model Peramalan:",
        ('ARIMA', 'SARIMA', 'ExponentialSmoothing', 'Prophet', 'NeuralProphet', 'LSTM'),
        index=0
    )

    st.write(f"Model peramalan yang dipilih adalah: {model_option}")

    # Dictionary to map model options to their respective functions
    model_functions = {
        'ARIMA': train_arima,
        'SARIMA': train_sarima,
        'ExponentialSmoothing': train_exponential_smoothing,
        'Prophet': train_prophet_model,
        'NeuralProphet': train_neuralprophet_model,
        'LSTM': train_lstm_model
    }

    if model_option not in model_functions:
        st.warning("Pilih model peramalan terlebih dahulu.")
        st.stop()

    # Train the selected model
    if model_option == 'LSTM':
        y_pred, sequence_length = model_functions[model_option](train, test)
    else:
        y_pred = model_functions[model_option](train, test)
        sequence_length = 0  # Tidak digunakan untuk model lain

    # Prepare y_true based on y_pred's index to ensure konsistensi
    y_true = test.loc[y_pred.index]

    # Compute metrics
    try:
        y_true_array = y_true.values
        y_pred_array = y_pred.values

        mse = mean_squared_error(y_true_array, y_pred_array)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_true_array, y_pred_array)
        mape = mean_absolute_percentage_error(y_true_array, y_pred_array)
        r2 = r2_score(y_true_array, y_pred_array)
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam perhitungan metrik: {e}")
        st.stop()

    st.subheader("Hasil Evaluasi Model")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"MAE: {mae:.4f}")
    st.write(f"MAPE: {mape:.4f}%")
    st.write(f"R²: {r2:.4f}")

    return y_true, y_pred, mse, mae, rmse, mape, r2

# --------------------- Evaluation and Plotting Functions ---------------------

def evaluate_plot_model_results(train, y_test, y_pred):
    """
    Plot actual vs predicted and residuals.
    """
    st.header("Evaluasi dan Visualisasi Hasil Model")

    # Plot Keseluruhan Data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train.values, label='Train')
    ax.plot(y_test.index, y_test, label='Test')
    ax.plot(y_test.index, y_pred, label='Predicted')
    ax.set_title('Train, Test, dan Predicted')
    ax.legend()
    st.pyplot(fig)

    # Plot Actual vs Predicted
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test, label='Actual')
    ax.plot(y_test.index, y_pred, label='Predicted')
    ax.set_title('Actual vs Predicted')
    ax.legend()
    st.pyplot(fig)

    # Plot Residuals
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, residuals, label='Residuals')
    ax.hlines(0, y_test.index.min(), y_test.index.max(), colors='r', linestyles='dashed')
    ax.set_title('Residuals')
    ax.legend()
    st.pyplot(fig)

# --------------------- Handler Function ---------------------

def handle_model_training(train, test):
    """
    Handle model training and evaluation.
    """
    y_true, y_pred, mse, mae, rmse, mape, r2 = train_model_timeseries(train, test)
    evaluate_plot_model_results(pd.Series(train, index=train.index), y_true, y_pred)


