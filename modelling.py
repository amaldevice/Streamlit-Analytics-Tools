import pandas as pd
import numpy as np
import streamlit as st
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import math

# --------------------- LSTM Model Function ---------------------
    
def train_lstm_model(train, test, freq=None):
    st.subheader("Pengaturan LSTM")
    
    # Step 1: Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    try:
        train_values = train.values.reshape(-1, 1) if isinstance(train, (pd.DataFrame, pd.Series)) else np.array(train).reshape(-1, 1)
        test_values = test.values.reshape(-1, 1) if isinstance(test, (pd.DataFrame, pd.Series)) else np.array(test).reshape(-1, 1)
        full_data = np.concatenate((train_values, test_values), axis=0)
        scaled_data = scaler.fit_transform(full_data)
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam scaling data: {e}")
        st.stop()
    
    # Step 2: Sequence creation
    sequence_length = st.number_input(
        "Masukkan panjang sequence:", 
        min_value=1, 
        step=1,
        help="Panjang sequence diperlukan untuk LSTM.",
        value=None
    )
    if not sequence_length:
        st.warning("Isi terlebih dahulu panjang sequence.")
        st.stop()
    
    # Prompt user to select frequency if freq is None
    if freq is None:
        freq = st.selectbox(
            "Pilih frekuensi data:",
            options=['Harian', 'Bulanan', 'Triwulan', 'Tahunan'],
            help="Pilih frekuensi data yang sesuai dengan dataset Anda."
        )
        if not freq:
            st.warning("Pilih frekuensi data.")
            st.stop()
    
    # Mapping frekuensi ke label forecast_horizon
    freq_label_map = {
        'Harian': "jumlah hari untuk forecasting:",
        'Bulanan': "jumlah bulan untuk forecasting:",
        'Triwulan': "jumlah triwulan untuk forecasting:",
        'Tahunan': "jumlah tahun untuk forecasting:"
    }
    forecast_label = "Masukkan " + freq_label_map.get(freq, "jumlah periode untuk forecasting:")
    
    forecast_horizon = st.number_input(
        forecast_label,
        min_value=1,
        step=1,
        value=None,
        help=f"Jumlah {freq.lower()} ke depan yang akan diprediksi.",
        key='forecast_horizon'
    )
    if not forecast_horizon:
        st.warning("Isi terlebih dahulu jumlah periode untuk forecasting.")
        st.stop()

    st.write(f"Melakukan forecasting untuk {forecast_horizon} {freq.lower()} ke depan.")

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    X_data, y_data = create_sequences(scaled_data, sequence_length)
    split_index = len(train_values) - sequence_length
    X_train = X_data[:split_index]
    y_train = y_data[:split_index]
    X_test_seq = X_data[split_index:]
    y_test = y_data[split_index:]
    
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
    
    # Step 4: Predict on Test Data
    try:
        y_pred_scaled = model.predict(X_test_seq)
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        y_pred = pd.Series(y_pred, index=test.index)
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam prediksi dan inverse scaling: {e}")
        st.stop()
    
    # Step 5: Forecast Future Values
    try:
        forecast = []
        last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
        
        for _ in range(forecast_horizon):
            next_pred_scaled = model.predict(last_sequence)
            next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
            forecast.append(next_pred)
            
            # Update the sequence with the new prediction
            last_sequence = np.concatenate((last_sequence[:, 1:, :], next_pred_scaled.reshape(1, 1, 1)), axis=1)
        
        # Create a date range for the forecasted values
        last_date = test.index[-1]
        
        # Map freq to pandas frequency string
        freq_map = {
            'Harian': 'D',
            'Bulanan': 'M',
            'Triwulan': 'Q',
            'Tahunan': 'Y'
        }
        freq_pandas = freq_map.get(freq, 'D')  # Default to 'D' if not found
        
        # Create date range for forecasted values
        future_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(freq_pandas),
            periods=forecast_horizon,
            freq=freq_pandas
        )
        forecast_series = pd.Series(forecast, index=future_dates)
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam forecast: {e}")
        st.stop()
    
    return y_pred, forecast_series

# --------------------- Main Training Function ---------------------

def train_model_timeseries(train, test, freq):
    """
    Train LSTM time series model and forecast future values.
    """
    st.header("Training Model Peramalan dengan LSTM")

    y_pred, forecast_series = train_lstm_model(train, test, freq)

    # Prepare y_true based on y_pred's index to ensure consistency
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

    return y_true, y_pred, forecast_series

# --------------------- Evaluation and Plotting Functions ---------------------

def evaluate_plot_model_results(train, y_test, y_pred, forecast_series):
    """
    Plot actual vs predicted and residuals, and show forecasted values.
    """
    st.header("Evaluasi dan Visualisasi Hasil Model")

    # Plot Keseluruhan Data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train.values, label='Train')
    ax.plot(y_test.index, y_test, label='Test')
    ax.plot(y_test.index, y_pred, label='Predicted')
    ax.plot(forecast_series.index, forecast_series.values, label='Forecast', linestyle='--')
    ax.set_title('Train, Test, Predicted, dan Forecast')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Plot Actual vs Predicted
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test, label='Actual')
    ax.plot(y_test.index, y_pred, label='Predicted')
    ax.set_title('Actual vs Predicted')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Plot Forecasted Values
    st.subheader("Forecasting untuk Waktu Mendatang")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(forecast_series.index, forecast_series.values, label='Forecast', color='orange')
    ax.set_title('Forecasted Values')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Plot Residuals
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, residuals, label='Residuals')
    ax.hlines(0, y_test.index.min(), y_test.index.max(), colors='r', linestyles='dashed')
    ax.set_title('Residuals')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --------------------- Handler Function ---------------------

def handle_model_training(train, test, freq):
    """
    Handle model training, forecasting, and evaluation.
    """
    y_true, y_pred, forecast_series = train_model_timeseries(train, test, freq)
    evaluate_plot_model_results(pd.Series(train, index=train.index), y_true, y_pred, forecast_series)
