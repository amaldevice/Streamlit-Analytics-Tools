import pandas as pd
import numpy as np
import streamlit as st
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import math
import io

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
        "Masukkan panjang sequence (menentukan seberapa banyak data sebelumnya yang dilihat oleh model saat membuat prediksi):", 
        min_value=1, 
        step=1,
        help="Panjang sequence diperlukan untuk LSTM (menentukan seberapa banyak data sebelumnya yang dilihat oleh model saat membuat prediksi).",
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
        'Harian': "jumlah hari untuk peramalan:",
        'Bulanan': "jumlah bulan untuk peramalan:",
        'Triwulan': "jumlah triwulan untuk peramalan:",
        'Tahunan': "jumlah tahun untuk peramalan:"
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
    st.header("Melatih Model Peramalan dengan LSTM")

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
    # Penjelasan formal untuk setiap metrik
    st.write(f"1. Mean Squared Error (MSE): {mse:.2f}")
    st.write("   MSE mengukur rata-rata kesalahan dengan mengkuadratkan selisih antara nilai prediksi dan nilai aktual. ")
    st.write("   Nilai yang lebih rendah menunjukkan bahwa rata-rata kesalahan model lebih kecil, yang berarti prediksi lebih dekat ke nilai sebenarnya.\n")

    st.write(f"2. Mean Absolute Error (MAE): {mae:.2f}")
    st.write("   MAE menghitung rata-rata dari nilai absolut selisih antara prediksi dan nilai aktual.")
    st.write("   Ini menunjukkan seberapa besar rata-rata kesalahan tanpa memperhitungkan arah kesalahan (terlalu tinggi atau rendah).")
    st.write("   Nilai MAE yang rendah menunjukkan bahwa prediksi rata-rata cukup dekat dengan nilai aktual.\n")

    st.write(f"3. Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write("   RMSE adalah akar kuadrat dari MSE, yang mengembalikan kesalahan pada skala yang sama dengan data asli.")
    st.write("   Metrik ini berguna untuk interpretasi langsung dalam satuan asli data, dengan nilai yang lebih rendah menunjukkan akurasi yang lebih tinggi.\n")

    st.write(f"4. Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    st.write("   MAPE mengukur kesalahan dalam bentuk persentase rata-rata, sehingga memudahkan pemahaman kesalahan relatif terhadap nilai aktual.")
    st.write("   MAPE cocok untuk mengetahui seberapa besar rata-rata kesalahan model dalam konteks persentase dari nilai aktual.\n")

    st.write(f"5. R² Score: {r2:.2f}")
    st.write("   R², atau koefisien determinasi, mengukur seberapa baik model menjelaskan variasi dalam data.")
    st.write("   Nilai yang mendekati 1 menunjukkan bahwa model mampu menjelaskan sebagian besar variasi dalam data.")
    st.write("   Nilai ini adalah indikator seberapa cocok model dalam memberikan prediksi yang akurat.")

    return y_true, y_pred, forecast_series

# --------------------- Evaluation and Plotting Functions ---------------------

def evaluate_plot_model_results(train, y_test, y_pred, forecast_series):
    """
    Plot actual vs predicted and residuals, and show forecasted values,
    and return the plots as img_buffer for further interpretation.
    """
    st.header("Evaluasi dan Visualisasi Hasil Model")

    # List untuk menyimpan buffer gambar dari setiap plot
    img_buffers = []

    # Plot Keseluruhan Data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train.values, label='Data Pelatihan')
    ax.plot(y_test.index, y_test, label='Data Testing')
    ax.plot(y_test.index, y_pred, label='Data Hasil Prediksi')
    ax.plot(forecast_series.index, forecast_series.values, label='Data Hasil Peramalan', linestyle='--')
    ax.set_title('Data Keseluruhan & Hasil Prediksi')
    ax.grid(True)
    ax.legend()
    # Simpan gambar dalam buffer
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_buffers.append(img_buffer)
    st.pyplot(fig)
    plt.close(fig)

    # Plot Actual vs Predicted
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test, label='Data Asli')
    ax.plot(y_test.index, y_pred, label='Data Hasil Prediksi', linestyle='--')
    ax.set_title('Perbandingan Data Asli dan Data Hasil Prediksi')
    ax.grid(True)
    ax.legend()
    # Simpan gambar dalam buffer
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_buffers.append(img_buffer)
    st.pyplot(fig)
    plt.close(fig)

    # Plot Forecasted Values
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(forecast_series.index, forecast_series.values, label='Nilai Peramalan', color='orange')
    ax.set_title(f'Hasil Peramalan Untuk {forecast_series.index.min().date()} - {forecast_series.index.max().date()}')
    ax.grid(True)
    ax.legend()
    # Simpan gambar dalam buffer
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_buffers.append(img_buffer)
    st.pyplot(fig)
    plt.close(fig)

    # Mengembalikan list berisi buffer dari semua gambar
    return img_buffers


# --------------------- Handler Function ---------------------
@st.cache_data
def handle_model_training(train, test, freq):
    """
    Handle model training, forecasting, and evaluation.
    """
    y_true, y_pred, forecast_series = train_model_timeseries(train, test, freq)
    img = evaluate_plot_model_results(pd.Series(train, index=train.index), y_true, y_pred, forecast_series)
    return img

@st.cache_data
def train_model_regresi(X_train, X_test, y_train, y_test):
    """
    Train regression model and evaluate on test data.
    """
    # Preprocessing
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', XGBRegressor(n_estimators=5000, learning_rate=0.1, random_state=0, n_jobs=-1, objective='reg:squarederror', max_depth=5))])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, y_pred, mse, mae, rmse, mape, r2

def input_new_data(df, X):
    """
    Create input form for new data prediction.

    """
    st.subheader("Masukkan Data baru untuk Prediksi ")
    with st.form(key='prediction_form'):
        input_data = {}
        for feature in X:
            if df[feature].dtype == 'object':
                input_val = st.text_input(f"{feature} (string)")
            else:
                input_val = st.number_input(f"{feature} (numeric)", value=0.0)
            input_data[feature] = input_val
        submit_button = st.form_submit_button(label='Prediksi')

        if submit_button:
            input_df = pd.DataFrame([input_data])
            return input_df

        else:
            st.stop()

def predict_new_data(model, input_df):

    prediction = model.predict(input_df)
    return st.success(f"Prediksi: {prediction[0]:.2f}")

