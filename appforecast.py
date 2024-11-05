import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import requests
import streamlit.components.v1 as components
from streamlit_cookies_manager import EncryptedCookieManager
from streamlit_cookies_controller import CookieController
import warnings
warnings.filterwarnings("ignore")
from data_plot import (plot_correlation, plot_distribution,
                       plot_countplot, plot_countplot_categoric,
                       plot_aggregasi_data, plot_ai_interpretation,
                       plot_shap_plot, multiplot_ai_interpretation)
from data_describe import display_statistics
from data_cleaning import (display_data, read_file, clean_data, 
                           format_date_columns, time_series_formatting,
                           handle_upload, handle_data_cleaning, handle_date_formatting,
                           handle_time_series_formatting)
from data_prep import (diff_data,line_plot, plot_statistik, uji_stasioner,
                       time_series_split, splitting_plot, handle_time_series_split,
                       handle_plot_statistik, regression_formatting, handle_regression_formatting)
from modelling import (handle_model_training, train_model_timeseries, evaluate_plot_model_results,
                       train_model_regresi, input_new_data, predict_new_data)

def main():

    st.title("Selamat Datang di Tools Analisis Data & Peramalan Pentagon Dinas Kominfotik Gorontalo")

    # Upload Data
    df = handle_upload()

    if df is not None:
        st.subheader('Menampilkan Dataframe Awal :')
        display_data(df)
    else:
        st.stop()
    
    # Pembersihan Data
    df = handle_data_cleaning(df)

    # Statistik Deskriptif
    display_statistics(df)

    # Plot
    numeric = df.select_dtypes(include=['int', 'float'])
    categorical = df.select_dtypes(include=['object'])
    for x in categorical.columns:
        if categorical[x].nunique() > 10:
            categorical.drop(x, axis=1, inplace=True)

    # Plot Korelasi
    corr_img = plot_correlation(numeric)
    if corr_img:
        plot_ai_interpretation(corr_img, 'corr')

    #Plot Distribusi Data (Persebaran Data)
    dist_img = plot_distribution(df, numeric.columns)
    if dist_img:
        plot_ai_interpretation(dist_img, 'dist')

    #Plot Countplot
    count_img = plot_countplot(df, categorical.columns)
    if count_img:
        plot_ai_interpretation(count_img, 'count')

    #Plot Countplot Categoric
    count_img_categoric = plot_countplot_categoric(df, categorical.columns)
    if count_img_categoric:
        plot_ai_interpretation(count_img_categoric, 'count_categoric')

    #Plot Aggregasi Data
    agg_img = plot_aggregasi_data(df, df.columns)
    if agg_img:
        plot_ai_interpretation(agg_img, 'agg')


    lanjut = st.radio("Apakah Anda ingin melanjutkan ke tahap peramalan?", ('Ya', 'Tidak'), index=None)
    if lanjut == 'Ya':
        methods = st.radio("Pilih Metode Peramalan", ('Time Series', 'Regression'), index=None)
        if methods == 'Time Series':
            st.header("Peramalan Berderet Waktu")

            # Format Data Tanggal
            df, freq = handle_date_formatting(df)
        
            # Format Data Time Series
            df = handle_time_series_formatting(df)

            # Plot Data
            line_plot(df)

            # Data Preparation
            train, test = handle_time_series_split(df)

            st.write(train.head())
            st.write(test.tail())
            # Sequence Length
            sequence_length = st.number_input(
                "Masukkan panjang sequence (menentukan seberapa banyak data sebelumnya yang dilihat oleh model saat membuat prediksi):", 
                min_value=1, 
                step=1,
                help="Panjang sequence diperlukan untuk LSTM (menentukan seberapa banyak data sebelumnya yang dilihat oleh model saat membuat prediksi).",
                value=None
            )
            if sequence_length:
                pass
            else:
                st.warning("Masukkan panjang sequence.")
                st.stop()

            # Frequency Selection
            freq = st.selectbox(
                "Pilih frekuensi data:",
                options=['Harian', 'Bulanan', 'Triwulan', 'Tahunan'],
                help="Pilih frekuensi data yang sesuai dengan dataset Anda.",
                index= None
            )
            if freq:
                pass
            else:
                st.warning("Pilih frekuensi data.")
                st.stop()

            # Forecast Horizon
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
                help=f"Jumlah {freq.lower()} ke depan yang akan diprediksi.",
                value=None
            )
            if forecast_horizon:
                pass
            else:
                st.warning("Masukkan periode untuk forecasting.")
                st.stop()

            img = handle_model_training(train, test, sequence_length, freq, forecast_horizon)
            if img:
                multiplot_ai_interpretation(img, 'timeseries')
            else:
                st.warning("Klik tombol 'Train Model' untuk melatih model peramalan.")
                st.stop()
            
            
        elif methods == 'Regression':
            st.header("Peramalan dengan Regresi")

            X_train, X_test, y_train, y_test, X, y = handle_regression_formatting(df)

            model, y_pred, mse, mae, rmse, mape, r2 = train_model_regresi(X_train, X_test, y_train, y_test)

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


            shap_plot = plot_shap_plot(model, X_test)
            if shap_plot:
                plot_ai_interpretation(shap_plot, 'shap')
            
            new_data = input_new_data(df, X)

            predict_new_data(model, new_data)

        else:
            st.warning("Pilih metode Prediksi.")
            st.stop()
    elif lanjut == 'Tidak':
        st.write("Analisis Selesai")
    else:
        st.warning("Pilih opsi untuk melanjutkan atau tidak.")
        st.stop()
    

if __name__ == '__main__':
    main()