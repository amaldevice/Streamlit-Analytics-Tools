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
                       plot_shap_plot)
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

            # Uji Statistik
            handle_plot_statistik(df)

            # Uji Stasioneritas
            uji_stasioner(df)

            # Differencing
            df = diff_data(df)

            # Data Preparation
            train, test = handle_time_series_split(df)

            st.write(train.head())
            st.write(test.tail())
            # Modelling
            handle_model_training(train, test, freq)
        elif methods == 'Regression':
            st.header("Peramalan dengan Regresi")

            X_train, X_test, y_train, y_test, X, y = handle_regression_formatting(df)

            model, y_pred, mse, mae, rmse, mape, r2 = train_model_regresi(X_train, X_test, y_train, y_test)

            shap_plot = plot_shap_plot(model, X_test)
            if shap_plot:
                plot_ai_interpretation(shap_plot, 'shap')
            
            st.write(f"Mean Squared Error (MSE) : {mse:.2f}")
            st.write(f"Mean Absolute Error (MAE) : {mae:.2f}")
            st.write(f"Root Mean Squared Error (RMSE) : {rmse:.2f}")
            st.write(f"Mean Absolute Percentage Error (MAPE) : {mape:.2f}")
            st.write(f"R2 Score : {r2:.2f}")

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