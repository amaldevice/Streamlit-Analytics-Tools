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

controller = CookieController()


cookies = EncryptedCookieManager(
    prefix="streamlit_app",  # Prefix untuk cookie kamu
    password="YOUR_SECRET_KEY",  # Gantilah ini dengan kunci rahasia
)
if not cookies.ready():
    st.stop()  # Hentikan eksekusi aplikasi sampai cookie siap

REDIRECT_URI = "http://localhost:8002?callback"
APP_URL_SSO = "https://dev1.gorontaloprov.go.id"
AUTHORIZATION_URL = f"{APP_URL_SSO}/oauth/authorize"
ACCESS_TOKEN_URL = f"{APP_URL_SSO}/oauth/token"
RESOURCE_URL = f"{APP_URL_SSO}/api/v2/me"
REDIRECT_URL = f"{REDIRECT_URI}"
LOGOUT_URL = f"{APP_URL_SSO}/logout"
CLIENT_ID = 17
CLIENT_SECRET = "qWdID0Pc9M3MxcMDi5Zl60brQjs5mRkQ6G6CbbUj"
USER_IDENTIFIER = ""
SCOPES = ""
SSO_LOGO_URL = "{APP_URL_SSO}/images/logo-small.png"


token_url = f"{APP_URL_SSO}/oauth/token"
userinfo_url = f"{APP_URL_SSO}/api/v2/me"

# Fungsi untuk login menggunakan Auth0


def login():
    login_url = f"{AUTHORIZATION_URL}?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
    st.write(f"[Login dengan Auth0]({login_url})")

# Fungsi untuk menangani callback dari Auth0 dan mengambil token


def handle_callback():
    # code = st.query_params['code'];
    # code = st.experimental_get_query_params().get('code')
    code = st.query_params.get('code')
    if code:
        payload = {
            'grant_type': 'authorization_code',
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'code':  code,
            'redirect_uri': REDIRECT_URI
        }
        token_info = requests.post(token_url, json=payload).json()
        access_token = token_info.get('access_token')
        # st.write(f"{access_token}")

        # Mengambil informasi pengguna
        if access_token:
            headers = {'Authorization': f'Bearer {access_token}'}
            user_info = requests.get(userinfo_url, headers=headers).json()

            controller.set('user_info', json.dumps(user_info))

            return user_info
    return None

# Halaman utama Streamlit
def sudahLogin():
    cookie = controller.get('user_info')
    if cookie:
        if isinstance(cookie, dict):
            user_info = cookie  # Directly use the dictionary
        else:
            user_info = json.loads(cookie)  # Parsing JSON string ke dictionary
        st.write(f"Selamat datang, {user_info.get('nama', 'Pengguna')}!")
        st.write(f"Email: {user_info.get('email', 'Tidak ada email')}")

        if st.button("Logout"):
            controller.remove('user_info')
            st.rerun()  # Ganti dengan st.rerun() yang baru

        st.header("Selamat Datang di Tools Analisis Data & Peramalan  Pentagon Dinas Kominfotik Gorontalo")

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
            if st.button("Logout"):
                controller.remove('user_info')
                st.rerun()  # Ganti dengan st.rerun() yang baru
            st.stop()
        else:
            st.warning("Pilih opsi untuk melanjutkan atau tidak.")
            st.stop()


def main():
    st.title("Streamlit SSO dengan Auth0")
    cookie = controller.get('user_info')
    if not cookie:
        user_info = handle_callback()
        if user_info:
            sudahLogin()
            
        else : 
            login()
        
    else:
        sudahLogin()
        

if __name__ == '__main__':
    main()