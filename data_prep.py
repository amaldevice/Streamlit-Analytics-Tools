import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from data_cleaning import display_data


@st.cache_data
def line_plot(df):
    st.subheader("Plot Data:")
    plt.figure(figsize=(12,10))
    plt.plot(df)
    st.pyplot(plt)

@st.cache_data
def plot_statistik(df):
    st.subheader("Plot Decomposition, ACF, and PACF")
    
    # Create a grid with 6 rows (4 for decomposition, 1 for ACF, and 1 for PACF)
    fig, ax = plt.subplots(6, 1, figsize=(12, 18))
    
    # Decomposition
    decomposition = seasonal_decompose(df, model='additive', period=1)
    
    ax[0].plot(decomposition.observed)
    ax[0].set_title('Observed')
    
    ax[1].plot(decomposition.trend)
    ax[1].set_title('Trend')
    
    ax[2].plot(decomposition.seasonal)
    ax[2].set_title('Seasonal')
    
    ax[3].plot(decomposition.resid)
    ax[3].set_title('Residual')
    
    # ACF Plot
    plot_acf(df, lags=40, ax=ax[4])
    ax[4].set_title("Autocorrelation Function (ACF)")
    
    # PACF Plot
    plot_pacf(df, lags=40, ax=ax[5])
    ax[5].set_title("Partial Autocorrelation Function (PACF)")
    
    # Adjust layout
    fig.tight_layout()
    
    # Display the combined plot
    st.pyplot(fig)
    
@st.cache_data
def uji_stasioner(df):
    """
    Fungsi untuk menguji stasioneritas data menggunakan uji ADF dan KPSS.
    
    Parameter:
    -----------
    time_series : array-like
        Data deret waktu yang akan diuji.

    Output:
    -----------
    Dicetak:
    - Nilai ADF statistik, p-value, dan kesimpulan
    - Nilai KPSS statistik, p-value, dan kesimpulan
    - Kesimpulan akhir berdasarkan hasil uji ADF dan KPSS
    """
    
    # Uji ADF (Augmented Dickey-Fuller)
    adf_result = adfuller(df)
    adf_statistic = adf_result[0]
    adf_p_value = adf_result[1]
    
    # Menentukan kesimpulan berdasarkan hasil ADF
    if adf_p_value < 0.05:
        adf_conclusion = "Data Stasioner (tolak H0 dari uji ADF)"
    else:
        adf_conclusion = "Data tidak Stasioner (gagal tolak H0 dari uji ADF)"
    
    # Uji KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
    kpss_result = kpss(df, regression='c', nlags='auto')
    kpss_statistic = kpss_result[0]
    kpss_p_value = kpss_result[1]
    
    # Menentukan kesimpulan berdasarkan hasil KPSS
    if kpss_p_value < 0.05:
        kpss_conclusion = "Data tidak Stasioner (tolak H0 dari uji KPSS)"
    else:
        kpss_conclusion = "Data Stasioner (gagal tolak H0 dari uji KPSS)"
    
    # Output hasil uji ADF dan KPSS di Streamlit
    st.write("### Hasil Uji ADF:")
    st.write("""
    **Hipotesis untuk uji ADF (Augmented Dickey-Fuller):**
    - **H0 (Hipotesis Nol)**: Data memiliki unit root, artinya data **tidak Stasioner**.
    - **H1 (Hipotesis Alternatif)**: Data **Stasioner** (tidak memiliki unit root).
    """)
    st.write(f"ADF Statistic: {adf_statistic}")
    st.write(f"ADF p-value: {adf_p_value}")
    st.write(f"Kesimpulan ADF: {adf_conclusion}")
    
    st.write("### Hasil Uji KPSS:")
    st.write("""
    **Hipotesis untuk uji KPSS (Kwiatkowski-Phillips-Schmidt-Shin):**
    - **H0 (Hipotesis Nol)**: Data **Stasioner**.
    - **H1 (Hipotesis Alternatif)**: Data memiliki unit root, artinya data **tidak Stasioner**.
    """)
    st.write(f"KPSS Statistic: {kpss_statistic}")
    st.write(f"KPSS p-value: {kpss_p_value}")
    st.write(f"Kesimpulan KPSS: {kpss_conclusion}")
    
    # Kesimpulan gabungan
    if adf_p_value < 0.05 and kpss_p_value >= 0.05:
        overall_conclusion = "Data Stasioner berdasarkan uji ADF dan KPSS."
    elif adf_p_value < 0.05 and kpss_p_value < 0.05:
        overall_conclusion = "Data Stasioner, tetapi kemungkinan terdapat tren."
    elif adf_p_value >= 0.05 and kpss_p_value >= 0.05:
        overall_conclusion = "Data mungkin Stasioner, tetapi pemeriksaan lebih lanjut diperlukan."
    else:
        overall_conclusion = "Data tidak Stasioner, transformasi diperlukan."

    st.write(f"### Kesimpulan Akhir: {overall_conclusion}")

def diff_data(df):
    diff_option = st.radio("""Berdasarkan hasil uji Stasioneritas diatas 
             , Apakah anda ingin melakukan differencing pada data?"""
             , ('Ya', 'Tidak'), index=None)
    if diff_option  == 'Ya':
        df_diff = df.diff().dropna()
        st.write("Data setelah differencing:")
        st.write(df_diff.head())
        return df_diff
    elif diff_option  == 'Tidak':
        st.write("Data tidak di differencing.")
        return df
    else:
        st.warning("Pilih opsi untuk melakukan differencing pada data.")
        st.stop()

# Splitting Data & Plot Train & Test

@st.cache_data
def time_series_split(df, train_size):
    """
    Membagi DataFrame menjadi set pelatihan dan pengujian berdasarkan ukuran pelatihan.

    Parameters:
    df (pd.DataFrame): DataFrame yang akan dibagi.
    train_size (float): Proporsi data untuk pelatihan (antara 0.1 dan 0.95).

    Returns:
    tuple: (train, test) DataFrame pelatihan dan pengujian.
    """
    train_size_int = int(len(df) * train_size)
    train = df[:train_size_int]
    test = df[train_size_int:]
    return train, test

@st.cache_data
def splitting_plot(train, test):
    """
    Membuat plot pembagian data pelatihan dan pengujian.

    Parameters:
    train (pd.DataFrame): DataFrame pelatihan.
    test (pd.DataFrame): DataFrame pengujian.
    """
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.axvline(train.index[-1], color='r', linestyle='--', lw=2, label='Split Point')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Train-Test Split')
    st.pyplot(plt)
    plt.clf()

def handle_time_series_split(df):
    """
    Menangani proses pembagian data time series berdasarkan pilihan pengguna.

    Parameters:
    df (pd.DataFrame): DataFrame yang akan dibagi.

    Returns:
    tuple: (train, test) DataFrame pelatihan dan pengujian.
    """
    st.header("Pembagian Data Time Series")
    
    # Slider di luar fungsi yang di-cache
    train_size = st.slider(
        "Porsi data training:",
        min_value=0.1,
        max_value=0.95,
        value=0.8,
        step=0.01
    )
    
    # Memanggil fungsi yang di-cache dengan parameter train_size
    train, test = time_series_split(df, train_size)
    
    st.write(f"Porsi data training: {train_size*100:.2f}%")
    st.write(f"Porsi data testing: {100 - train_size*100:.2f}%")
    
    # Tampilkan plot pembagian
    splitting_plot(train, test)
    
    return train, test

    