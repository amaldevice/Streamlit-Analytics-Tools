import pandas as pd
import numpy as np  
import streamlit as st
import io

@st.cache_data
def read_file(uploaded_file, separator=',', sheet_name=None):
    """Membaca file CSV, TSV, atau Excel dan mengembalikan DataFrame."""
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=separator)
            return df
        elif filename.endswith('.tsv'):
            df = pd.read_csv(uploaded_file, sep='\t')
            return df
        elif filename.endswith('.xlsx'):
            if sheet_name is None:
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_name = excel_file.sheet_names[0]  # Default ke sheet pertama
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            return df
        else:
            st.error("Format file tidak didukung!")
            return None
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        return None

def handle_upload():
    """
    Menangani proses upload file, termasuk pemilihan separator atau sheet.
    Mengembalikan DataFrame jika berhasil, atau None jika gagal.
    """
    uploaded_file = st.file_uploader("Upload file CSV, TSV, atau Excel", type=['csv', 'xlsx', 'tsv'])
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        file_extension = filename.split('.')[-1]

        if file_extension in ['csv', 'tsv']:
            # Tentukan separator berdasarkan jenis file
            if file_extension == 'csv':
                separator = st.text_input(
                    "Masukkan separator yang digunakan dalam file CSV (misalnya, ',' atau ';')",
                    value=','
                )
            else:
                separator = '\t'

            df = read_file(uploaded_file, separator=separator)

            if df is not None:
                st.success("File berhasil dibaca!")
                return df
        
        elif file_extension == 'xlsx':
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names

            if not sheet_names:
                st.error("Tidak ada sheet yang ditemukan dalam file Excel.")
                return None

            sheet_name = st.selectbox("Pilih nama sheet yang akan digunakan:", sheet_names, index=None)

            if sheet_name:
                df = read_file(uploaded_file, sheet_name=sheet_name)

                if df is not None:
                    st.success("File Excel berhasil dibaca!")
                    return df
            else:
                st.warning("Silakan pilih nama sheet yang akan digunakan.")
                return None
        
        else:
            st.error("Format file tidak didukung!")
            return None
    
    else:
        st.info("Silakan upload file CSV, TSV, atau Excel.")
        return None

@st.cache_data
def display_data(df):
    """Display the DataFrame."""
    st.subheader('Menampilkan Dataframe:')
    st.write("Ukuran dan Bentuk Data : ")
    st.write(f'**Di data tersebut terdapat Total {df.shape[0]} Baris dan {df.shape[1]} Kolom**')
    st.write("Dataframe:")
    st.write(df.head())
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.write("Informasi Dataframe:")
    st.text(info_str)

@st.cache_data
def clean_data(df):
    """
    Clean the DataFrame by:
    1. Dropping columns with more than 30% missing values.
    2. Dropping rows with missing values for columns with less than 30% missing values.
    """
    df = df.copy()
    
    # Step 1: Drop columns with more than 30% missing values
    threshold = 0.3 * len(df)
    df = df.dropna(thresh=threshold, axis=1)
    
    # Step 2: Drop rows with missing values in the remaining columns
    df_cleaned = df.dropna()
    
    # Drop duplicated rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    return df_cleaned

def handle_data_cleaning(df):
    """
    Menangani proses pembersihan data berdasarkan pilihan pengguna.
    
    Parameters:
    df (pd.DataFrame): DataFrame asli.
    
    Returns:
    pd.DataFrame: DataFrame yang telah dibersihkan atau asli berdasarkan pilihan pengguna.
    """
    st.header("Pembersihan Data")
    
    cleaning_option = st.radio(
        "Apakah Anda ingin membersihkan data?",
        ('Ya', 'Tidak'),
        index=None # Mengatur default ke 'Tidak'
    )
    
    if cleaning_option == 'Ya':
        df = clean_data(df)
        st.success("Data berhasil dibersihkan!")
        st.subheader("Data setelah dibersihkan:")
        display_data(df)
    elif cleaning_option == 'Tidak':
        st.info("Data tidak dibersihkan.")
    else:
        st.warning("Pilih opsi untuk membersihkan data.")
        st.stop()
    
    return df

@st.cache_data
def format_date_columns(df, date_columns):
    """Format specified columns to datetime."""
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def handle_date_formatting(df):
    """
    Menangani proses pemformatan kolom tanggal berdasarkan pilihan pengguna.
    
    Parameters:
    df (pd.DataFrame): DataFrame yang akan diformat.
    
    Returns:
    pd.DataFrame: DataFrame dengan kolom tanggal yang diformat atau tanpa perubahan.
    """
    st.header("Pemformatan Kolom Tanggal")
    
    format_date_option = st.radio(
        "Apakah Anda ingin memformat Kolom/Variabel tanggal?",
        ('Ya', 'Tidak'),
        index=None  # Mengatur default ke 'Tidak'
    )
    
    if format_date_option == 'Ya':
        date_columns = st.multiselect(
            "Pilih Kolom/Variabel yang berisi tanggal:",
            options=df.columns,
            key='date_columns'
        )
        if date_columns:
            df = format_date_columns(df, date_columns)
            st.success("Kolom/Variabel tanggal berhasil diformat!")
            st.subheader("Data setelah pemformatan tanggal:")
            display_data(df)
        else:
            st.warning("Pilih Kolom/Variabel yang berisi tanggal.")
            st.stop()
    elif format_date_option == 'Tidak':
        st.info("Kolom/Variabel tanggal tidak diformat.")
    else:
        st.warning("Pilih opsi untuk memformat Kolom/Variabel tanggal.")
        st.stop()
    
    return df

def clean_timeseries_data(df):
    df = df.copy()
    df = df.interpolate(method='linear')
    return df

@st.cache_data
def time_series_formatting(df, date_column, target_column):
    df = df.copy()
    df = df.set_index(date_column)
    df = df[target_column]
    df = df.sort_index()
    return df

def handle_time_series_formatting(df, resample=None):

    ts_format = st.radio('Apakah Anda ingin memformat data time series?', ('Ya', 'Tidak'), index=None)
    if ts_format == 'Ya':    
        st.header("Format Data Time Series")
        
        date_column = st.selectbox(
            "Pilih Kolom Tanggal:",
            options=df.columns,
            index=None
        )
        if date_column is None:
            st.warning("Pilih kolom tanggal yang berisi data time series.")
            st.stop()
        
        target_column = st.selectbox(
            "Pilih Kolom Target:",
            options=df.columns,
            index=None
        )
        
        if target_column is None:
            st.warning("Pilih kolom target yang berisi data time series.")
            st.stop()
        
        resample = st.text_input("Resample data time series (opsional):", value='', key='resample')

        if resample:
            df = time_series_formatting(df, date_column, target_column)
            df = df.resample(resample).mean()
            df = clean_timeseries_data(df)
            st.success(f"Data Time Series berhasil di-resample dengan interval {resample}!")
            st.write(df.head())
            return df
        
        else:
            df = time_series_formatting(df, date_column, target_column)
            st.success("Data Time Series berhasil diformat!")
            st.subheader("Data Time Series:")
            st.write(df.head())
        
            return df
    
    elif ts_format == 'Tidak':
         st.info("Data Time Series tidak diformat.")
         return None
    else:
         st.warning("Pilih opsi untuk memformat data time series.")
         st.stop()