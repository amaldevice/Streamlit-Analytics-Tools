import pandas as pd
import numpy as np  
import streamlit as st
import io
import requests
from io import BytesIO

@st.cache_data
def read_file(uploaded_file=None, separator=',', sheet_name=None, api_url=None):
    """
    Reads a CSV, TSV, or Excel file or fetches data from an API, returning a DataFrame.
    """
    try:
        if api_url:
            # Fetch data from the API
            response = requests.get(api_url)
            response.raise_for_status()
            # Assume CSV format with specified separator
            df = pd.read_csv(BytesIO(response.content), sep=separator)
            return df
        elif uploaded_file:
            # Handle uploaded file
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
                    sheet_name = excel_file.sheet_names[0]  # Default to first sheet
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                return df
            else:
                st.error("Unsupported file format!")
                return None
        else:
            st.warning("No file or API URL provided.")
            return None
    except Exception as e:
        st.error(f"Error reading data: {e}")
        return None

def handle_upload():
    """
    Manages file upload and API data fetching, including separator or sheet selection.
    Returns a DataFrame if successful, or None if it fails.
    """
    data_source = st.selectbox("Select data source:", ["Upload File", "Fetch from API"], index=0)
    
    if data_source == "Upload File":
        uploaded_file = st.file_uploader("Upload a CSV, TSV, or Excel file", type=['csv', 'xlsx', 'tsv'])
        if uploaded_file:
            filename = uploaded_file.name.lower()
            file_extension = filename.split('.')[-1]

            if file_extension in ['csv', 'tsv']:
                separator = ',' if file_extension == 'csv' else '\t'
                separator = st.text_input("Enter separator used in the file:", value=separator)
                df = read_file(uploaded_file=uploaded_file, separator=separator)
                if df is not None:
                    st.success("File successfully read!")
                    return df
            elif file_extension == 'xlsx':
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                if not sheet_names:
                    st.error("No sheets found in the Excel file.")
                    return None
                sheet_name = st.selectbox("Select the sheet to use:", sheet_names)
                if sheet_name:
                    df = read_file(uploaded_file=uploaded_file, sheet_name=sheet_name)
                    if df is not None:
                        st.success("Excel file successfully read!")
                        return df
            else:
                st.error("Unsupported file format!")
                return None
        else:
            st.info("Please upload a CSV, TSV, or Excel file.")
            return None
    
    elif data_source == "Fetch from API":
        api_url = st.text_input("Enter the API URL:")
        if api_url:
            st.info(f"Fetching data from API: {api_url}")
            df = read_file(api_url=api_url)
            if df is not None:
                st.success("Data successfully fetched from API!")
                return df
            else:
                st.error("Failed to fetch data from API.")
                return None
        else:
            st.warning("Please enter a valid API URL.")
            return None
    else:
        st.warning("Please select a data source.")
        st.stop()

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
def format_date_columns(df, date_columns, freq):
    """Format specified columns to datetime."""
    freq_map = {
        'Harian': 'D',
        'Bulanan': 'M',
        'Triwulan': 'Q',
        'Tahunan': 'Y'
    }
    freq_pandas = freq_map.get(freq)

    if not freq_pandas:
        st.warning("Frekuensi tidak dikenali. Tidak ada perubahan yang dilakukan pada data.")
        return df

    for col in date_columns:
        try:
            if freq_pandas == 'D':
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
            elif freq_pandas == 'M':
                df[col] = pd.to_datetime(df[col], format='%Y-%m', errors='coerce')
            elif freq_pandas == 'Q':
                # Pandas tidak mendukung format khusus untuk triwulan, jadi kita parse tanpa format
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # Opsional: Konversi ke PeriodIndex jika diperlukan
                # df[col] = df[col].dt.to_period('Q')
            elif freq_pandas == 'Y':
                df[col] = pd.to_datetime(df[col], format='%Y', errors='coerce')
        except Exception as e:
            st.error(f"Gagal memformat kolom {col}: {e}")

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
        index=None,
        key='format_date_option'
    )
    
    if format_date_option == 'Ya':
        date_columns = st.multiselect(
            "Pilih Kolom/Variabel yang berisi tanggal:",
            options=df.columns,
            key='date_columns',
            max_selections=1
        )

        if date_columns:

            full_format = st.radio('Apakah format tanggal lengkap? (Memiliki Tahun Bulan Hari, Contoh : 2024-31-01)', 
                                ('Ya', 'Tidak'), 
                                index=None, 
                                key='full_format')
            
            if full_format == 'Tidak':
                freq = st.selectbox("Pilih Frekuensi Data:", options=['Harian', 'Bulanan', 'Triwulan', 'Tahunan'], key='freq_selectbox', index=None)
                if freq:
                    df = format_date_columns(df, date_columns, freq)
                    st.success("Kolom/Variabel tanggal berhasil diformat!")
                    st.subheader("Data setelah diformat:")
                    display_data(df)
                    return df, freq
                else:
                    st.warning("Pilih Frekuensi Data.")
                    st.stop()
            elif full_format == 'Ya':
                for col in date_columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception as e:
                        st.error(f"Gagal memformat kolom {col}: {e}")
                
                st.success("Kolom/Variabel tanggal berhasil diformat!")
                st.subheader("Data setelah diformat:")
                display_data(df)
                return df, None
            else:
                st.warning("Pilih Kolom/Variabel yang berisi tanggal.")
                st.stop()
    elif format_date_option == 'Tidak':
        st.info("Kolom/Variabel tanggal tidak diformat.")
        return df, None
    else:
        st.warning("Pilih opsi untuk memformat Kolom/Variabel tanggal.")
        st.stop()
    
    return df, None

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
            df = df.resample(resample).sum()
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