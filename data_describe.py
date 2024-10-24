import pandas as pd
import streamlit as st
import io


@st.cache_data
def display_data(df):
    """Display the DataFrame."""
    st.write('Menampilkan Dataframe:')
    st.write("Ukuran dan Bentuk Data : ", df.shape)
    st.write("Dataframe:")
    st.write(df.head())
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.write("Informasi Dataframe:")
    st.text(info_str)
    
@st.cache_data
def display_statistics(df):
    """Display descriptive statistics of the DataFrame."""
    st.subheader("Statistik Deskriptif:")
    st.write(df.describe())