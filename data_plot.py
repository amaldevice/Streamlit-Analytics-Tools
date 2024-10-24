import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import PIL.Image
import streamlit as st
import base64
from generate_plot import generate_distribution_plot, generate_countplot, generate_countplot_categoric, generate_aggregasi_plot, generate_ai_interpret

@st.cache_data
def plot_correlation(df):
    """Plot and display correlation heatmap."""
    st.subheader('Korelasi antar Kolom/Variabel:')
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    st.pyplot(plt)
    plt.clf()
    return img_buffer

def plot_distribution(df, numeric_columns):
    """Handle user interaction for plotting distribution and return image buffer if plotted."""
    # Pilihan pengguna untuk melihat distribusi data
    dist_plot = st.radio("Apakah Anda ingin melihat distribusi data pada kolom numerik?", ('Ya', 'Tidak'), index=None)

    if dist_plot == 'Ya':
        # Pilih kolom numerik untuk divisualisasikan
        dist_column = st.selectbox(
            "Pilih kolom numerik untuk divisualisasikan distribusinya:",
            numeric_columns,
            index=None,
            placeholder='Pilih Kolom untuk Distribusi'
        )
        
        if dist_column:
            st.subheader(f'Distribusi Data pada Kolom {dist_column}:')
            # Panggil fungsi generate_distribution_plot untuk mendapatkan buffer gambar
            dist_img = generate_distribution_plot(df, dist_column)
            # Tampilkan gambar menggunakan st.image
            st.image(dist_img, caption=f'Distribusi {dist_column}')
            return dist_img
        else:
            st.warning("Silakan pilih kolom numerik untuk divisualisasikan distribusinya.")
            st.stop()
    elif dist_plot == 'Tidak':
        st.write("Distribusi data tidak divisualisasikan.")
        return None
    else:
        st.warning("Pilih opsi untuk melihat distribusi data.")
        st.stop()

def plot_countplot(df, categorical_columns):
    """Handle user interaction for single countplot and return image buffer if plotted."""
    # Pilihan pengguna untuk melihat countplot satu kolom kategorik
    count_plot = st.radio("Apakah Anda ingin melihat countplot untuk kolom kategorik?", ('Ya', 'Tidak'), index=None)

    if count_plot == 'Ya':
        # Pilih kolom kategorik untuk divisualisasikan
        count_column = st.selectbox(
            "Pilih kolom kategorik untuk divisualisasikan countplotnya:",
            categorical_columns,
            index=None,
            placeholder='Pilih Kolom untuk Countplot'
        )

        if count_column:
            st.subheader(f'Countplot untuk Kolom {count_column}:')
            # Panggil fungsi generate_countplot untuk mendapatkan buffer gambar
            count_img = generate_countplot(df, count_column)
            # Tampilkan gambar menggunakan st.image
            st.image(count_img, caption=f'Countplot {count_column}')
            return count_img
        else:
            st.warning("Silakan pilih kolom kategorik untuk divisualisasikan countplotnya.")
            st.stop()
    elif count_plot == 'Tidak':
        st.write("Countplot tidak divisualisasikan.")
        return None
    else:
        st.warning("Pilih opsi untuk melihat countplot.")
        st.stop()

def plot_countplot_categoric(df, categorical_columns):
    """Handle user interaction for countplot dua kolom kategorik dan return image buffer jika divisualisasikan."""
    # Pilihan pengguna untuk melihat countplot dua kolom kategorik
    cat_countplot = st.radio("Apakah Anda ingin melihat countplot untuk dua kolom kategorik?", ('Ya', 'Tidak'), index=None)

    if cat_countplot == 'Ya':
        # Pilih dua kolom kategorik untuk divisualisasikan
        cat_cols = st.multiselect(
            "Pilih dua kolom kategorik untuk divisualisasikan countplotnya:",
            categorical_columns,
            default=None
        )

        if len(cat_cols) == 2:
            st.subheader(f'Countplot untuk Kolom {cat_cols[0]} dan {cat_cols[1]}:')
            # Panggil fungsi generate_countplot_categoric untuk mendapatkan buffer gambar
            count_img = generate_countplot_categoric(df, tuple(cat_cols))
            # Tampilkan gambar menggunakan st.image
            st.image(count_img, caption=f'Countplot {cat_cols[0]} vs {cat_cols[1]}')
            return count_img
        else:
            st.warning("Silakan pilih dua kolom kategorik untuk divisualisasikan countplotnya.")
            st.stop()
    elif cat_countplot == 'Tidak':
        st.write("Countplot untuk dua kolom kategorik tidak divisualisasikan.")
        return None
    else:
        st.warning("Pilih opsi untuk melihat countplot untuk dua kolom kategorik.")
        st.stop()

def plot_aggregasi_data(df, all_columns):
    """
    Handle user interaction for data aggregation plotting and return image buffer if plotted.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        all_columns (list): List of all column names in the DataFrame.

    Returns:
        BytesIO or None: Image buffer containing the plot if plotted, else None.
    """
    agg = st.radio("Apakah Anda ingin melihat plot aggregasi data?", ('Ya', 'Tidak'), index=None)

    if agg == 'Ya':
        st.header("Plot Aggregasi Data")

        # Pilih dua kolom untuk agregasi
        col = st.multiselect(
            "Pilih dua kolom untuk divisualisasikan aggregasi datanya, Kolom pertama sebagai Sumbu X, kolom kedua sebagai Sumbu Y",
            options=all_columns,
            default=None,
            max_selections=2
        )

        # Pilih jenis agregasi
        jenis_aggregasi = st.selectbox(
            "Pilih jenis agregasi yang diinginkan:",
            options=['sum', 'mean', 'count', 'median', 'max', 'min'],
            index=None,
            key='aggregasi_selectbox'
        )

        if jenis_aggregasi:
            st.success(f"Jenis agregasi yang dipilih: {jenis_aggregasi}")
        else:
            st.warning("Silakan pilih jenis agregasi yang diinginkan.")
            st.stop()

        if len(col) == 2:
            group_col, agg_col = col

            # Validasi bahwa kolom kedua adalah numerik
            if not pd.api.types.is_numeric_dtype(df[agg_col]):
                st.error('Kolom kedua harus berupa data numerik.')
                st.stop()

            try:
                # Panggil fungsi generate_aggregasi_plot dengan tuple kolom
                agg_img = generate_aggregasi_plot(df, tuple(col), jenis_aggregasi)
                # Tampilkan gambar menggunakan st.image
                st.image(agg_img, caption=f'Aggregasi {jenis_aggregasi.capitalize()} of {agg_col} by {group_col}')
                return agg_img
            except ValueError as e:
                st.error(str(e))
                st.stop()
        else:
            st.warning("Silakan pilih dua kolom untuk divisualisasikan aggregasi datanya.")
            st.stop()
    elif agg == 'Tidak':
        st.write("Aggregasi data tidak divisualisasikan.")
        return None

    else:
        st.warning("Pilih opsi untuk melihat aggregasi data.")
        st.stop()

def plot_ai_interpretation(img_buffer, identifier):
    """
    Handle user interaction for AI interpretation and display the result.

    Parameters:
        img_buffer (BytesIO): Image buffer containing the plot to be interpreted.

    Returns:
        interpretation (str): AI interpretation hasil dari proses AI.
    """
    st.header("Interpretasi AI")

    # Form untuk input pengguna
    ai_choice = st.selectbox(
        "Pilih AI yang akan digunakan:",
        ['Google Gemini', 'OpenAI ChatGPT'],
        key=identifier + '_ai_choice',
        index=None
    )

    if ai_choice:
        st.success(f"AI yang dipilih: {ai_choice}")
    else:
        st.warning("Silakan pilih AI yang akan digunakan.")
        st.stop()

    system_prompt = st.text_area(
        "Masukkan peran sistem (System Role):",
        value="",
        help = "Contoh : Anda adalah seorang senior data analyst dengan keahlian dalam interpretasi visualisasi data dan memberikan insight serta rekomendasi "
               "yang relevan. Berdasarkan visualisasi yang diberikan, Anda akan mengidentifikasi tren, pola, atau anomali utama. Kemudian, Anda akan memberikan insight mendalam terkait hasil tersebut serta menyarankan langkah-langkah strategis atau "
               "rekomendasi yang sesuai untuk meningkatkan kinerja atau mengatasi masalah yang terdeteksi dalam data. Sertakan konteks bisnis atau industri yang relevan dalam analisis Anda untuk memberikan rekomendasi yang dapat diimplementasikan secara praktis.",
        height=150,
        key = identifier + '_system_prompt'
    )

    if system_prompt:
        st.success("Peran sistem telah dimasukkan.")
    else:
        st.warning("Silakan masukkan peran sistem.")
        st.stop()

    user_prompt = st.text_area(
        "Masukkan prompt pengguna (User Prompt):",
        value="",
        help = "Berdasarkan hasil analisis plot di atas, insight apa yang dapat diperoleh dari pola yang terlihat dan hubungan antar variabel? "
               "Berikan rekomendasi tindakan yang dapat diambil serta langkah-langkah selanjutnya berdasarkan temuan tersebut. Buatkan penjelasannya dalam bahasa indonesia.",
        height=150,
        key = identifier + '_user_prompt'
    )

    if user_prompt:
        st.success("Prompt pengguna telah dimasukkan.")
    else:
        st.warning("Silakan masukkan prompt pengguna.")
        st.stop()

    # Tombol untuk memulai interpretasi AI
    interpretation = generate_ai_interpret(ai_choice, system_prompt, user_prompt, img_buffer)
    st.write(interpretation)
