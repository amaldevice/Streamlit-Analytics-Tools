import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import io
import streamlit as st
import PIL.Image
import io
from io import BytesIO
import base64
import openai
from openai import OpenAI
import google.generativeai as genai

# Konfigurasi OpenAI
client = OpenAI(api_key='sk-Q9EnJHuYEL1ywyo5s59RW7UM0l9RjvrPXZe4FEQmHYT3BlbkFJRzVpHcwRSVoLIAt4p6zg_BI2LLBT8G9TzZK1bXsHMA')
# Konfigurasi Google Generative AI
genai.configure(api_key="AIzaSyDQfpY0Oo-Qv_V1szhqXKSfAedCKA1czTc")
model = genai.GenerativeModel("gemini-1.5-flash-latest")

@st.cache_data
def generate_distribution_plot(df, column):
    """Generate a histogram plot for the specified numerical column and return as an image buffer."""
    plt.figure(figsize=(12, 10))
    sns.histplot(df[column], kde=True)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()  # Menutup figure untuk menghemat memori
    img_buffer.seek(0)
    return img_buffer

@st.cache_data
def generate_plot_correlation(df):
    """Plot and display correlation heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    return img_buffer


@st.cache_data
def generate_countplot(df, column):
    """Generate a countplot for the specified categorical column and return as an image buffer."""
    plt.figure(figsize=(12, 10))
    ax = sns.countplot(x=df[column])
    # Menambahkan nilai di atas setiap bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), 
                    textcoords='offset points')
    plt.xticks(rotation=45)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    return img_buffer

@st.cache_data
def generate_countplot_categoric(df, columns):
    """Generate a countplot for two categorical columns and return as an image buffer."""
    plt.figure(figsize=(12, 10))
    ax = sns.countplot(x=df[columns[0]], hue=df[columns[1]])
    plt.xticks(rotation=45)
    # Menambahkan nilai di atas setiap bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), 
                    textcoords='offset points')
    plt.legend(title=columns[1])
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    return img_buffer

@st.cache_data
def generate_aggregasi_plot(df, columns, jenis_aggregasi):
    """
    Generate an aggregation barplot for two selected columns and return as an image buffer.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        columns (tuple): Tuple containing two column names (group_col, agg_col).
        jenis_aggregasi (str): Type of aggregation ('sum', 'mean', 'count', 'median', 'max', 'min').

    Returns:
        BytesIO: Image buffer containing the plot.
    """
    group_col, agg_col = columns

    # Pastikan kolom kedua adalah numerik
    if not pd.api.types.is_numeric_dtype(df[agg_col]):
        raise ValueError('Kolom kedua harus berupa data numerik.')

    # Grouping data dan menghitung aggregasi berdasarkan pilihan pengguna
    if jenis_aggregasi == 'sum':
        grouped_data = df.groupby(group_col)[agg_col].sum().reset_index().sort_values(by=agg_col, ascending=False)
    elif jenis_aggregasi == 'mean':
        grouped_data = df.groupby(group_col)[agg_col].mean().reset_index().sort_values(by=agg_col, ascending=False)
    elif jenis_aggregasi == 'count':
        grouped_data = df.groupby(group_col)[agg_col].count().reset_index().sort_values(by=agg_col, ascending=False)
    elif jenis_aggregasi == 'min':
        grouped_data = df.groupby(group_col)[agg_col].min().reset_index()
    elif jenis_aggregasi == 'max':
        grouped_data = df.groupby(group_col)[agg_col].max().reset_index()
    elif jenis_aggregasi == 'median':
        grouped_data = df.groupby(group_col)[agg_col].median().reset_index()
    else:
        raise ValueError('Jenis aggregasi tidak valid.')

    # Plot data hasil agregasi
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(data=grouped_data, x=group_col, y=agg_col)
    plt.title(f'Aggregasi {jenis_aggregasi.capitalize()} of {agg_col} by {group_col}', fontsize=16)
    plt.xlabel(group_col, fontsize=14)
    plt.ylabel(jenis_aggregasi.capitalize(), fontsize=14)

    # Menambahkan nilai di atas setiap bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), 
                    textcoords='offset points')

    # Atur layout dan simpan plot ke buffer
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    return img_buffer

@st.cache_data
def generate_ai_interpret(ai_choice, system_prompt, user_prompt, img_buffer):
    """
    Fungsi untuk menghasilkan interpretasi menggunakan AI pilihan (OpenAI ChatGPT atau Google Gemini)
    berdasarkan prompt sistem, prompt pengguna, dan visualisasi gambar yang telah diproses menjadi img_buffer.

    Parameters:
        ai_choice (str): Pilihan AI yang akan digunakan ('Google Gemini' atau 'OpenAI ChatGPT').
        system_prompt (str): Prompt sistem yang berisi instruksi untuk AI.
        user_prompt (str): Prompt pengguna yang mengajukan pertanyaan atau permintaan ke AI.
        img_buffer (BytesIO): Gambar visualisasi yang telah diproses dalam bentuk buffer.

    Returns:
        interpretation (str): Hasil interpretasi dari AI berdasarkan input yang diberikan.
    """
    try:
        # Buka gambar dari img_buffer
        img = PIL.Image.open(img_buffer)

        if ai_choice == 'OpenAI ChatGPT':
            # Encode gambar ke base64 untuk dikirim ke OpenAI
            img_buffer.seek(0)  # Pastikan posisi buffer di awal
            img_str = base64.b64encode(img_buffer.getvalue()).decode()

            # Panggilan ke OpenAI API (sesuaikan dengan konfigurasi OpenAI yang benar)
            response = client.chat.completions.create(
                model="gpt-4o",  # Pastikan menggunakan model OpenAI yang benar
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {   
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt 
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            interpretation = response.choices[0].message.content

        elif ai_choice == 'Google Gemini':
            # Implementasi untuk Google Generative AI (Google Gemini)
            # Gambar dikirim langsung dalam format PIL image atau img_buffer, tergantung implementasi API Gemini
            # Asumsikan model.generate_content menerima gambar dari img_buffer atau img PIL

            # Contoh asumsi jika Google Gemini menerima img_buffer:
            response = model.generate_content([
                    user_prompt,
                    img  
                
            ])
            response.resolve()  # Jika diperlukan, tunggu proses asinkron selesai
            interpretation = response.text  # Asumsikan ini adalah teks respons dari Gemini

        return st.write(interpretation)

    except Exception as e:
        st.error(f"Error generating content: {e}")
        return None
