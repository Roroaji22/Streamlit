import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Fungsi untuk memuat data dari file langsung
@st.cache_data
def load_data():
    data = pd.read_csv('bantuanSosial.csv', delimiter=';')
    return data


# Fungsi untuk melatih model
@st.cache_data
def train_model(data):
    # Encode fitur dan target
    le_jenis_bantuan = LabelEncoder()
    le_kluster = LabelEncoder()
    le_ragam = LabelEncoder()
    le_dampak = LabelEncoder()

    data['Jenis Bantuan Encoded'] = le_jenis_bantuan.fit_transform(data['Jenis Bantuan'])
    data['Kluster Encoded'] = le_kluster.fit_transform(data['kluster'])
    data['Ragam Encoded'] = le_ragam.fit_transform(data['ragam'])
    data['Dampak Encoded'] = le_dampak.fit_transform(data['Dampak'])

    X = data[['Jenis Bantuan Encoded', 'Kluster Encoded', 'Ragam Encoded']]
    y = data['Dampak Encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, le_jenis_bantuan, le_kluster, le_ragam, le_dampak, data


# Sidebar untuk memilih menu
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Pilih Menu", ["Analisis Data", "Prediksi"])

# Variabel untuk pengaturan pagination
if "page" not in st.session_state:
    st.session_state.page = 0

ROWS_PER_PAGE = 10  # Banyak data per halaman

# Menu Analisis Data
if menu == "Analisis Data":
    st.title("Analisis Data Bantuan Sosial - Nusa Tenggara Timur (2022-2024)")

    # Memuat data
    data = load_data()

    # Menampilkan Peta Sebaran Bantuan Sosial
    st.write("Peta Sebaran Bantuan Sosial:")
    st.map(data[["latitude", "longitude"]])

    # Distribusi Jumlah Bantuan per Tahun (dengan Line Chart)
    st.write("### Distribusi Jumlah Bantuan per Tahun")
    st.markdown("""
    Grafik berikut menunjukkan distribusi jumlah bantuan sosial yang diberikan per tahun. Kamu dapat melihat bagaimana jumlah bantuan berubah seiring waktu.
    """)

    # Menghitung jumlah bantuan per tahun
    bantuan_per_tahun = data.groupby("Tahun Pendataan").size().reset_index(name='Jumlah Bantuan')

    # Urutkan data berdasarkan tahun
    bantuan_per_tahun = bantuan_per_tahun.sort_values("Tahun Pendataan")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bantuan_per_tahun["Tahun Pendataan"], bantuan_per_tahun["Jumlah Bantuan"], marker='o', linestyle='-',
            color='skyblue')
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Jumlah Bantuan")
    ax.set_title("Distribusi Jumlah Bantuan per Tahun")
    ax.set_xticks(bantuan_per_tahun["Tahun Pendataan"])
    ax.set_xticklabels(bantuan_per_tahun["Tahun Pendataan"].astype(int))

    # Menambahkan label pada setiap titik
    for i, row in bantuan_per_tahun.iterrows():
        ax.text(row["Tahun Pendataan"], row["Jumlah Bantuan"], row["Jumlah Bantuan"], fontsize=9, ha='right',
                va='bottom')

    st.pyplot(fig)

    # Distribusi Bantuan per Kluster (dengan Pie Chart)
    st.write("### Distribusi Bantuan per Kluster")
    st.markdown("""
    Grafik lingkaran berikut menunjukkan distribusi bantuan sosial berdasarkan kluster. Ini memberikan gambaran tentang proporsi bantuan di setiap kluster.
    """)

    # Menghitung distribusi bantuan per kluster
    bantuan_per_kluster = data["kluster"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(bantuan_per_kluster, labels=bantuan_per_kluster.index, autopct='%1.1f%%',
           colors=plt.cm.Paired(range(len(bantuan_per_kluster))))
    ax.set_title("Distribusi Bantuan per Kluster")
    st.pyplot(fig)

    # Distribusi Bantuan per Ragam (dengan Bar Chart)
    st.write("### Distribusi Bantuan per Ragam")
    st.markdown("""
    Grafik batang berikut menunjukkan distribusi bantuan sosial berdasarkan ragam. Filter kluster tersedia di bawah grafik untuk menyaring data sesuai dengan kluster yang dipilih.
    """)

    # Mendapatkan daftar kluster yang ada
    kluster_list = data["kluster"].unique()
    selected_kluster = kluster_list[0]  # Pilih kluster pertama secara default

    # Filter Kluster untuk Ragam
    kluster_filter_ragam = st.selectbox("Pilih Kluster untuk Distribusi Ragam", kluster_list, index=3)

    # Menampilkan data berdasarkan kluster terpilih
    filtered_data_ragam = data[data["kluster"] == kluster_filter_ragam]

    # Menghitung distribusi bantuan per ragam
    bantuan_per_ragam = filtered_data_ragam["ragam"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(bantuan_per_ragam.index, bantuan_per_ragam, color='skyblue')
    ax.set_xlabel("Ragam")
    ax.set_ylabel("Jumlah Bantuan")
    ax.set_title("Distribusi Bantuan per Ragam")
    ax.set_xticklabels(bantuan_per_ragam.index, rotation=45)

    # Menambahkan label pada setiap batang
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, int(height), ha='center', va='bottom', fontsize=9)

    st.pyplot(fig)

# Menu Prediksi
elif menu == "Prediksi":
    st.title("Prediksi Bantuan Sosial")

    # Memuat data dan melatih model
    data = load_data()
    model, le_jenis_bantuan, le_kluster, le_ragam, le_dampak, all_data = train_model(data)

    # Mendapatkan daftar kluster dan ragam
    kluster_list = le_kluster.classes_

    # Input fitur untuk prediksi
    st.write("### Masukkan Data untuk Prediksi")
    jenis_bantuan = st.selectbox("Jenis Bantuan", le_jenis_bantuan.classes_)
    kluster = st.selectbox("Kluster", kluster_list)

    # Mendapatkan daftar ragam untuk kluster yang dipilih
    ragam_list = all_data[all_data["kluster"] == kluster]["ragam"].unique()
    ragam = st.selectbox("Ragam", ragam_list)

    # Menyiapkan data untuk prediksi
    input_data = pd.DataFrame({
        'Jenis Bantuan Encoded': [le_jenis_bantuan.transform([jenis_bantuan])[0]],
        'Kluster Encoded': [le_kluster.transform([kluster])[0]],
        'Ragam Encoded': [le_ragam.transform([ragam])[0]]
    })

    # Melakukan prediksi dan probabilitas
    if st.button("Prediksi"):
        prediction_proba = model.predict_proba(input_data)[0]

        # Menampilkan dua probabilitas tertinggi
        top_indices = prediction_proba.argsort()[-1:][::-1]
        top_classes = le_dampak.classes_[top_indices]
        top_probas = prediction_proba[top_indices]

        st.write("**Probabilitas Prediksi:**")
        for cls, proba in zip(top_classes, top_probas):
            st.write(f"{cls}: {proba:.2%}")
