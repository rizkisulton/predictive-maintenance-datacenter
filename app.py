import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# 1. Definisi Arsitektur Model PyTorch (Harus sama persis dengan di Colab)
class ServerFailureModel(nn.Module):
    def __init__(self):
        super(ServerFailureModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.fc(x)

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Data Center Predictive Maintenance", layout="centered")

st.title("🖥️ Server Failure Prediction System")
st.write("Aplikasi Predictive Maintenance Infrastruktur Data Center berbasis PyTorch & Machine Learning.")
st.markdown("---")

# 2. Memuat File Model .pth
@st.cache_resource
def load_model():
    model = ServerFailureModel()
    model.load_state_dict(torch.load("predictive_maintenance_model.pth"))
    model.eval()
    return model

try:
    model = load_model()
    st.success("✅ Model PyTorch (`predictive_maintenance_model.pth`) Berhasil Dimuat!")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}. Pastikan file .pth ada di folder D:\\app\\")

# 3. Form Input User Interface (UI)
st.subheader("📊 Masukkan Metrik Real-time Server")

col1, col2 = st.columns(2)
with col1:
    cpu_usage = st.slider("Penggunaan CPU (%)", 0.0, 100.0, 15.0)
    ram_usage = st.slider("Penggunaan RAM (%)", 0.0, 100.0, 20.0)

with col2:
    disk_io = st.number_input("Disk I/O Speed (MB/s)", 0.0, 1000.0, 100.0)
    network_latency = st.number_input("Network Latency (ms)", 0.0, 500.0, 20.0)

# 4. Tombol Prediksi & Logika Inferensi
if st.button("🔍 Analisis Risiko Server"):
    # Normalisasi fitur ke skala 0..1 agar sesuai dengan skala training
    input_normalized = [
        cpu_usage / 100.0,
        ram_usage / 100.0,
        disk_io / 1000.0,
        network_latency / 500.0
    ]
    
    input_tensor = torch.tensor([input_normalized], dtype=torch.float32)
    
    with torch.no_grad():
        output_prob = model(input_tensor).item()
    
    st.markdown("---")
    st.subheader("📈 Hasil Analisis Ketersediaan Server")
    
    risk_percentage = output_prob * 100
    
    if output_prob >= 0.5:
        st.error(f"⚠️ **STATUS: HIGH FAILURE RISK ({risk_percentage:.1f}%)**")
        st.write("Rekomendasi: Lakukan migrasi VM / tindakan pencegahan otomatis proaktif.")
    else:
        st.success(f"✅ **STATUS: NORMAL / HEALTHY ({risk_percentage:.1f}%)**")
        st.write("Rekomendasi: Kondisi server stabil dan beroperasi optimal.")