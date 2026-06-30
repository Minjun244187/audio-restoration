import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os

# 페이지 기본 설정
st.set_page_config(page_title="SP Record Restoration", layout="wide")
st.title("🎛️ SP Record DSP Restoration Dashboard")
st.markdown("78rpm SP판의 디지털 복원 및 노이즈 필터링(Notch + LPF) 결과를 실시간으로 분석합니다.")

# ==========================================
# DSP 및 분석 함수
# ==========================================
def apply_notch_filter(data, fs, freq=60.0, Q=30.0):
    b, a = signal.iirnotch(freq, Q, fs)
    return signal.filtfilt(b, a, data)

def apply_lowpass_filter(data, fs, cutoff=6000.0, order=5):
    nyq = 0.5 * fs 
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)

def get_snr(signal_section, noise_section):
    p_sig = np.mean(np.square(signal_section.astype(np.float64))) if len(signal_section) > 0 else 0
    p_noise = np.mean(np.square(noise_section.astype(np.float64))) if len(noise_section) > 0 else 0
    if p_noise == 0: return float('inf')
    return 10 * np.log10(p_sig / p_noise)

def plot_spectrogram(data, fs, title, nfft=2048):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.specgram(data, Fs=fs, NFFT=nfft, noverlap=nfft//2, cmap='inferno')
    ax.set_title(title)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(0, 10000)
    plt.tight_layout()
    return fig

# ==========================================
# 메인 파이프라인 (웹 UI)
# ==========================================
filename = 'sp_record_sample.wav'
# 기존의 wavfile.read(filename) 부분을 이렇게 바꿔봐
uploaded_file = st.file_uploader("Upload your SP Record WAV file", type=["wav"])

if uploaded_file is not None:
    # 업로드된 파일을 처리   
    fs, audio_data = wavfile.read(filename)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    total_duration = len(audio_data) / fs
    
    st.audio(filename, format="audio/wav")

    # 2. 사이드바 - 파라미터 컨트롤
    st.sidebar.header("⚙️ Filter & Analysis Settings")
    
    st.sidebar.subheader("1. Audio Profiling")
    noise_range = st.sidebar.slider("Noise Profile (Silence) [sec]", 0.0, total_duration, (0.0, 1.0), step=0.1)
    signal_range = st.sidebar.slider("Signal Profile (Music) [sec]", 0.0, total_duration, (2.0, 5.0), step=0.1)
    
    st.sidebar.subheader("2. Filter Tuning")
    notch_freq = st.sidebar.number_input("Notch Frequency (Hz)", value=60.0, step=1.0)
    lpf_cutoff = st.sidebar.number_input("LPF Cutoff (Hz)", value=6000.0, step=100.0)

    # 구간 데이터 추출
    noise_profile = audio_data[int(noise_range[0]*fs) : int(noise_range[1]*fs)]
    signal_profile = audio_data[int(signal_range[0]*fs) : int(signal_range[1]*fs)]
    original_snr = get_snr(signal_profile, noise_profile)

    # 3. 메인 화면 - 결과 시각화
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Audio (Before)")
        st.metric(label="Initial SNR", value=f"{original_snr:.2f} dB")
        st.pyplot(plot_spectrogram(audio_data, fs, "Original Spectrogram"))

    with col2:
        st.subheader("Filtered Audio (After)")
        
        # 필터 적용
        notched = apply_notch_filter(audio_data, fs, freq=notch_freq)
        filtered = apply_lowpass_filter(notched, fs, cutoff=lpf_cutoff)
        
        # 새로운 SNR 계산
        filtered_noise = filtered[int(noise_range[0]*fs) : int(noise_range[1]*fs)]
        filtered_signal = filtered[int(signal_range[0]*fs) : int(signal_range[1]*fs)]
        final_snr = get_snr(filtered_signal, filtered_noise)
        
        st.metric(label="Final SNR", value=f"{final_snr:.2f} dB", delta=f"{final_snr - original_snr:+.2f} dB")
        st.pyplot(plot_spectrogram(filtered, fs, "Filtered Spectrogram"))