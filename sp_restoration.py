import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd

# ==========================================
# 1. Interactive UI Functions (With Playback)
# ==========================================
def select_audio_region(data, fs, title="Select Audio Region"):
    """
    Displays the audio waveform and allows the user to select a region.
    Includes Play and Stop buttons to preview the selection.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25)
    
    time_axis = np.arange(len(data)) / fs
    ax.plot(time_axis, data, color='#34495e', alpha=0.8, linewidth=0.5)
    
    ax.set_title(title + "\n[Drag to select region. Click 'Play' to listen. Close window to proceed.]", fontweight='bold')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle='--', alpha=0.5)

    selected_span = [0.0, 1.0]

    def onselect(xmin, xmax):
        selected_span[0] = xmin
        selected_span[1] = xmax
        print(f"  -> Selected region: {xmin:.2f}s to {xmax:.2f}s\r", end='')

    span = SpanSelector(
        ax, onselect, 'horizontal', useblit=True,
        props=dict(alpha=0.3, facecolor='red'),
        interactive=True
    )
    ax.span = span 

    ax_play = plt.axes([0.7, 0.05, 0.1, 0.075])
    btn_play = Button(ax_play, 'Play Selected')
    
    ax_stop = plt.axes([0.81, 0.05, 0.1, 0.075])
    btn_stop = Button(ax_stop, 'Stop')

    def play_audio(event):
        start_idx = int(selected_span[0] * fs)
        end_idx = int(selected_span[1] * fs)
        sd.stop()
        sd.play(data[start_idx:end_idx], fs)

    def stop_audio(event):
        sd.stop()

    btn_play.on_clicked(play_audio)
    btn_stop.on_clicked(stop_audio)
    
    ax.btn_play = btn_play
    ax.btn_stop = btn_stop

    plt.show()
    sd.stop()
    
    print()
    return selected_span[0], selected_span[1]

# ---------------------------------------------------------
# [NEW] NFFT 크기별 해상도 변화 시각화 함수
# ---------------------------------------------------------
def plot_nfft_comparison(data, fs, title="STFT Resolution Trade-off (NFFT Comparison)"):
    """
    Plots three spectrograms of the same audio using different NFFT window sizes
    to visually demonstrate the Time-Frequency uncertainty principle.
    """
    nfft_sizes = [256, 2048, 8192]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(bottom=0.12, hspace=0.4)
    
    # Render spectrogram for each NFFT size
    for ax, nfft in zip(axes, nfft_sizes):
        # noverlap is set to 50% of NFFT size for smoothness
        ax.specgram(data, Fs=fs, NFFT=nfft, noverlap=nfft//2, cmap='inferno')
        
        # Specific sub-captions explaining what to look for
        if nfft == 256:
            desc = " - High Time Resolution (Sharp vertical lines for clicks/crackle, blurry bands)"
        elif nfft == 8192:
            desc = " - High Frequency Resolution (Sharp horizontal lines for hum/tones, blurry timeline)"
        else:
            desc = " - Balanced Resolution (Standard Configuration)"
            
        ax.set_title(f"{title} [NFFT = {nfft}{desc}]", fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(0, 10000)
    
    axes[-1].set_xlabel('Time (seconds)')
    
    # Add Playback controls to the multi-plot window
    ax_play = plt.axes([0.7, 0.02, 0.1, 0.04])
    btn_play = Button(ax_play, 'Play Audio')
    ax_stop = plt.axes([0.81, 0.02, 0.1, 0.04])
    btn_stop = Button(ax_stop, 'Stop')

    def play_audio(event):
        sd.stop()
        sd.play(data, fs)

    def stop_audio(event):
        sd.stop()

    btn_play.on_clicked(play_audio)
    btn_stop.on_clicked(stop_audio)
    
    fig.btn_play = btn_play
    fig.btn_stop = btn_stop

    plt.show()
    sd.stop()

def plot_spectrogram(data, fs, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)
    
    ax.specgram(data, Fs=fs, NFFT=2048, noverlap=1024, cmap='inferno')
    ax.set_title(title)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(0, 10000) 
    
    ax_play = plt.axes([0.7, 0.05, 0.1, 0.075])
    btn_play = Button(ax_play, 'Play Audio')
    ax_stop = plt.axes([0.81, 0.05, 0.1, 0.075])
    btn_stop = Button(ax_stop, 'Stop')

    def play_audio(event):
        sd.stop()
        sd.play(data, fs)

    def stop_audio(event):
        sd.stop()

    btn_play.on_clicked(play_audio)
    btn_stop.on_clicked(stop_audio)
    
    ax.btn_play = btn_play
    ax.btn_stop = btn_stop

    plt.show()
    sd.stop()

# ==========================================
# 2. DSP Filter Functions
# ==========================================
def apply_notch_filter(data, fs, freq=60.0, Q=30.0):
    b, a = signal.iirnotch(freq, Q, fs)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data, b, a

def apply_lowpass_filter(data, fs, cutoff=6000.0, order=5):
    nyq = 0.5 * fs 
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data, b, a

# ==========================================
# 3. Analysis Functions (SNR)
# ==========================================
def calculate_power(signal_array):
    if len(signal_array) == 0:
        return 0
    return np.mean(np.square(signal_array.astype(np.float64)))

def get_snr(signal_section, noise_section):
    p_sig = calculate_power(signal_section)
    p_noise = calculate_power(noise_section)
    if p_noise == 0:
        return float('inf')
    return 10 * np.log10(p_sig / p_noise)

# ==========================================
# 4. Main Pipeline Execution
# ==========================================
if __name__ == "__main__":
    input_filename = 'sp_record_sample.wav' 
    output_filename = 'sp_record_cleaned.wav'
    
    try:
        fs, audio_data = wavfile.read(input_filename)
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
            
        print(f"[SUCCESS] Audio loaded. Sampling Rate: {fs} Hz")

        print("\n[Step 1] Select the 'NOISE-ONLY' profile (e.g., lead-in groove with no music).")
        noise_start, noise_end = select_audio_region(audio_data, fs, "1. Select Noise Profile (Silence/Hiss)")
        
        print("\n[Step 2] Select the 'SIGNAL' profile (clear music section).")
        signal_start, signal_end = select_audio_region(audio_data, fs, "2. Select Signal Profile (Music)")

        noise_profile = audio_data[int(noise_start * fs) : int(noise_end * fs)]
        signal_profile = audio_data[int(signal_start * fs) : int(signal_end * fs)]

        # ---------------------------------------------------------
        # [NEW] Run NFFT Comparison on Original Audio
        # ---------------------------------------------------------
        print("\n[ANALYSIS] Rendering NFFT Resolution Comparison Multi-Plot...")
        plot_nfft_comparison(audio_data, fs, "Original Audio NFFT Comparison")
        # ---------------------------------------------------------

        print("\nRendering Standard Original Spectrogram...")
        plot_spectrogram(audio_data, fs, "Original 78rpm Audio Spectrogram (NFFT=2048)")

        original_snr = get_snr(signal_profile, noise_profile)
        print(f"\n[ANALYSIS] Initial SNR (Before Filtering): {original_snr:.2f} dB")

        print("[PROCESSING] Applying 60Hz IIR Notch Filter...")
        notched_audio, _, _ = apply_notch_filter(audio_data, fs, freq=60.0, Q=30.0)
        
        print("[PROCESSING] Applying Butterworth Low-Pass Filter (6kHz Cutoff)...")
        final_audio, _, _ = apply_lowpass_filter(notched_audio, fs, cutoff=6000.0, order=5)

        filtered_noise = final_audio[int(noise_start * fs) : int(noise_end * fs)]
        filtered_signal = final_audio[int(signal_start * fs) : int(signal_end * fs)]
        final_snr = get_snr(filtered_signal, filtered_noise)

        print("\nRendering Filtered Spectrogram...")
        plot_spectrogram(final_audio, fs, "Filtered Audio Spectrogram (Notch + LPF)")
        
        print(f"\n[RESULTS] Final SNR: {final_snr:.2f} dB")
        print(f"[RESULTS] Total SNR Improvement: {final_snr - original_snr:+.2f} dB")

        final_audio_norm = np.int16(final_audio / np.max(np.abs(final_audio)) * 32767)
        wavfile.write(output_filename, fs, final_audio_norm)
        print(f"\n[SUCCESS] Cleaned audio saved to: {output_filename}\n")
        
    except FileNotFoundError:
        print(f"[ERROR] File '{input_filename}' not found. Please place a .wav file in the current directory.")