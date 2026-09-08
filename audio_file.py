import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
import soundfile as sf
import os
import harmonic_utils as hu


class SPRecordRestorer:
    def __init__(self, filepath):
        # Store the file path and load the audio right away
        self.filepath = filepath
        self.sr = None           # sr = sample rate 
        self.y_original = None   # keep the raw audio untouched, for comparison
        self.y = None            # this will be the working copy we process

    def load(self):
        # Step 1: load audio, keep original sample rate
        self.y_original, self.sr = librosa.load(self.filepath, sr=None) # Return value is Numpy
        self.y = self.y_original.copy()
        print(f"Sample rate: {self.sr} Hz")
        print(f"Duration: {len(self.y)/self.sr:.2f} sec")

        return self #data, data_copy return.
    
    def save_wav(self, y, filename, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, y, self.sr)
        print(f"Saved: {filepath}")
        return self
    
    # Show graph 
    def plot_waveform_and_spectrogram(self, title_suffix=""):
        
        # Step 2: visualize current state of self.y
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
     
        librosa.display.waveshow(self.y, sr=self.sr, ax=axes[0])
        axes[0].set_title(f"Waveform {title_suffix}")

        #STFT(short time Fourier transform) magnitude to dB scale
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max) #ref=np.max: set 0 dB is max magnitude(reference)
        # A spectrogram shows how strong each frequency is, so we only need the magnitude.
        # 0 dB represents the strongest component.
        # librosa.amplitude_to_db() converts linear amplitude to logarithmic dB using 20*log10(x).
        '''
        20*log10(x): a 6 dB increase means approximately twice the amplitude.
            0 dB   → strongest component
            -20 dB → 10× weaker
            -40 dB → 100× weaker
        '''

        img = librosa.display.specshow(D, sr=self.sr, x_axis="time", y_axis="log", ax=axes[1])
        axes[1].set_title(f"Spectrogram {title_suffix}")
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
        axes[1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        plt.tight_layout()
        plt.show()
        return self

    #Graph function for comparison with the original
    def compare_with_original(self, y_processed, method_name=""):
        """
        Compare the original signal (first loaded) with y_processed (result after a specific process)
        side by side using waveforms and spectrograms.

        Parameters
        ----------
        y_processed : ndarray
            The processed signal to compare (e.g., result of hum removal, rumble removal, etc.).
        method_name : str
            The processing method name to include in the plot title (e.g., "Hum Removal").
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # --- Waveform ---
        librosa.display.waveshow(self.y_original, sr=self.sr, ax=axes[0, 0])
        axes[0, 0].set_title("Waveform - Original")

        librosa.display.waveshow(y_processed, sr=self.sr, ax=axes[0, 1])
        axes[0, 1].set_title(f"Waveform - After {method_name}")

        # --- Spectrogram ---
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(self.y_original)), ref=np.max)
        D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(y_processed)), ref=np.max)

        img1 = librosa.display.specshow(D_orig, sr=self.sr, x_axis="time", y_axis="log", ax=axes[1, 0])
        axes[1, 0].set_title("Spectrogram - Original")
        fig.colorbar(img1, ax=axes[1, 0], format="%+2.0f dB")

        img2 = librosa.display.specshow(D_proc, sr=self.sr, x_axis="time", y_axis="log", ax=axes[1, 1])
        axes[1, 1].set_title(f"Spectrogram - After {method_name}")
        fig.colorbar(img2, ax=axes[1, 1], format="%+2.0f dB")

        for ax in (axes[1, 0], axes[1, 1]):
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

        plt.tight_layout()
        plt.show()
        return self

    def compare_with_original2(self, y_processed, method_name=""):
        """
        self.y_original(최초 로드한 원본)과 y_processed(특정 처리 후 결과)를
        waveform + spectrogram + diff로 비교

        Parameters
        ----------
        y_processed : ndarray   비교하고 싶은 처리 결과 (예: remove_hum 결과, remove_rumble 결과 등)
        method_name : str       그래프 제목에 붙일 처리 이름 (예: "Hum Removal")
        """
        #--- Length safeguard code: STFT/ISTFT processed results may differ in length from the original ---
        min_len = min(len(self.y_original), len(y_processed))
        y_orig = self.y_original[:min_len]
        y_proc = y_processed[:min_len]

        if len(self.y_original) != len(y_processed):
            print(f"Different lengths- original: {len(self.y_original)}, processed: {len(y_processed)} → {min_len}<-Clip")

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # --- Waveform ---
        librosa.display.waveshow(y_orig, sr=self.sr, ax=axes[0, 0])
        axes[0, 0].set_title("Waveform - Original")

        librosa.display.waveshow(y_proc, sr=self.sr, ax=axes[0, 1])
        axes[0, 1].set_title(f"Waveform - After {method_name}")

        # --- Spectrogram ---
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig)), ref=np.max)
        D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(y_proc)), ref=np.max)

        img1 = librosa.display.specshow(D_orig, sr=self.sr, x_axis="time", y_axis="log", ax=axes[1, 0])
        axes[1, 0].set_title("Spectrogram - Original")
        fig.colorbar(img1, ax=axes[1, 0], format="%+2.0f dB")

        img2 = librosa.display.specshow(D_proc, sr=self.sr, x_axis="time", y_axis="log", ax=axes[1, 1])
        axes[1, 1].set_title(f"Spectrogram - After {method_name}")
        fig.colorbar(img2, ax=axes[1, 1], format="%+2.0f dB")

        # --- Waveform diff (Original - processed = removed component) ---
        diff_wave = y_orig - y_proc
        t = np.linspace(0, len(diff_wave) / self.sr, len(diff_wave))
        axes[2, 0].plot(t, diff_wave, color="purple", linewidth=0.5)
        axes[2, 0].set_title(f"Waveform Diff - Removed by {method_name}")

        # --- Spectrogram diff (Original dB - processed dB; positive values = removed part) ---
        min_frames = min(D_orig.shape[1], D_proc.shape[1])
        D_diff = D_orig[:, :min_frames] - D_proc[:, :min_frames]
        vmax = np.max(np.abs(D_diff))
        img3 = librosa.display.specshow(D_diff, sr=self.sr, x_axis="time", y_axis="log",
                                        ax=axes[2, 1], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[2, 1].set_title(f"Spectrogram Diff - Removed by {method_name}")
        fig.colorbar(img3, ax=axes[2, 1], format="%+2.0f dB")

        for ax in (axes[1, 0], axes[1, 1], axes[2, 1]):
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

        plt.tight_layout()
        plt.show()
        return self

    def remove_hum(self, freq=60, harmonics=5, quality=30):
        """
        y: Audio signal (NumPy array)
        sr: Sampling rate
        freq: Fundamental hum frequency (60 Hz for power systems in Korea/US)
        harmonics: Number of harmonics to remove (60, 120, 180, 240, 300 Hz, ...)
        quality: Q factor that determines the notch filter bandwidth (higher = narrower and more precise attenuation)
        """
        nyquist = self.sr / 2 #ex) audio recorded at 44.1 kHz can represent frequencies up to 22.05 kHz.

        for h in range(1, harmonics + 1):
            f = freq * h              # 1st harmonic = 60 Hz, 2nd harmonic = 120 Hz, 3rd harmonic = 180 Hz, ...
            if f >= nyquist:          # Stop because frequencies above the Nyquist frequency cannot be represented.
                break

            b, a = signal.iirnotch(f, quality, self.sr)
            # Infinite Impulse Response (IIR) notch filter.
            # Calculates the filter coefficients to narrowly attenuate the target frequency.
            # b → numerator coefficients / a → denominator coefficients
            # quality = f / desired bandwidth (e.g., 2, 3, 4 Hz)
            # Higher Q → narrower bandwidth; lower Q → wider bandwidth.

            self.y = signal.filtfilt(b, a, self.y)  # Apply the filter to the actual audio signal.

            # b: numerator coefficients [1, -2cos(w0), 1]
            # a: denominator coefficients [1, -2r cos(w0), r²]

        return self.y

    def declick(self, threshold=0.3, max_width=5):
        """
        Click / Pop Removal : Removes sudden large spikes in the middle of the signal.
        threshold: Threshold based on the first derivative
                (0-1 for normalized audio)
        max_width: Maximum number of samples considered as a click
        """
        
        diff = np.diff(self.y) # 1. First derivative
        diff_abs = np.abs(diff) # 2. abs(diff)

        # 3. Threshold
        # Find points with changes larger than the threshold and mark them as True.
        threshold_value = threshold * np.max(diff_abs)
        mask = diff_abs > threshold_value  
        # Compare each value with threshold_value: True if larger, False otherwise.

        # 4. Find True regions – detect where sudden spikes occur in the audio.
        click_regions = []
        start = None

        for i, value in enumerate(mask):
            if value and start is None:  # Start of a True region
                start = i
            elif not value and start is not None:  # End of a True region
                end = i
                width = end - start  # Length of the region
                if width <= max_width:  # Only short spikes are considered click noise.
                    click_regions.append((start, end))  # Store the detected click region.
                start = None

        # Handle the last region if it extends to the end of the signal.
        if start is not None:
            end = len(mask)
            width = end - start

            if width <= max_width:
                click_regions.append((start, end))

        print("Detected clicks:", len(click_regions))

        # 5. Restore CubicSpline 
        for start, end in click_regions:
            # Actual audio indices
            left = max(0, start - 5)
            right = min(len(self.y), end + 5)  # 5 good samples on the left |-- click --| 5 good samples on the right

            # Select the good data
            x_good = np.concatenate(
                [
                    # np.arange() creates an array of evenly spaced integers.
                    # np.arange(0, 5) → array([0, 1, 2, 3, 4])
                    np.arange(left, start),  # Get the indices and concatenate them.
                    np.arange(end, right)
                ]
            )

            y_good = np.concatenate(
                [
                    self.y[left:start],  # Get the corresponding audio values.
                    self.y[end:right]
                ]
            )

            # Skip if there is not enough data
            if len(x_good) < 4:
                continue

            # Create a cubic spline
            spline = CubicSpline(x_good, y_good)  # Estimate the click region using only the good data.

            # Indices of the region to be restored
            x_bad = np.arange(start, end)  # x_bad contains the indices of the click samples.

            # Generate replacement values
            self.y[x_bad] = spline(x_bad)  # Evaluate the spline at x_bad to generate replacement values.
            # NumPy allows multiple indices and values to be processed at once,
            # instead of using a Python for loop.
        
        return self.y

    def denoise(self, noise_duration=1.0):  # Remove broadband noise, such as continuous "hiss" in the background.

        # STFT
        D = librosa.stft(self.y)

        '''
        librosa.stft() default parameters:
        n_fft = 2048      # Number of samples analyzed in each frame.
        hop_length = 512  # Number of samples between consecutive frames.
        win_length = None # Defaults to n_fft (2048).
        '''

        magnitude = np.abs(D)    # Magnitude
        phase = np.angle(D)      # Phase

        # Assume the first 1 second contains only noise.
        hop_length = 512  # Default hop length used by librosa.stft().
        noise_frames = int(self.sr / hop_length)  # Number of STFT frames in 1 second.
                                                # Approximately 86 frames at 44.1 kHz.
        noise_frames = max(1, min(noise_frames, magnitude.shape[1]))
        # Limit the number of noise frames to at least 1 and at most the total number of frames.

        noise_profile = np.mean(
            magnitude[:, :noise_frames],  # All frequencies from the first 86 time frames.
            axis=1,                       # Average across the time axis for each frequency.
            keepdims=True                 # Keep the result as a 2D array: [frequency, mean magnitude].
        )
        '''
                  time frame
                  0  1  2 ... 85

        freq 0    x  x  x ... x
        freq 1    x  x  x ... x
        freq 2    x  x  x ... x
        ...
        freq1024  x  x  x ... x
        '''

        # Spectral subtraction
        clean_mag = magnitude - noise_profile  # Subtract the average noise magnitude from the original magnitude.

        # Prevent negative values
        clean_mag = np.maximum(clean_mag, 0)  # Clip negative values to 0.

        # Reconstruct the complex spectrogram
        D_clean = clean_mag * np.exp(1j * phase)  # Keep the original phase and apply the cleaned magnitude.

        # iSTFT
        y_clean = librosa.istft(D_clean)  # Convert the cleaned spectrogram back to a time-domain audio signal.
        self.y = y_clean
        return self.y
    
    def harmonic_restoration(self, n_fft=4096, hop_length=1024,
                            peak_threshold=0.15, harmonic_order=6,
                            boost_strength=1.0):
        y = self.y

        # ==================================================
        # 1. STFT
        # ==================================================
        '''
        hop_length = 1024 → Time step between frames.
        1024 / 44100 ≈ 0.02322 sec (23 ms)

        n_fft = 4096 → Number of samples analyzed in each frame.

        Custom STFT parameters are used instead of the defaults
        to observe the frequency components in greater detail.

        Overlap = 1 - (hop_length / n_fft)
                = 1 - (1024 / 4096)
                = 75%
        '''

        D = librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length
        )

        magnitude = np.abs(D)
        phase = np.angle(D)

        # ==================================================
        # 2. Estimate the roll-off frequency
        # ==================================================
        cutoff_freq, rolloff_curve = hu.estimate_rolloff(D, self.sr)
        print(f"Estimated roll-off: {cutoff_freq:.0f} Hz")

        # ==================================================
        # 3. Generate a harmonic map
        # ==================================================
        harmonic_map = hu.harmonic_analysis(
            D, self.sr,
            cutoff_freq=cutoff_freq,
            peak_threshold=peak_threshold,
            harmonic_order=harmonic_order
        )
       
        # ==================================================
        # 4. iSTFT
        # ==================================================
        y_restored = hu.magnitude_restore_istft(D, harmonic_map, boost_strength, length=len(y))

        self.y = y_restored

        return self.y

#######################################
    
# Usage example — this reads like the project pipeline itself
restorer = SPRecordRestorer("./audio/sample2.wav")
restorer.load()
#restorer.plot_waveform_and_spectrogram(title_suffix="(before)")
''' 
#hum
y_hum_removed = restorer.remove_hum()
restorer.compare_with_original2(y_hum_removed, method_name="Hum Removal")
restorer.save_wav(y_hum_removed, "hum_removed.wav")

#declick
y_declicked = restorer.declick(threshold=0.3, max_width=5)   # self.y -> declick result
restorer.compare_with_original2(y_declicked, method_name="Declick")
restorer.save_wav(y_declicked, "declicked.wav")

#denoise
y_denoised = restorer.denoise()
restorer.compare_with_original2(y_denoised, method_name="denoise")
restorer.save_wav(y_denoised, "denoise.wav")

#harmonic_restoration
y_restored = restorer.harmonic_restoration()
restorer.compare_with_original2(y_restored, method_name="Harmonic Restoration")
restorer.save_wav(y_restored, "harmonic.wav")
'''
y_hum_removed = restorer.remove_hum()
y_declicked = restorer.declick(threshold=0.3, max_width=5)
y_denoised = restorer.denoise()
y_restored = restorer.harmonic_restoration()
#restorer.compare_with_original2(y_restored, method_name="Full Process")
restorer.save_wav(y_restored, "Final.wav")