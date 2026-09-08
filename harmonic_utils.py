import numpy as np
import librosa
from scipy.signal import find_peaks


def harmonic_analysis(D, sr, cutoff_freq, peak_threshold=0.15, harmonic_order=8, tolerance_hz=25):# tolerance_hz may need adjustment.
    """
    Parameters
    ----------
    D : complex ndarray 
        STFT result from librosa.stft()
    sr : int
        Sample rate
    peak_threshold : float
        Peak detection threshold (0~1)
    harmonic_order : int
        Maximum harmonic order to search
    tolerance_hz : float
        Allowed frequency error for harmonic matching
    Returns
    -------
    harmonic_map : ndarray
        Harmonic energy map
        shape = magnitude.shape
    """
    # STFT magnitude
    magnitude = np.abs(D) #magnitude.shape = (Number of frequency bins, number of time frames)
    # Frequency axis (0Hz ~ Nyquist)
    freq = np.linspace(0, sr / 2, magnitude.shape[0]) # np.linspace(start, stop, num) returns an array of num evenly spaced values from start to stop.
                                # Divide 0–2250 Hz into as many values as there are frequency bins and return the Hz values.
    # Output map
    harmonic_map = np.zeros_like(magnitude) #Create an array with the same shape as magnitude, filled with zeros

    # Process each frame
    for frame in range(magnitude.shape[1]):
        spectrum = magnitude[:, frame]
        # Normalize for stable peak detection
        spectrum_norm = spectrum / (np.max(spectrum) + 1e-10)  # Normalize to 1; 1e-10 prevents division-by-zero errors.

        # Detect spectral peaks
        peaks, _ = find_peaks(
            spectrum_norm,
            height=peak_threshold  # Only treat peaks above the threshold as significant; smaller peaks may be noise.
        )
        peak_freqs = freq[peaks]       # Frequency values at the detected peak indices.
        peak_amps = spectrum[peaks]    # Amplitudes (same as magnitude).

        # Treat each peak as a possible fundamental
        for i, f0 in enumerate(peak_freqs):  # e.g., [100 Hz, 200 Hz]

            # Ignore very low frequencies
            if f0 < 80:  # Exclude frequencies below 80 Hz as fundamental candidates.
                continue

            # Search harmonic multiples
            for n in range(2, harmonic_order + 1):
                # harmonic_order = 8 based on experience.
                # 5–10 → more natural restoration
                # 15+ → may start following high-frequency noise.

                harmonic_freq = f0 * n

                # Stop above Nyquist
                if harmonic_freq >= sr / 2:  # Limit the search to the Nyquist frequency.
                    break

                # Find the nearest frequency bin
                bin_width = sr / (2 * (magnitude.shape[0] - 1))
                idx = int(round(harmonic_freq / bin_width))
                # The harmonic frequency may not exactly match an existing frequency bin,
                # so find the nearest bin.

                # If close enough, mark as harmonic
                if abs(freq[idx] - harmonic_freq) < tolerance_hz:
                    # If the frequency is within the tolerance range, mark it as a harmonic.
                    # Higher harmonics are usually weaker.
                    # Harmonics should occur at integer multiples of the fundamental frequency.
                    if harmonic_freq > cutoff_freq:
                        current_amp = magnitude[idx, frame]
                        expected_amp = peak_amps[i] / n

                        missing_amp = max(
                            expected_amp - current_amp,
                            0
                        )

                        harmonic_map[idx, frame] += missing_amp

    return harmonic_map #map contains arrays of frequencies and corresponding values.
    '''
    harmonic_map =

        frame0 frame1 frame2
    bin0      0     0     0
    bin1      0     0     0
    bin2      0     0     0
    bin3      0     0     0
    bin4      0     0     0
    '''

def estimate_rolloff( D, sr, min_freq=3000,
                    slope_window=10,
                    threshold=-6):
    """
    Estimate spectral roll-off cutoff frequency from harmonic energy distribution.
    Parameters
    ----------
    D: magnitude
    sr : int            Sampling rate
    min_freq : float    Start searching roll-off above this frequency
    slope_window : int  Number of frequency bins used for slope calculation
    threshold : float   dB drop threshold
    Returns
    -------
    cutoff_freq : float       Estimated roll-off frequency
    rolloff_curve : ndarray   Average harmonic spectrum in dB
    """
    # ------------------------------------
    # 1. Average over time
    # harmonic_map:
    #
    # frequency
    #    |
    #    |
    #    x x x x x   <- frame
    #    x x x x x
    #
    # 1.Calculate the average energy at each frequency
    magnitude = np.abs(D)

    spectrum_avg = np.mean(
        magnitude,
        axis=1
    )

    # 2. Convert to dB
    # Prevent very small values; rolloff_curve.shape = (frequency_bins,) 1D array
    rolloff_curve = 20 * np.log10(spectrum_avg + 1e-10)
    # Add 1e-10 to prevent log(0), which would result in negative infinity.

    # 3. Create the frequency axis: convert indices to frequency values in Hz
    # STFT frequency range: 0 Hz ~ sr/2
    freq = np.linspace(0, sr / 2, len(spectrum_avg))
    # np.linspace(start, stop, num) returns an array of evenly spaced values.
    # Divide the full frequency range into frequency bins.

    cutoff_freq = sr / 2 # max 
    # ------------------------------------
    # 4. Search for the roll-off
    # Find the point where the spectrum slope
    # drops sharply at high frequencies
    # Ex):
    #  |
    #  |\
    #  | \
    #  |  \
    #  |   \_____
    #  |
    #      ^
    #     cutoff
    # ------------------------------------
    for i in range(slope_window, len(freq)):
        # Exclude very low frequencies
        if freq[i] < min_freq:
            continue

        # Calculate local slope : Cur dB -Pre dB
        slope = (rolloff_curve[i] - rolloff_curve[i-slope_window])
      
        # Sharp decrease
        if slope < threshold:
            cutoff_freq = freq[i] 
            break

    print("rolloff min:", np.min(rolloff_curve))
    print("rolloff max:", np.max(rolloff_curve))
    return cutoff_freq, rolloff_curve


def magnitude_restore_istft(D, harmonic_map, boost_strength, length):
    """
    Adjust magnitude → keep phase → iSTFT
    """

    # seperate magnitude and phase 
    magnitude = np.abs(D)
    phase = np.angle(D)

    # adjust harmonic map 
    magnitude_boosted = magnitude + harmonic_map * boost_strength

    # maintain origin phase
    D_boosted = magnitude_boosted * np.exp(1j * phase)

    # iSTFT
    y_restored = librosa.istft(
        D_boosted,
        length=length
    )

    return y_restored # Final data.time-domain audio signal