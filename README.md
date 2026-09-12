# SP Record Restoration

Restoring degraded 78 RPM records with digital signal processing in Python.

I collect SP records from before 1940 as a hobby, but many have poor sound quality. I wanted to restore the recordings myself and see how much of the original sound I could recover using signal-processing techniques.

I built my own restoration pipeline to experiment with different types of noise and audio degradation.

The project includes:

* Hum removal
* Click and pop detection/restoration
* Broadband noise reduction
* Harmonic analysis and restoration
* Waveform and spectrogram visualization

---

## Processing Steps

```text
78 RPM Recording → Load Audio → Remove Hum → Remove Clicks / Pops → Reduce Background Noise → Restore Harmonics → Save Restored Audio
```

I process the recording one stage at a time so I can compare the result after each step.

---

## 1. Hum Removal

Older recordings can have a low buzzing sound caused by electrical interference. In the U.S., the power-line frequency is 60 Hz, so this noise can show up around 60 Hz and at multiples of 60 Hz, such as 120 Hz and 180 Hz.
```text
60 Hz
120 Hz
180 Hz
240 Hz
300 Hz
...
```

I use `scipy.signal.iirnotch()` to create notch filters for these frequencies.

The quality factor (`Q`) controls how narrow each notch is. I wanted the filter to remove the unwanted frequency without removing too much of the surrounding audio.

I use filtfilt() so that the filtering does not shift the timing of the audio.

---

## 2. Click and Pop Removal

Clicks are different from continuous background noise. They are short, sudden changes in the waveform.

I detect possible clicks by looking at the first derivative of the audio data:

```python
diff = np.diff(self.y) # 1. First derivative
diff_abs = np.abs(diff) # 2. Abs(diff)
```

Large changes in the derivative can indicate a click or pop.

I check the size of the detected region to avoid removing parts of the actual recording.

For short regions, I replace the damaged samples using cubic spline interpolation.

The basic idea is:

```text
Damaged waveform

──────────╲╱──────────
           ↑
         click
           ↓
Interpolated waveform
──────────────────────
```

This does not recover the exact original waveform. It estimates the missing part from the surrounding samples.

---

## 3. Broadband Noise Reduction

There is still a lot of background noise after reducing clicks, pops, and hum.

I assume the background noise is present throughout the recording. I use the beginning of the recording to estimate the background noise spectrum.

The average magnitude of that section is used as a noise profile for each frequency bin.

The basic operation is:

```text
Clean Magnitude = |STFT(audio)| - Noise Profile
```

If the subtraction produces a negative value, I set it to zero.

I keep the original phase and combine it with the modified magnitude before using the inverse STFT to reconstruct the audio.

---

## 4. Harmonic Restoration

Harmonics are frequency components that occur at integer multiples of a fundamental frequency.

For each STFT frame, I:

1. Find spectral peaks.
2. Treat peaks above `peak_threshold=0.15` as possible fundamental frequencies.
3. Check the 2nd to 8th multiples for harmonic components.
4. If the actual amplitude is lower than `peak_amp / n`, I add only the missing amount to `harmonic_map`.
5. Restore only the harmonics above `cutoff_freq`.

The expected harmonic amplitude is modeled as:

```text
Expected Amplitude ≈ Fundamental Amplitude / Harmonic Order
```

I keep the original phase and use the inverse STFT to reconstruct the audio.

This is a simplified model, so this part of the project is still experimental. The goal is to restore some missing high-frequency energy and make the sound clearer and more crisp.


### Looking for Harmonics

For example, if a possible fundamental is 200 Hz, I check for energy around:

```text
200 Hz
400 Hz
600 Hz
800 Hz
...
```

The actual harmonic frequency does not always line up exactly with an STFT frequency bin, so I find the nearest bin and check whether it is close enough to the expected frequency.

The allowed difference is controlled by `tolerance_hz`.

---

### Estimating Missing Harmonics

I estimate the expected amplitude of each harmonic using a simple model:

```text
Expected Amplitude ≈ Fundamental Amplitude / Harmonic Order
```

If the actual amplitude is lower than the expected value, I add only the missing amount to the harmonic map.

This is a simplified model, so this part of the project is still experimental.

---

### Adding the Harmonic Energy

I add the harmonic map to the original STFT magnitude:

```text
Restored Magnitude = Original Magnitude + Harmonic Map × Boost Strength
```

The `boost_strength` parameter controls how much harmonic energy is added.

I keep the original phase and use the inverse STFT to reconstruct the audio.

---

## Why I Used STFT

A normal FFT gives information about which frequencies are present, but it does not tell me when those frequencies occur.

For a historical recording, the frequency content changes over time, so I wanted to see both dimensions.

The STFT gives me:

```text
        Frequency
           ▲
           │
           │   ███
           │ ███████
           │   ███
           └────────────► Time
```

This also makes it easier to compare the recording before and after each restoration step.

---

## Project Structure

The main restoration pipeline is handled by `SPRecordRestorer`, while the harmonic processing is separated into `harmonic_utils`.

### `SPRecordRestorer`

Handles the main audio restoration steps:

```text
load()
save_wav()
plot_waveform_and_spectrogram()
compare_with_original()
compare_with_original2()
remove_hum()
declick()
denoise()
harmonic_restoration()
```

### `harmonic_utils`

Contains the functions used for harmonic analysis and restoration:

```text
harmonic_analysis()
estimate_rolloff()
magnitude_restore_istft()
```
---

## Visualization

I use both waveforms and spectrograms to see what each processing stage is doing.

Depending on the comparison, I can look at:

* Original waveform / Processed waveform
* Original spectrogram / Processed spectrogram
* Waveform difference / Spectrogram difference
---

## Parameters I Am Experimenting With

The harmonic restoration stage is not finished yet, so I am currently experimenting with several parameters:

```text
* `peak_threshold` affects which spectral peaks are considered significant.
* `harmonic_order` controls how many harmonics are checked.
* `tolerance_hz` controls how close a spectral peak needs to be to the expected harmonic frequency.
* `boost_strength` controls how much estimated harmonic energy is added.
```

These values can affect the result significantly, so I don't consider the current settings to be universal.

---

## Limitations

* **Fundamental Detection:** I currently use spectral peaks as possible fundamental frequencies, so incorrect harmonic relationships can occur.
* **Noise Profile:** I use the beginning of the recording to estimate background noise, so this method may not work well for every recording.
* **Harmonic Model:** The current `Amplitude / Harmonic Order` model is simple, so it cannot fully represent the complex harmonic structure of real recordings.
* **Parameter Selection:** Since I do not have the original recording, it is difficult to know which parameter values produce the best restoration. Different records may also require different settings.
* **Possible Artifacts:** Incorrect harmonic estimates or adding too much energy can create sounds that were not in the original recording.

---

## What I Want to Try Next

There are several directions I would like to explore:

* Develop a better harmonic amplitude model
* Improve noise estimation for different recordings
* Automatically tune restoration parameters
* Reduce artifacts caused by harmonic restoration

---

## Technologies

Python, NumPy, SciPy, Librosa, SoundFile, Matplotlib

---

## What I Learned

* **I learned that analyzing and processing a signal in both the time and frequency domains is a basic part of signal processing.** I used STFT to convert the audio into the frequency domain and used it to analyze and process noise and harmonics.
* **Noise reduction is a major part of audio restoration.**
* **Different types of noise require different approaches**, such as clicks, pops, hum, and background noise.
* **Harmonic components are an important part of the sound of a recording**, and restoring missing harmonics is difficult when the original recording is not available. If the harmonics are estimated incorrectly or too much energy is added, it can create **artifacts that were not in the original recording**.

---
