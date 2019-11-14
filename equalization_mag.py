import numpy as np
import matplotlib.pyplot as plt
from scipy import pi
from scipy.fftpack import fft
from scipy.stats import norm
import matplotlib.mlab as mlab

# parameters
sample_rate = 1024
num_seconds = 3
num_samples = (num_seconds - 0) * sample_rate

# x-axis time values
time = np.linspace(0, num_seconds, num_samples)

# main waveform
freq1 = 60
magnitude1 = 1
waveform1 = magnitude1 * np.sin(2 * pi * freq1 * time)

num_noise_sources = 2

########### Double Magnitude
# initialize time domain signal
time_data = waveform1


noise_mag_A = 2 * magnitude1
# additive noise
for i in range(num_noise_sources):
    noise = np.random.normal(0, noise_mag_A, num_samples)
    # combine together
    time_data = time_data + noise

# plot time domain
plt.figure(figsize=(17, 8))
plt.plot(time, time_data)
plt.title('Time Domain Signal Double Magnitude')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# frequency domain
frequency = np.linspace(0.0, sample_rate / 2, int(num_samples / 2))
freq_data = fft(time_data)
freq_mag = 2 / num_samples * np.abs(freq_data[0:np.int(num_samples / 2)])

# plot frequency domain
plt.figure(figsize=(17, 8))
plt.plot(frequency, freq_mag)
plt.title('Frequency domain Signal Double Magnitude')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()


########### 100 Magnitude
# initialize time domain signal
time_data = waveform1

noise_mag_B = 100 * magnitude1
# additive noise
for i in range(num_noise_sources):
    noise = np.random.normal(0, noise_mag_B, num_samples)
    # combine together
    time_data = time_data + noise

# plot time domain
plt.figure(figsize=(17, 8))
plt.plot(time, time_data)
plt.title('Time Domain Signal 100x Magnitude')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# frequency domain
frequency = np.linspace(0.0, sample_rate / 2, int(num_samples / 2))
freq_data = fft(time_data)
freq_mag = 2 / num_samples * np.abs(freq_data[0:np.int(num_samples / 2)])

# plot frequency domain
plt.figure(figsize=(17, 8))
plt.plot(frequency, freq_mag)
plt.title('Frequency domain Signal 100x Magnitude')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()


########### 100 Magnitude Equalized
# initialize time domain signal
time_data = waveform1

noise_mag_B = 100 * magnitude1
# additive noise
for i in range(num_noise_sources):
    noise = np.random.normal(0, noise_mag_B, num_samples)
    # combine together
    time_data = time_data + noise/(np.max(noise))

# plot time domain
plt.figure(figsize=(17, 8))
plt.plot(time, time_data)
plt.title('Time Domain Signal 100x Magnitude Equalized')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# frequency domain
frequency = np.linspace(0.0, sample_rate / 2, int(num_samples / 2))
freq_data = fft(time_data)
freq_mag = 2 / num_samples * np.abs(freq_data[0:np.int(num_samples / 2)])

# plot frequency domain
plt.figure(figsize=(17, 8))
plt.plot(frequency, freq_mag)
plt.title('Frequency domain Signal 100x Magnitude Equalized')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()

