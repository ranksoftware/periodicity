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
magnitude1 = 2
waveform1 = magnitude1 * np.sin(2 * pi * freq1 * time)


########### 2 Noise Sources
# initialize time domain signal
time_data = waveform1

num_noise_sources_A = 1
noise_mag = magnitude1
# additive noise
for i in range(num_noise_sources_A):
    noise = np.random.normal(0, noise_mag, num_samples)
    # combine together
    time_data = time_data + noise

# plot time domain
plt.figure(figsize=(17, 8))
plt.plot(time, time_data)
plt.title('Time Domain Signal 2 Noise Source')
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
plt.title('Frequency domain Signal 2 Noise Source')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()


########### 20 Noise Sources
# initialize time domain signal
time_data = waveform1

num_noise_sources_B = 20
noise_mag = magnitude1
# additive noise
for i in range(num_noise_sources_B):
    noise = np.random.normal(0, noise_mag, num_samples)
    # combine together
    time_data = time_data + noise

# plot time domain
plt.figure(figsize=(17, 8))
plt.plot(time, time_data)
plt.title('Time Domain Signal 20 Noise Source')
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
plt.title('Frequency domain Signal 20 Noise Source')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()


########### 200 Noise Sources
# initialize time domain signal
time_data = waveform1

num_noise_sources_C = 200
noise_mag = magnitude1
# additive noise
for i in range(num_noise_sources_C):
    noise = np.random.normal(0, noise_mag, num_samples)
    # combine together
    time_data = time_data + noise

# plot time domain
plt.figure(figsize=(17, 8))
plt.plot(time, time_data)
plt.title('Time Domain Signal 200 Noise Source')
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
plt.title('Frequency domain Signal 200 Noise Source')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()
