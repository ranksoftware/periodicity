import numpy as np
import matplotlib.pyplot as plt
from scipy import pi
from scipy.fftpack import fft
from scipy.stats import gamma
import matplotlib.mlab as mlab

# parameters
sample_rate = 1024
num_seconds = 3
num_samples = (num_seconds - 0) * sample_rate

# x-axis time values
time = np.linspace(0, num_seconds, num_samples)

# main waveform
freq1 = 60
magnitude1 = 110
waveform1 = magnitude1 * np.sin(2 * pi * freq1 * time)

# additive noise
noise_mag = 3
noise = np.zeros(num_samples)

no_sparse_size = 20
noise[20:20 + no_sparse_size] = np.random.normal(0, noise_mag, no_sparse_size)

# combine together
time_data = noise

# plot time domain
plt.figure(figsize=(17, 8))
plt.plot(time, time_data)
plt.title('Time Domain Signal')
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
plt.title('Frequency domain Signal')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()

# plot histogram
plt.figure(figsize=(17, 8))

# best fit of data
# (mu, sigma) = norm.fit(freq_mag)
fit_alpha, fit_loc, scale = gamma.fit(freq_mag)
# the histogram of the data
n, bins, patches = plt.hist(freq_mag, bins=30, density=True, facecolor='green', alpha=0.75)
# add a 'best fit' line
# y = mlab.normpdf(bins, mu, sigma)
y = gamma.pdf(bins, a=fit_alpha, loc=fit_loc, scale=scale)
plt.plot(bins, y, 'r--', linewidth=2)

plt.title('Frequency domain Signal Histogram')
plt.xlabel('Fourier Amplitude')
plt.ylabel('Probability')
plt.show()


# Auto-correlation Waveform
autoCorr_Wave = np.correlate(waveform1, waveform1, 'same')
plt.figure(figsize=(17, 8))
plt.plot(range(int(-num_samples/2), int(num_samples/2), 1), autoCorr_Wave)
plt.title('ACF Signal')
plt.xlabel('Time shift')
plt.ylabel('ACF Amplitude')
plt.show()

# Auto-correlation Waveform
autoCorr_Pulse = np.correlate(time_data, time_data, 'same')
plt.figure(figsize=(17, 8))
plt.plot(range(int(-num_samples/2), int(num_samples/2), 1), autoCorr_Pulse)
plt.title('ACF Pulse')
plt.xlabel('Time shift')
plt.ylabel('ACF Amplitude')
plt.show()
